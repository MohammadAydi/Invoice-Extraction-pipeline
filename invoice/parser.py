"""Turn a StructuredDocument into an invoice.

Strategy, in two halves:

* **Header** -- group the fragments above the table into lines, classify each
  line by the keywords and number patterns it carries, and read the labelled
  fields off it.
* **Line items** -- prefer the table extractor's grid: when it found cells,
  each grid row is a line item and each cell is a column, which is far more
  reliable than inferring columns from x-positions. Fall back to line grouping
  when no grid was detected, so a borderless receipt still produces rows.

Every value carries a confidence and a ranked candidate list, and anything
uncertain is surfaced as a warning rather than silently guessed -- the desktop
app's verification screen exists precisely because this stage cannot be perfect.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from core.domain.document import StructuredDocument, TableCellElement
from core.domain.geometry import BoundingBox
from core.domain.ocr import OCRFragment, OCRResult
from invoice.models import (
    REVIEW_CONFIDENCE_THRESHOLD,
    ExtractedValue,
    InvoiceDraft,
    InvoiceHeader,
    InvoiceLineItem,
    InvoiceWarning,
    WarningCodes,
)
from invoice.text_lines import TextLine, Word, group_lines
from string_matching.algorithms import DEFAULT_TOP_K
from string_matching.catalog import (
    CatalogEntry,
    match_city,
    match_merchant,
    match_product,
)
from string_matching.normalization import (
    format_date,
    normalize_quantity,
    normalize_text,
    parse_date,
    parse_number_strict,
)

# Label keywords, Arabic and English. Matched against normalized text, so they
# must themselves be written in normalized form -- no diacritics, folded letters.
_MERCHANT_HINTS = (
    "مؤسسه", "موسسه", "شركه", "متجر", "محل", "سوبرماركت", "معرض",
    "est", "trading", "co", "ltd", "store",
)
_INVOICE_NO_HINTS = (
    "رقم الفاتوره", "فاتوره رقم", "رقم", "invoice no", "invoice #", "inv no",
    "bill no", "no.",
)
_DATE_HINTS = ("تاريخ", "التاريخ", "date", "dated")
_CITY_HINTS = ("مدينه", "المدينه", "محافظه", "المحافظه", "city", "location", "فرع")
_TOTAL_HINTS = (
    "الاجمالي", "الاجمالى", "اجمالي", "المجموع", "الصافي", "total", "grand total",
    "amount due", "net",
)
_SUBTOTAL_HINTS = ("المجموع الفرعي", "subtotal", "sub total")
_TAX_HINTS = ("ضريبه", "الضريبه", "vat", "tax")
_COLUMN_HEADER_HINTS = (
    "الصنف", "البيان", "الوصف", "الكميه", "العدد", "السعر", "المبلغ",
    "item", "description", "qty", "quantity", "price", "unit", "amount",
)

# Paper invoices round each line to two decimals, so a cent of slack is right.
_ARITHMETIC_TOLERANCE = 0.01

# Sum-of-lines vs stated total, which accumulates per-line rounding.
_TOTAL_TOLERANCE = 0.05


@dataclass
class Catalogs:
    """The match targets the desktop app sends with each request.

    Empty lists are normal and simply disable matching for that field, leaving
    the raw OCR text and an empty candidate array.
    """

    merchants: Sequence[CatalogEntry] = field(default_factory=list)
    products: Sequence[CatalogEntry] = field(default_factory=list)
    cities: Sequence[CatalogEntry] = field(default_factory=list)

    # How many ranked alternatives each matched field carries back.
    top_k: int = DEFAULT_TOP_K


class InvoiceParser:
    """Reads an invoice out of a StructuredDocument. Stateless and reusable."""

    def parse(
        self,
        document: StructuredDocument,
        ocr_result: OCRResult,
        catalogs: Catalogs | None = None,
    ) -> InvoiceDraft:
        catalogs = catalogs or Catalogs()
        draft = InvoiceDraft()

        cells = [el for el in document.elements if isinstance(el, TableCellElement)]
        lines = group_lines(ocr_result.fragments)

        if not lines and not cells:
            draft.warnings.append(
                InvoiceWarning(
                    code=WarningCodes.NO_LINE_ITEMS,
                    message="No text was recognized in this document.",
                )
            )
            return draft

        table_top = _table_top(cells)

        # Header lines are the ones above the grid. Without a grid every line is
        # a header candidate and the table-start heuristic sorts it out below.
        header_lines = (
            [line for line in lines if line.top < table_top]
            if table_top is not None
            else lines
        )

        draft.header = self._parse_header(header_lines or lines, catalogs)

        # The grid is the better source when there is one, but a degenerate
        # grid -- a couple of cells detected off a printed border, none of them
        # holding an item row -- must not cost the whole table. Falling back to
        # line grouping recovers the rows a borderless read would have found
        # anyway, and returning nothing is strictly worse than trying.
        draft.line_items = self._items_from_cells(cells, catalogs) if cells else []
        if not draft.line_items:
            draft.line_items = self._items_from_lines(lines, catalogs)

        _add_warnings(draft)
        return draft

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _parse_header(self, lines: list[TextLine], catalogs: Catalogs) -> InvoiceHeader:
        return InvoiceHeader(
            merchant_name=self._extract_merchant(lines, catalogs.merchants, catalogs.top_k),
            invoice_number=_extract_invoice_number(lines),
            invoice_date=_extract_date(lines),
            city=self._extract_city(lines, catalogs.cities, catalogs.top_k),
            total_amount=_extract_total(lines),
        )

    def _extract_merchant(
        self,
        lines: list[TextLine],
        merchants: Sequence[CatalogEntry],
        top_k: int = DEFAULT_TOP_K,
    ) -> ExtractedValue:
        """Merchant name: a catalog match anywhere beats a positional guess.

        Matching is by record, so a letterhead printed with a customer's alias
        resolves to that customer exactly as the canonical name would, and the
        value returned is the canonical name either way.
        """
        best_score = -1.0
        best: ExtractedValue | None = None

        # Pass 1: catalog match against word runs within the top lines. Run
        # matching keeps a merged "merchant ... invoice no" line usable.
        for line in lines[:8]:
            for segment in line.segments():
                match = match_merchant(segment, merchants, top_k)
                if match.best is None or match.best.similarity_score <= best_score:
                    continue

                best_score = match.best.similarity_score
                best = ExtractedValue.from_match(
                    match,
                    fallback_text=segment,
                    confidence=line.confidence,
                    bbox=line.bbox,
                )

        if best is not None and not best.requires_manual_review:
            return best

        # Pass 2: a line carrying a business-type keyword ("متجر", "Est.", ...).
        positional = _first_merchant_by_shape(lines)

        # Keep the ranked candidates from pass 1 even when nothing cleared the
        # threshold: the dropdown is exactly what lets the user pick the right
        # customer for a letterhead OCR read badly.
        if positional is not None and best is not None:
            positional.candidates = best.candidates
            positional.match_score = best.match_score
            positional.requires_manual_review = True

        return positional or best or ExtractedValue()

    def _extract_city(
        self,
        lines: list[TextLine],
        cities: Sequence[CatalogEntry],
        top_k: int = DEFAULT_TOP_K,
    ) -> ExtractedValue:
        raw = _extract_city_text(lines)
        if raw is None:
            return ExtractedValue()

        text, confidence, bbox = raw
        match = match_city(text, cities, top_k)

        if not match.candidates:
            # No city catalog was sent: keep what was read, at the confidence
            # the label-adjacency heuristic earns.
            return ExtractedValue(value=text, confidence=confidence, bbox=bbox)

        return ExtractedValue.from_match(
            match, fallback_text=text, confidence=confidence, bbox=bbox
        )

    # ------------------------------------------------------------------
    # Line items, from the table grid
    # ------------------------------------------------------------------

    def _items_from_cells(
        self, cells: list[TableCellElement], catalogs: Catalogs
    ) -> list[InvoiceLineItem]:
        rows: dict[tuple[str, int], list[TableCellElement]] = {}
        for cell in cells:
            rows.setdefault((cell.table_id, cell.row), []).append(cell)

        # Grid rows are numbered top to bottom by the extractor; sorting by the
        # row index keeps the invoice in the order it was printed.
        ordered_rows = [rows[key] for key in sorted(rows, key=lambda k: (k[0], k[1]))]

        items: list[InvoiceLineItem] = []
        for row in ordered_rows:
            if _is_column_header_row(row):
                continue

            item = self._item_from_row(row, len(items), catalogs)
            if item is not None:
                items.append(item)

        return items

    def _item_from_row(
        self,
        row: list[TableCellElement],
        row_index: int,
        catalogs: Catalogs,
    ) -> InvoiceLineItem | None:
        """One grid row: the wordiest cell is the product, the rest are numbers."""
        populated = [cell for cell in row if cell.merged_text.strip()]
        if not populated:
            return None

        numeric: list[tuple[TableCellElement, float]] = []
        textual: list[TableCellElement] = []

        for cell in populated:
            value = parse_number_strict(cell.merged_text)
            if value is None:
                textual.append(cell)
            else:
                numeric.append((cell, value))

        if not textual or not numeric:
            # A row with no description or no figures is a title, a note or a
            # totals line -- not an item.
            return None

        # The description is the cell with the most letters; a stray "1" that
        # failed strict parsing must not win the product column.
        product_cell = max(textual, key=lambda c: len(normalize_text(c.merged_text)))

        # Left to right, which is the order quantity/price/total is printed in
        # even on an Arabic form.
        numeric.sort(key=lambda pair: pair[0].bbox.x)
        columns = _assign_columns([value for _, value in numeric])

        item = InvoiceLineItem(row_index=row_index)

        item.product_name = ExtractedValue.from_match(
            match_product(product_cell.merged_text, catalogs.products, catalogs.top_k),
            fallback_text=product_cell.merged_text,
            confidence=_cell_confidence([product_cell]),
            bbox=product_cell.bbox,
        )

        texts = [cell.merged_text for cell, _ in numeric]
        values = [value for _, value in numeric]
        boxes = [cell.bbox for cell, _ in numeric]
        confidences = [_cell_confidence([cell]) for cell, _ in numeric]

        item.quantity = _quantity_field(columns.quantity_index, texts, boxes, confidences)
        item.unit_price = _price_field(columns.unit_price_index, values, boxes, confidences)
        item.total_price = _price_field(
            columns.total_index, values, boxes, confidences, derived=columns.derived_total
        )

        item.arithmetic_ok = _arithmetic_ok(
            item.quantity.value, item.unit_price.value, item.total_price.value
        )
        return item

    # ------------------------------------------------------------------
    # Line items, from reconstructed lines
    # ------------------------------------------------------------------

    def _items_from_lines(
        self, lines: list[TextLine], catalogs: Catalogs
    ) -> list[InvoiceLineItem]:
        start = _find_table_start(lines)
        end = _find_table_end(lines, start)

        items: list[InvoiceLineItem] = []
        for line in lines[start:end]:
            item = self._item_from_line(line, len(items), catalogs)
            if item is not None:
                items.append(item)

        return items

    def _item_from_line(
        self, line: TextLine, row_index: int, catalogs: Catalogs
    ) -> InvoiceLineItem | None:
        number_words = [w for w in line.words if parse_number_strict(w.text) is not None]
        number_words.sort(key=lambda w: w.center_x)

        # A real item row carries at least a quantity and a price.
        if len(number_words) < 2:
            return None

        name_words = [w for w in line.words if parse_number_strict(w.text) is None]
        product_text = " ".join(
            w.text for w in sorted(name_words, key=lambda w: -w.center_x)
        ).strip()

        if not product_text:
            return None

        values = [parse_number_strict(w.text) or 0.0 for w in number_words]
        columns = _assign_columns(values)

        item = InvoiceLineItem(row_index=row_index)

        item.product_name = ExtractedValue.from_match(
            match_product(product_text, catalogs.products, catalogs.top_k),
            fallback_text=product_text,
            confidence=line.confidence,
            bbox=_words_bbox(name_words),
        )

        texts = [w.text for w in number_words]
        boxes = [w.bbox for w in number_words]
        confidences = [w.confidence for w in number_words]

        item.quantity = _quantity_field(columns.quantity_index, texts, boxes, confidences)
        item.unit_price = _price_field(columns.unit_price_index, values, boxes, confidences)
        item.total_price = _price_field(
            columns.total_index, values, boxes, confidences, derived=columns.derived_total
        )

        item.arithmetic_ok = _arithmetic_ok(
            item.quantity.value, item.unit_price.value, item.total_price.value
        )
        return item


# --------------------------------------------------------------------------
# Header field helpers
# --------------------------------------------------------------------------


def _first_merchant_by_shape(lines: list[TextLine]) -> ExtractedValue | None:
    for line in lines[:6]:
        if line.has_any(_MERCHANT_HINTS):
            return ExtractedValue(
                value=line.text,
                confidence=line.confidence * 0.9,
                bbox=line.bbox,
            )

    # The topmost line with no digits: letterheads put the name first.
    for line in lines[:4]:
        if not any(ch.isdigit() for ch in line.text) and len(line.text.strip()) >= 3:
            return ExtractedValue(
                value=line.text,
                # Purely positional, so scored low enough to land under the
                # review threshold and prompt a human check.
                confidence=line.confidence * 0.6,
                bbox=line.bbox,
            )

    return None


def _extract_invoice_number(lines: list[TextLine]) -> ExtractedValue:
    for line in lines[:12]:
        if not line.has_any(_INVOICE_NO_HINTS):
            continue

        # Prefer an alphanumeric token like "INV-2291" over a bare number.
        token = re.search(r"\b([A-Za-z]{2,}[-/]?\d{2,}|\d{3,})\b", line.ltr_text)
        if token:
            return ExtractedValue(
                value=token.group(1),
                confidence=line.confidence,
                raw=line.text,
                bbox=line.bbox,
            )

    return ExtractedValue()


def _extract_date(lines: list[TextLine]) -> ExtractedValue:
    # Prefer a line that says "date"; fall back to any parseable date near the top.
    for candidates, penalty in (
        ([line for line in lines if line.has_any(_DATE_HINTS)], 1.0),
        (lines[:15], 0.8),
    ):
        for line in candidates:
            parsed = parse_date(line.ltr_text)
            if parsed:
                return ExtractedValue(
                    value=format_date(parsed),
                    confidence=line.confidence * penalty,
                    raw=line.text,
                    bbox=line.bbox,
                )

    return ExtractedValue()


def _extract_city_text(
    lines: list[TextLine],
) -> tuple[str, float, BoundingBox | None] | None:
    """City text read from the word run adjacent to a city label.

    Works at word level rather than on the whole line: invoice headers routinely
    put "المدينة: عمّان" and "التاريخ: 14/03/2026" on the same physical line,
    and splitting the line on ':' would carry the date into the city value.
    """
    for line in lines[:12]:
        # The value may sit on either side of the label depending on whether the
        # line reads Arabic or Latin, so both directions are tried.
        words = sorted(line.words, key=lambda w: w.center_x)

        label_index = next(
            (
                i
                for i, w in enumerate(words)
                if any(hint in normalize_text(w.text) for hint in _CITY_HINTS)
            ),
            None,
        )
        if label_index is None:
            continue

        after = _collect_value_words(words[label_index + 1 :])
        # Scanning leftward, the word nearest the label comes first.
        before = _collect_value_words(list(reversed(words[:label_index])))

        text = (after or before).strip().lstrip(":：").strip()
        if text:
            return text, line.confidence * 0.85, line.bbox

    return None


def _collect_value_words(words: list[Word]) -> str:
    """Take words up to the first token belonging to a different field.

    Stops at any digit-bearing token (a date, phone or amount) and at any other
    field label, so "المدينة: عمّان   التاريخ: 14/03/2026" yields just "عمّان".
    """
    collected: list[str] = []

    for word in words:
        if any(ch.isdigit() for ch in word.text):
            break

        normalized = normalize_text(word.text)
        if any(
            hint in normalized
            for hints in (_DATE_HINTS, _INVOICE_NO_HINTS, _TOTAL_HINTS, _CITY_HINTS)
            for hint in hints
        ):
            break

        collected.append(word.text)

    return " ".join(collected).strip()


def _extract_total(lines: list[TextLine]) -> ExtractedValue:
    """Grand total: prefer an explicit total label, scanning bottom-up."""
    for line in reversed(lines):
        if not line.has_any(_TOTAL_HINTS):
            continue
        # "Subtotal" and "VAT" lines contain total-ish words but are not it.
        if line.has_any(_SUBTOTAL_HINTS) or line.has_any(_TAX_HINTS):
            continue

        numbers = line.numbers()
        if numbers:
            # The largest number on a totals line is the total; a tax rate or a
            # line count may sit beside it.
            return ExtractedValue(
                value=max(numbers),
                confidence=line.confidence,
                raw=line.text,
                bbox=line.bbox,
            )

    return ExtractedValue()


# --------------------------------------------------------------------------
# Shared row helpers
# --------------------------------------------------------------------------


def _table_top(cells: list[TableCellElement]) -> float | None:
    return min((cell.bbox.y for cell in cells), default=None)


def _is_column_header_row(row: list[TableCellElement]) -> bool:
    """True for the printed "الصنف | الكمية | السعر" row above the items."""
    joined = " ".join(normalize_text(cell.merged_text) for cell in row)
    return sum(1 for hint in _COLUMN_HEADER_HINTS if hint in joined) >= 2


def _cell_confidence(cells: Iterable[TableCellElement]) -> float:
    fragments: list[OCRFragment] = [f for cell in cells for f in cell.fragments]
    scored = [
        float(f.confidence) for f in fragments if f.confidence is not None
    ]
    if not scored:
        # Cells the engine boxed but could not read, or a detector-only engine
        # that reports no score. Neither is "confident".
        return 0.5 if fragments else 0.0
    return sum(scored) / len(scored)


def _words_bbox(words: list[Word]) -> BoundingBox | None:
    if not words:
        return None
    x1 = min(w.bbox.x for w in words)
    y1 = min(w.bbox.y for w in words)
    x2 = max(w.bbox.x + w.bbox.w for w in words)
    y2 = max(w.bbox.y + w.bbox.h for w in words)
    return BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)


def _quantity_field(
    index: int | None,
    texts: list[str],
    boxes: list[BoundingBox],
    confidences: list[float],
) -> ExtractedValue:
    """The quantity cell, re-read under the integer rule.

    Deliberately parsed from the cell *text* rather than from the float used
    for column assignment: a quantity is a whole number, and a separator inside
    one is a mis-read stroke, not a decimal point. "٢.٠" is 20, not 2 -- see
    :func:`string_matching.normalization.normalize_quantity`.
    """
    if index is None or index >= len(texts):
        return ExtractedValue()

    raw = texts[index]
    value = normalize_quantity(raw)
    if value is None:
        return ExtractedValue()

    normalized_differs = str(value) != (raw or "").strip()

    return ExtractedValue(
        value=value,
        confidence=confidences[index] if index < len(confidences) else 0.0,
        raw=raw if normalized_differs else None,
        bbox=boxes[index] if index < len(boxes) else None,
    )


def _price_field(
    index: int | None,
    values: list[float],
    boxes: list[BoundingBox],
    confidences: list[float],
    derived: float | None = None,
) -> ExtractedValue:
    """A monetary cell, or a value derived from the other two."""
    if index is None or index >= len(values):
        if derived is None:
            return ExtractedValue()

        # A derived total was computed, not read, so it has no box and no
        # confidence of its own -- and must not claim either.
        return ExtractedValue(value=round(derived, 2), confidence=0.0)

    return ExtractedValue(
        value=round(float(values[index]), 2),
        confidence=confidences[index] if index < len(confidences) else 0.0,
        bbox=boxes[index] if index < len(boxes) else None,
    )


@dataclass(frozen=True)
class _ColumnAssignment:
    """Which position in a row's number list holds which column."""

    quantity_index: int | None = None
    unit_price_index: int | None = None
    total_index: int | None = None

    # A total the row did not print, computed from quantity x unit price.
    derived_total: float | None = None


def _assign_columns(numbers: list[float]) -> _ColumnAssignment:
    """Map a row's numbers, given left to right, to quantity, unit price and total.

    Column order is **not** assumed. Invoice tables are conventionally ordered
    quantity, unit price, line total, but which end of the page that starts from
    depends on how the form was printed: an Arabic form usually runs the
    columns right to left, a Latin one left to right, and both turn up in the
    same shoebox of receipts. Guessing wrong scrambles every line on the page,
    so the row's own arithmetic decides it -- the ordering under which
    ``quantity x unit price`` actually equals the printed total is the right
    one, whichever direction it reads in.

    When neither ordering checks out (a mis-read digit, a discount column) the
    left-to-right reading stands and `arithmetic_ok` reports the mismatch, which
    is what the verification screen highlights.

    Returns indices rather than values, so the caller can go back to the cell
    each number came from -- which is what lets the quantity be re-read under
    its own rule and every field carry the box it was read from.
    """
    count = len(numbers)

    if count == 0:
        return _ColumnAssignment()

    if count == 1:
        # One number on an item row is a price; a lone quantity with no price is
        # not something an invoice prints.
        return _ColumnAssignment(unit_price_index=0)

    if count == 2:
        return _assign_two(numbers)

    forward = _assign_ordered(numbers)
    if _is_consistent(forward, numbers):
        return forward

    # Same row read the other way round, then the indices mapped back.
    backward = _assign_ordered(list(reversed(numbers)))
    if _is_consistent(backward, list(reversed(numbers))):
        return _mirror(backward, count)

    return forward


def _assign_two(numbers: list[float]) -> _ColumnAssignment:
    """Two numbers: quantity and unit price, with the total derived.

    With no printed total there is no arithmetic to check the order against, so
    the shape of the numbers decides: a quantity is a whole number and a unit
    price usually is not. When both look alike the left-to-right reading stands.
    """
    left_is_whole = float(numbers[0]).is_integer()
    right_is_whole = float(numbers[1]).is_integer()

    quantity_index, unit_price_index = (
        (1, 0) if right_is_whole and not left_is_whole else (0, 1)
    )

    return _ColumnAssignment(
        quantity_index=quantity_index,
        unit_price_index=unit_price_index,
        derived_total=round(numbers[quantity_index] * numbers[unit_price_index], 2),
    )


def _assign_ordered(numbers: list[float]) -> _ColumnAssignment:
    """The conventional qty / price / total mapping over an ordered list."""
    count = len(numbers)
    quantity_index, unit_price_index, total_index = count - 3, count - 2, count - 1

    # If the stated total contradicts qty x price but the number before them is
    # consistent with it, the row probably led with an index column.
    if (
        abs(numbers[quantity_index] * numbers[unit_price_index] - numbers[total_index])
        > _ARITHMETIC_TOLERANCE
        and count >= 4
        and abs(numbers[count - 4] * numbers[count - 3] - numbers[total_index])
        <= _ARITHMETIC_TOLERANCE
    ):
        quantity_index, unit_price_index = count - 4, count - 3

    return _ColumnAssignment(
        quantity_index=quantity_index,
        unit_price_index=unit_price_index,
        total_index=total_index,
    )


def _is_consistent(assignment: _ColumnAssignment, numbers: list[float]) -> bool:
    if (
        assignment.quantity_index is None
        or assignment.unit_price_index is None
        or assignment.total_index is None
    ):
        return False

    return _arithmetic_ok(
        numbers[assignment.quantity_index],
        numbers[assignment.unit_price_index],
        numbers[assignment.total_index],
    )


def _mirror(assignment: _ColumnAssignment, count: int) -> _ColumnAssignment:
    """Translate indices from a reversed list back onto the original order."""

    def flip(index: int | None) -> int | None:
        return None if index is None else count - 1 - index

    return _ColumnAssignment(
        quantity_index=flip(assignment.quantity_index),
        unit_price_index=flip(assignment.unit_price_index),
        total_index=flip(assignment.total_index),
        derived_total=assignment.derived_total,
    )


def _arithmetic_ok(quantity, unit_price, total_price) -> bool:
    if quantity is None or unit_price is None or total_price is None:
        return False
    return (
        abs(float(quantity) * float(unit_price) - float(total_price))
        <= _ARITHMETIC_TOLERANCE
    )


def _find_table_start(lines: list[TextLine]) -> int:
    """Index just past the column-header row, or a sensible default."""
    for i, line in enumerate(lines):
        norm = line.normalized
        # Two or more column words on one line is a header row, not prose.
        if sum(1 for hint in _COLUMN_HEADER_HINTS if hint in norm) >= 2:
            return i + 1

    # No header found: skip the letterhead block and hope for the best.
    return min(4, len(lines))


def _find_table_end(lines: list[TextLine], start: int) -> int:
    """Index of the first totals line after the table, or the end of the document."""
    for i in range(start, len(lines)):
        line = lines[i]
        if (
            line.has_any(_TOTAL_HINTS)
            or line.has_any(_SUBTOTAL_HINTS)
            or line.has_any(_TAX_HINTS)
        ):
            return i
    return len(lines)


# --------------------------------------------------------------------------
# Warnings
# --------------------------------------------------------------------------


def _add_warnings(draft: InvoiceDraft) -> None:
    for name, value in draft.header.named_fields():
        if value.is_low_confidence:
            draft.warnings.append(
                InvoiceWarning(
                    code=WarningCodes.LOW_CONFIDENCE_FIELD,
                    field=name,
                    message=(
                        f"{name} confidence {value.confidence:.2f} is below "
                        f"{REVIEW_CONFIDENCE_THRESHOLD}"
                    ),
                )
            )

        if value.requires_manual_review and value.candidates:
            draft.warnings.append(
                InvoiceWarning(
                    code=WarningCodes.MANUAL_REVIEW_REQUIRED,
                    field=name,
                    message=(
                        f"{name} did not match the catalog confidently; "
                        f"{len(value.candidates)} candidate(s) offered."
                    ),
                )
            )

    if not draft.line_items:
        draft.warnings.append(
            InvoiceWarning(
                code=WarningCodes.NO_LINE_ITEMS,
                message="No line items were detected in this invoice.",
            )
        )

    for item in draft.line_items:
        if not item.arithmetic_ok:
            draft.warnings.append(
                InvoiceWarning(
                    code=WarningCodes.ARITHMETIC_MISMATCH,
                    field=f"line_items[{item.row_index}]",
                    message=(
                        f"Row {item.row_index + 1}: quantity x unit price does not "
                        "equal the line total."
                    ),
                )
            )

        if item.product_name.requires_manual_review and item.product_name.candidates:
            draft.warnings.append(
                InvoiceWarning(
                    code=WarningCodes.MANUAL_REVIEW_REQUIRED,
                    field=f"line_items[{item.row_index}].product_name",
                    message=(
                        f"Row {item.row_index + 1}: no confident product match; "
                        f"{len(item.product_name.candidates)} candidate(s) offered."
                    ),
                )
            )

    items_sum = sum(float(item.total_price.value or 0.0) for item in draft.line_items)
    stated_total = draft.header.total_amount.value

    if (
        draft.line_items
        and stated_total is not None
        and abs(items_sum - float(stated_total)) > _TOTAL_TOLERANCE
    ):
        draft.warnings.append(
            InvoiceWarning(
                code=WarningCodes.TOTAL_MISMATCH,
                field="total_amount",
                message=(
                    f"Line items sum to {items_sum:.2f} but the stated total is "
                    f"{float(stated_total):.2f}."
                ),
            )
        )
