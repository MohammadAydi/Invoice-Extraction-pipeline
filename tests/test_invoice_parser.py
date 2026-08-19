"""The semantic layer: elements in, an invoice out."""

from __future__ import annotations

import pytest

from core.domain.document import FreeFieldElement, StructuredDocument, TableCellElement
from core.domain.roles import CellRole, Zone
from core.domain.geometry import BoundingBox
from core.domain.ocr import OCRFragment, OCRResult
from core.domain.table import TableCell, TableExtractionResult, TableStructure
from table_extraction.classifiers.passthrough_classifier import (
    PassthroughLayoutClassifier,
)
from core.domain.invoice import WarningCodes
from core.domain.catalog import Catalogs
from invoice.parser import InvoiceParser
from mapping.cell_mapper import CellMapper
from core.domain.catalog import NamedEntry


def fragment(text: str, x: int, y: int, w: int = 60, h: int = 20, confidence: float = 0.9):
    return OCRFragment(
        text=text, bbox=BoundingBox(x=x, y=y, w=w, h=h), confidence=confidence
    )


def ocr(*fragments: OCRFragment) -> OCRResult:
    return OCRResult(fragments=list(fragments), engine_name="test")


def layout_of(table: TableExtractionResult):
    """The unclassified layout a bare table produces.

    CellMapper takes an InvoiceLayout now, because it maps the whole page --
    header captions and totals strip included -- rather than only table cells.
    `passthrough` is what turns a plain TableExtractionResult into one without
    labelling anything, which is exactly the behaviour these tests assert.
    """
    return PassthroughLayoutClassifier().classify(table, (1000, 1000))


def free_document(result: OCRResult) -> StructuredDocument:
    """Every fragment as a free field -- what you get with no table detected."""
    return StructuredDocument(
        elements=[
            FreeFieldElement(
                id=f"free:{i}", bbox=f.bbox, fragments=[f], merged_text=f.text
            )
            for i, f in enumerate(result.fragments)
        ]
    )


class TestHeader:
    def test_merchant_matches_the_customer_catalog(self):
        result = ocr(
            fragment("مؤسسه", 400, 40),
            fragment("النـور", 330, 40),
            fragment("التجارية", 250, 40),
            fragment("التاريخ:", 400, 100),
            fragment("14/03/2026", 300, 100),
        )
        catalogs = Catalogs(
            merchants=[NamedEntry(name="مؤسسة النور التجارية", entry_id=12)]
        )

        draft = InvoiceParser().parse(free_document(result), result, catalogs)
        merchant = draft.header.merchant_name

        # The OCR text stands as the value even on a confident match: the
        # catalog entry that won is reported alongside it, never in place of
        # it. The desktop app decides, against its own threshold.
        assert merchant.value == "مؤسسه النـور التجارية"
        assert merchant.matched_id == 12
        assert merchant.matched_to == "مؤسسة النور التجارية"
        assert not merchant.requires_manual_review
        assert merchant.candidates

    def test_merchant_falls_back_to_ocr_text_with_candidates_attached(self):
        result = ocr(fragment("متجر", 400, 40), fragment("الياسمين", 320, 40))
        catalogs = Catalogs(merchants=[NamedEntry(name="شركة الأمل", entry_id=13)])

        draft = InvoiceParser().parse(free_document(result), result, catalogs)
        merchant = draft.header.merchant_name

        # Nothing cleared the threshold, so the OCR text stands -- but the
        # candidate list is still there for the user to pick from.
        assert merchant.matched_id is None
        assert merchant.requires_manual_review
        assert merchant.candidates

    def test_date_is_iso(self):
        result = ocr(fragment("التاريخ", 400, 100), fragment("١٤/٠٣/٢٠٢٦", 300, 100))
        draft = InvoiceParser().parse(free_document(result), result)

        assert draft.header.invoice_date.value == "2026-03-14"

    def test_invoice_number_prefers_an_alphanumeric_token(self):
        result = ocr(fragment("رقم الفاتورة", 400, 60), fragment("INV-2291", 250, 60))
        draft = InvoiceParser().parse(free_document(result), result)

        assert draft.header.invoice_number.value == "INV-2291"

    def test_city_stops_before_the_next_field_on_the_same_line(self):
        result = ocr(
            fragment("المدينة:", 400, 140),
            fragment("دمشق", 330, 140),
            fragment("التاريخ:", 200, 140),
            fragment("14/03/2026", 100, 140),
        )
        draft = InvoiceParser().parse(free_document(result), result)

        assert draft.header.city.value == "دمشق"

    def test_city_matches_the_city_catalog(self):
        result = ocr(fragment("المدينة:", 400, 140), fragment("دمشق", 330, 140))
        catalogs = Catalogs(cities=[NamedEntry(name="دمشق"), NamedEntry(name="حلب")])

        draft = InvoiceParser().parse(free_document(result), result, catalogs)

        assert draft.header.city.value == "دمشق"
        assert draft.header.city.candidates

    def test_total_is_read_from_the_labelled_line(self):
        result = ocr(fragment("الإجمالي", 400, 900), fragment("128.75", 250, 900))
        draft = InvoiceParser().parse(free_document(result), result)

        assert draft.header.total_amount.value == 128.75

    def test_total_ignores_the_tax_line(self):
        result = ocr(
            fragment("الضريبة", 400, 860),
            fragment("18.75", 250, 860),
            fragment("الإجمالي", 400, 900),
            fragment("128.75", 250, 900),
        )
        draft = InvoiceParser().parse(free_document(result), result)

        assert draft.header.total_amount.value == 128.75


class TestLineItemsFromLines:
    def test_reads_a_row_of_qty_price_total(self):
        result = ocr(
            # Column header, so the table start is unambiguous.
            fragment("الصنف", 400, 200),
            fragment("الكمية", 250, 200),
            fragment("السعر", 150, 200),
            # The item row.
            fragment("سكر", 400, 240),
            fragment("2", 250, 240),
            fragment("1.25", 150, 240),
            fragment("2.50", 60, 240),
        )

        draft = InvoiceParser().parse(free_document(result), result)

        assert len(draft.line_items) == 1
        item = draft.line_items[0]
        assert item.product_name.value == "سكر"
        assert item.quantity.value == 2
        assert item.unit_price.value == 1.25
        assert item.total_price.value == 2.50
        assert item.arithmetic_ok

    def test_arithmetic_mismatch_is_flagged(self):
        result = ocr(
            fragment("الصنف", 400, 200),
            fragment("الكمية", 250, 200),
            fragment("سكر", 400, 240),
            fragment("2", 250, 240),
            fragment("1.25", 150, 240),
            fragment("9.99", 60, 240),
        )

        draft = InvoiceParser().parse(free_document(result), result)

        assert not draft.line_items[0].arithmetic_ok
        assert any(w.code == WarningCodes.ARITHMETIC_MISMATCH for w in draft.warnings)

    def test_reads_a_left_to_right_column_layout_too(self):
        # Same row, columns printed the other way round: qty, price, total from
        # the left. Nothing about the direction is assumed -- the ordering whose
        # arithmetic checks out is the one that is used.
        result = ocr(
            fragment("الصنف", 60, 200),
            fragment("الكمية", 200, 200),
            fragment("السعر", 300, 200),
            fragment("سكر", 60, 240),
            fragment("2", 200, 240),
            fragment("1.25", 300, 240),
            fragment("2.50", 400, 240),
        )

        item = InvoiceParser().parse(free_document(result), result).line_items[0]

        assert item.quantity.value == 2
        assert item.unit_price.value == 1.25
        assert item.total_price.value == 2.50
        assert item.arithmetic_ok

    def test_two_numbers_use_shape_to_tell_quantity_from_price(self):
        # No printed total means no arithmetic to check against, so the whole
        # number is the quantity and the fractional one is the price.
        result = ocr(
            fragment("الصنف", 400, 200),
            fragment("الكمية", 250, 200),
            fragment("سكر", 400, 240),
            fragment("1.25", 250, 240),
            fragment("3", 150, 240),
        )

        item = InvoiceParser().parse(free_document(result), result).line_items[0]

        assert item.quantity.value == 3
        assert item.unit_price.value == 1.25
        assert item.total_price.value == 3.75

    def test_a_leading_row_number_is_not_mistaken_for_the_quantity(self):
        result = ocr(
            fragment("الصنف", 400, 200),
            fragment("الكمية", 250, 200),
            fragment("سكر", 400, 240),
            # Printed left to right: row index, qty, price, total.
            fragment("1", 60, 240),
            fragment("2", 150, 240),
            fragment("1.25", 250, 240),
            fragment("2.50", 340, 240),
        )

        item = InvoiceParser().parse(free_document(result), result).line_items[0]

        assert item.quantity.value == 2
        assert item.unit_price.value == 1.25
        assert item.total_price.value == 2.50
        assert item.arithmetic_ok

    def test_a_row_with_no_numbers_is_not_an_item(self):
        result = ocr(
            fragment("الصنف", 400, 200),
            fragment("الكمية", 250, 200),
            fragment("ملاحظة", 400, 240),
            fragment("شكرا", 300, 240),
        )

        draft = InvoiceParser().parse(free_document(result), result)
        assert draft.line_items == []
        assert any(w.code == WarningCodes.NO_LINE_ITEMS for w in draft.warnings)


class TestLineItemsFromTableCells:
    @staticmethod
    def grid_document() -> StructuredDocument:
        """Two-row grid: a header row and one item row."""
        cells = [
            # row 0 -- printed column headers
            ("الصنف", 0, 0, 300, 200),
            ("الكمية", 0, 1, 200, 200),
            ("السعر", 0, 2, 100, 200),
            ("المبلغ", 0, 3, 20, 200),
            # row 1 -- the item
            ("جاكيت ازرق صوف", 1, 0, 300, 240),
            ("٢", 1, 1, 200, 240),
            ("1.25", 1, 2, 100, 240),
            ("2.50", 1, 3, 20, 240),
        ]

        elements = []
        for text, row, col, x, y in cells:
            bbox = BoundingBox(x=x, y=y, w=80, h=30)
            elements.append(
                TableCellElement(
                    id=f"t1:r{row}c{col}",
                    table_id="t1",
                    row=row,
                    col=col,
                    bbox=bbox,
                    fragments=[OCRFragment(text=text, bbox=bbox, confidence=0.9)],
                    merged_text=text,
                )
            )

        return StructuredDocument(elements=elements)

    def test_grid_rows_become_line_items(self):
        document = self.grid_document()
        result = ocr(*[f for el in document.elements for f in el.fragments])
        catalogs = Catalogs(products=[NamedEntry(name="جاكيت صوف أزرق", entry_id=31)])

        draft = InvoiceParser().parse(document, result, catalogs)

        # The printed header row is skipped, leaving exactly one item.
        assert len(draft.line_items) == 1
        item = draft.line_items[0]

        # Matched order-independently against the catalog, and reported without
        # overwriting what the paper said.
        assert item.product_name.value == "جاكيت ازرق صوف"
        assert item.product_name.matched_to == "جاكيت صوف أزرق"
        assert item.product_name.matched_id == 31

        # Arabic-Indic quantity folded to an integer.
        assert item.quantity.value == 2
        assert item.unit_price.value == 1.25
        assert item.total_price.value == 2.50
        assert item.arithmetic_ok

    def test_every_numeric_field_carries_its_own_box(self):
        document = self.grid_document()
        result = ocr(*[f for el in document.elements for f in el.fragments])

        item = InvoiceParser().parse(document, result).line_items[0]

        assert item.quantity.bbox is not None
        assert item.unit_price.bbox is not None
        assert item.total_price.bbox is not None
        # Different cells, so different boxes.
        assert item.quantity.bbox != item.unit_price.bbox


class TestCellMapper:
    def test_fragments_land_in_the_cell_containing_their_centroid(self):
        cell_a = TableCell(bbox=BoundingBox(x=0, y=0, w=100, h=50), row=0, col=0, table_id="t1")
        cell_b = TableCell(bbox=BoundingBox(x=100, y=0, w=100, h=50), row=0, col=1, table_id="t1")

        table = TableExtractionResult(
            tables=[
                TableStructure(
                    table_id="t1",
                    bbox=BoundingBox(x=0, y=0, w=200, h=50),
                    cells=[cell_a, cell_b],
                )
            ],
            extractor_name="test",
        )

        result = ocr(
            fragment("سكر", 10, 10, w=30, h=20),
            fragment("2", 110, 10, w=20, h=20),
            fragment("خارج", 10, 400, w=40, h=20),
        )

        document = CellMapper().map(result, layout_of(table))

        cells = [el for el in document.elements if isinstance(el, TableCellElement)]
        free = [el for el in document.elements if isinstance(el, FreeFieldElement)]

        assert {c.merged_text for c in cells} == {"سكر", "2"}
        assert [f.merged_text for f in free] == ["خارج"]

    def test_empty_cells_are_kept(self):
        cell = TableCell(bbox=BoundingBox(x=0, y=0, w=100, h=50), row=0, col=0, table_id="t1")
        table = TableExtractionResult(
            tables=[
                TableStructure(
                    table_id="t1", bbox=cell.bbox, cells=[cell]
                )
            ],
            extractor_name="test",
        )

        document = CellMapper().map(ocr(), layout_of(table))

        # A cell the engine could not read still needs a box to click on.
        assert len(document.elements) == 1
        assert document.elements[0].merged_text == ""

    def test_no_table_makes_everything_a_free_field(self):
        result = ocr(fragment("سكر", 10, 10))
        document = CellMapper().map(
            result, layout_of(TableExtractionResult(tables=[], extractor_name="test"))
        )

        assert len(document.elements) == 1
        assert isinstance(document.elements[0], FreeFieldElement)


class TestWarnings:
    def test_total_mismatch(self):
        result = ocr(
            fragment("الصنف", 400, 200),
            fragment("الكمية", 250, 200),
            fragment("سكر", 400, 240),
            fragment("2", 250, 240),
            fragment("1.25", 150, 240),
            fragment("2.50", 60, 240),
            fragment("الإجمالي", 400, 900),
            fragment("999.00", 250, 900),
        )

        draft = InvoiceParser().parse(free_document(result), result)
        assert any(w.code == WarningCodes.TOTAL_MISMATCH for w in draft.warnings)

    def test_no_text_at_all(self):
        result = ocr()
        draft = InvoiceParser().parse(StructuredDocument(elements=[]), result)

        assert draft.line_items == []
        assert any(w.code == WarningCodes.NO_LINE_ITEMS for w in draft.warnings)


class TestLabelledLayout:
    """Reading an invoice off a classified layout instead of guessing at one.

    When the layout stage labelled the boxes, the header fields and the item
    columns are known, and none of the keyword heuristics or the
    arithmetic-consistency column search should run. Those exist precisely
    because nobody knew which number was the quantity.
    """

    @staticmethod
    def cell(id, x, y, text, role, row=0, col=0, w=200, h=50):
        frag = OCRFragment(
            text=text, bbox=BoundingBox(x=x, y=y, w=w, h=h), confidence=0.9
        )
        return TableCellElement(
            id=id,
            table_id="t1",
            row=row,
            col=col,
            bbox=BoundingBox(x=x, y=y, w=w, h=h),
            fragments=[frag] if text else [],
            merged_text=text,
            role=role,
            zone=Zone.TABLE,
        )

    @staticmethod
    def field(id, x, y, text, role):
        frag = OCRFragment(
            text=text, bbox=BoundingBox(x=x, y=y, w=200, h=50), confidence=0.9
        )
        return FreeFieldElement(
            id=id,
            bbox=BoundingBox(x=x, y=y, w=200, h=50),
            fragments=[frag] if text else [],
            merged_text=text,
            role=role,
            zone=Zone.HEADER,
        )

    def document(self):
        elements = [
            self.field("no", 60, 40, "رقم الفاتورة : 00008", CellRole.INVOICE_NUMBER),
            self.field("city", 60, 170, "المدينة : ريف دمشق", CellRole.CITY),
            self.field("date", 460, 170, "التاريخ : 2026/8/15", CellRole.INVOICE_DATE),
            self.field("name", 860, 170, "الاسم : كرم هيثم", CellRole.MERCHANT_NAME),
            self.field("total", 60, 900, "795.25", CellRole.TOTAL_AMOUNT),
        ]
        for row, (product, qty, unit, line_total) in enumerate(
            [("سكر 1كغ", "2", "1.25", "2.50"), ("أرز بسمتي", "3", "8.50", "25.50")]
        ):
            y = 300 + row * 60
            elements += [
                self.cell(f"p{row}", 860, y, product, CellRole.PRODUCT_NAME, row, 4),
                self.cell(f"q{row}", 660, y, qty, CellRole.QUANTITY, row, 3),
                self.cell(f"u{row}", 460, y, unit, CellRole.UNIT_PRICE, row, 2),
                self.cell(f"t{row}", 260, y, line_total, CellRole.LINE_TOTAL, row, 1),
            ]
        return StructuredDocument(elements=elements)

    def test_header_fields_come_from_the_labelled_boxes(self):
        draft = InvoiceParser().parse(self.document(), ocr())

        assert draft.header.invoice_number.value == "00008"
        assert draft.header.invoice_date.value == "2026-08-15"
        assert draft.header.city.value == "ريف دمشق"
        assert draft.header.total_amount.value == 795.25

    def test_the_printed_caption_is_stripped_from_the_value(self):
        """These forms print the caption inside the same ruled box as the value,
        so the recognizer reads both and the caption has to come off before the
        value is matched against a catalog."""
        draft = InvoiceParser().parse(self.document(), ocr())

        assert "المدينة" not in str(draft.header.city.value)
        assert "رقم" not in str(draft.header.invoice_number.value)

    def test_a_leading_zero_invoice_number_survives(self):
        """"00008" is a string. Parsing it as a number eats the zeros."""
        draft = InvoiceParser().parse(self.document(), ocr())
        assert draft.header.invoice_number.value == "00008"

    def test_every_header_field_carries_the_box_it_was_read_from(self):
        """The point of the whole layout stage: clicking the merchant field on
        the verification screen highlights the box the value came from, not a
        line of reconstructed text."""
        draft = InvoiceParser().parse(self.document(), ocr())

        for name, value in draft.header.named_fields():
            assert value.bbox is not None, name

    def test_a_labelled_merchant_still_matches_the_catalog(self):
        catalogs = Catalogs(merchants=[NamedEntry(name="كرم هيثم", entry_id=7)])
        draft = InvoiceParser().parse(self.document(), ocr(), catalogs)

        assert draft.header.merchant_name.matched_id == 7
        assert draft.header.merchant_name.candidates

    def test_a_weak_merchant_match_is_flagged_rather_than_accepted(self):
        """Labelling the box says which field it is, not that whatever was read
        from it is right. A wrong confident match still corrupts an invoice."""
        catalogs = Catalogs(merchants=[NamedEntry(name="شركة الأمل التجارية", entry_id=9)])
        draft = InvoiceParser().parse(self.document(), ocr(), catalogs)

        assert draft.header.merchant_name.matched_id is None
        assert draft.header.merchant_name.requires_manual_review
        assert draft.header.merchant_name.candidates

    def test_line_items_come_from_the_labelled_columns(self):
        draft = InvoiceParser().parse(self.document(), ocr())

        assert len(draft.line_items) == 2
        first = draft.line_items[0]
        assert first.quantity.value == 2
        assert first.unit_price.value == 1.25
        assert first.total_price.value == 2.50
        assert first.arithmetic_ok

    def test_a_mismatched_row_is_reported_not_reinterpreted(self):
        """The column search exists to guess an unknown ordering. With the
        columns labelled, a row whose arithmetic fails has a misread digit --
        and reshuffling its columns to make the multiplication work would hide
        exactly the error the verification screen exists to surface.
        """
        elements = list(self.document().elements)
        for el in elements:
            if el.id == "t1" and el.role is CellRole.LINE_TOTAL:
                el.merged_text = "99.99"

        draft = InvoiceParser().parse(StructuredDocument(elements=elements), ocr())
        row = draft.line_items[1]

        assert row.quantity.value == 3
        assert row.unit_price.value == 8.50
        assert row.total_price.value == 99.99
        assert not row.arithmetic_ok
        assert any(w.code == WarningCodes.ARITHMETIC_MISMATCH for w in draft.warnings)

    def test_an_unlabelled_document_still_uses_the_heuristics(self):
        """The fallback has to stay: an unruled receipt has no layout at all."""
        result = ocr(
            fragment("التاريخ:", 400, 100),
            fragment("14/03/2026", 300, 100),
        )
        draft = InvoiceParser().parse(free_document(result), result)

        assert draft.header.invoice_date.value == "2026-03-14"
