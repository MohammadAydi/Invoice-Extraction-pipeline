"""The semantic layer: elements in, an invoice out."""

from __future__ import annotations

import pytest

from core.domain.document import FreeFieldElement, StructuredDocument, TableCellElement
from core.domain.geometry import BoundingBox
from core.domain.ocr import OCRFragment, OCRResult
from core.domain.table import TableCell, TableExtractionResult, TableStructure
from invoice.models import WarningCodes
from invoice.parser import Catalogs, InvoiceParser
from mapping.cell_mapper import CellMapper
from string_matching.catalog import NamedEntry


def fragment(text: str, x: int, y: int, w: int = 60, h: int = 20, confidence: float = 0.9):
    return OCRFragment(
        text=text, bbox=BoundingBox(x=x, y=y, w=w, h=h), confidence=confidence
    )


def ocr(*fragments: OCRFragment) -> OCRResult:
    return OCRResult(fragments=list(fragments), engine_name="test")


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

        assert draft.header.merchant_name.value == "مؤسسة النور التجارية"
        assert draft.header.merchant_name.matched_id == 12
        assert draft.header.merchant_name.candidates

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

        # Matched order-independently against the catalog.
        assert item.product_name.value == "جاكيت صوف أزرق"
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

        document = CellMapper().map(result, table)

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

        document = CellMapper().map(ocr(), table)

        # A cell the engine could not read still needs a box to click on.
        assert len(document.elements) == 1
        assert document.elements[0].merged_text == ""

    def test_no_table_makes_everything_a_free_field(self):
        result = ocr(fragment("سكر", 10, 10))
        document = CellMapper().map(
            result, TableExtractionResult(tables=[], extractor_name="test")
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
