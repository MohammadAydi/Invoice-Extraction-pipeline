"""Turning detected boxes into labelled invoice fields.

This is the stage that did not exist before: header fields and the totals row
were left to keyword heuristics running over reconstructed text lines, which
cannot see that a value sits inside a ruled box under a printed caption.
"""

from __future__ import annotations

from core.domain.geometry import BoundingBox
from core.domain.roles import CellRole, ContentKind, Zone
from core.domain.table import TableCell, TableExtractionResult, TableStructure
from table_extraction.classifiers.bill_layout_classifier import BillLayoutClassifier
from table_extraction.classifiers.passthrough_classifier import PassthroughLayoutClassifier

TABLE_ID = "t1"

# The standard form: an invoice-number box top-left, a row of three field boxes
# under it, a 6-column x (1 header + 10 data) table, then a 5-cell totals strip.
COL_X = [60, 260, 460, 660, 860, 1060]
COL_W = 190


def cell(x: int, y: int, w: int, h: int, row: int, col: int) -> TableCell:
    return TableCell(bbox=BoundingBox(x=x, y=y, w=w, h=h), row=row, col=col, table_id=TABLE_ID)


def standard_form(rows: int = 10) -> TableExtractionResult:
    cells: list[TableCell] = []
    row_index = 0

    # Header row 1: the invoice-number box, alone and to the left.
    cells.append(cell(60, 40, 220, 90, row_index, 0))
    row_index += 1

    # Header row 2: city, date, merchant -- left to right on this RTL form.
    for i, x in enumerate([60, 460, 860]):
        cells.append(cell(x, 170, 380, 90, row_index, i))
    row_index += 1

    table_top = 300
    for r in range(rows + 1):  # +1 for the printed column-header row
        y = table_top + r * 70
        for c, x in enumerate(COL_X):
            cells.append(cell(x, y, COL_W, 60, row_index, c))
        row_index += 1

    # Totals strip: five cells below the table.
    totals_y = table_top + (rows + 1) * 70 + 20
    for i, x in enumerate([60, 300, 540, 780, 1020]):
        cells.append(cell(x, totals_y, 220, 70, row_index, i))

    table_bbox = BoundingBox(
        x=COL_X[0],
        y=table_top,
        w=COL_X[-1] + COL_W - COL_X[0],
        h=(rows + 1) * 70,
    )

    return TableExtractionResult(
        tables=[TableStructure(table_id=TABLE_ID, bbox=table_bbox, cells=cells)],
        extractor_name="test",
    )


def classify(table: TableExtractionResult = None, **kwargs):
    return BillLayoutClassifier(**kwargs).classify(table or standard_form(), (1300, 1400))


class TestBillLayout:
    def test_header_fields_are_labelled_once_each(self):
        layout = classify()

        for role in (
            CellRole.INVOICE_NUMBER,
            CellRole.CITY,
            CellRole.INVOICE_DATE,
            CellRole.MERCHANT_NAME,
        ):
            assert len(layout.with_role(role)) == 1, role

    def test_the_invoice_number_is_the_top_left_box(self):
        layout = classify()
        region = layout.first_with_role(CellRole.INVOICE_NUMBER)

        assert region.bbox.x == 60
        assert region.bbox.y == 40

    def test_city_date_and_merchant_run_left_to_right(self):
        layout = classify()

        xs = {
            role: layout.first_with_role(role).bbox.x
            for role in (CellRole.CITY, CellRole.INVOICE_DATE, CellRole.MERCHANT_NAME)
        }
        assert xs[CellRole.CITY] < xs[CellRole.INVOICE_DATE] < xs[CellRole.MERCHANT_NAME]

    def test_letterhead_boxes_do_not_steal_the_merchant_field(self):
        """The bug the ported version had.

        It bucketed every cell above the table, so the letterhead's phone
        numbers -- which close as their own small boxes -- landed in the
        rightmost bucket and won the merchant field over the box that actually
        says الاسم. Only the header row nearest the table holds those fields.
        """
        table = standard_form()
        cells = list(table.tables[0].cells)
        # Two small boxes on a letterhead line, above everything else.
        cells.append(cell(900, 10, 120, 24, 0, 0))
        cells.append(cell(1040, 10, 120, 24, 0, 1))

        patched = TableExtractionResult(
            tables=[
                TableStructure(
                    table_id=TABLE_ID, bbox=table.tables[0].bbox, cells=cells
                )
            ],
            extractor_name="test",
        )

        layout = classify(patched)
        merchant = layout.first_with_role(CellRole.MERCHANT_NAME)

        assert len(layout.with_role(CellRole.MERCHANT_NAME)) == 1
        assert merchant.bbox.h > 24  # the real field box, not a letterhead sliver

    def test_a_stray_box_above_the_header_does_not_steal_the_invoice_number(self):
        """The same class of bug, one field over.

        The invoice number used to be scored within the *topmost* header row.
        A row holding a single stray cell normalizes to zero on both axes --
        `max(range, 1.0)` divided into a spread of zero -- so that cell won by
        default and its position was never consulted. Scoring across the whole
        header block is what makes the corner mean something.
        """
        table = standard_form()
        cells = list(table.tables[0].cells)
        cells.append(cell(900, 10, 120, 24, 0, 0))
        cells.append(cell(1040, 10, 120, 24, 0, 1))

        patched = TableExtractionResult(
            tables=[
                TableStructure(
                    table_id=TABLE_ID, bbox=table.tables[0].bbox, cells=cells
                )
            ],
            extractor_name="test",
        )

        region = classify(patched).first_with_role(CellRole.INVOICE_NUMBER)

        assert (region.bbox.x, region.bbox.y) == (60, 40)

    def test_a_lone_page_edge_sliver_does_not_steal_the_invoice_number(self):
        """Measured off `images/test1.jpg`, which is where this was found.

        A 71x222 sliver of the page edge closed as its own contour at the very
        top right. It clustered alone, took INVOICE_NUMBER, was cropped and read
        with the free-text prompt, and came back as an invented Arabic sentence
        -- while the box holding "رقم الفاتورة : 00010" stayed unlabelled.
        """
        table = standard_form()
        cells = list(table.tables[0].cells)
        # Scaled to this fixture's geometry: far right, above the header row,
        # and far enough above it to cluster on its own.
        cells.append(cell(1180, 8, 40, 120, 0, 0))

        patched = TableExtractionResult(
            tables=[
                TableStructure(
                    table_id=TABLE_ID, bbox=table.tables[0].bbox, cells=cells
                )
            ],
            extractor_name="test",
        )

        layout = classify(patched)
        region = layout.first_with_role(CellRole.INVOICE_NUMBER)

        assert (region.bbox.x, region.bbox.y) == (60, 40)
        # And the sliver is left unlabelled rather than given some other field.
        assert len(layout.with_role(CellRole.INVOICE_NUMBER)) == 1

    def test_the_column_header_row_is_labelled_as_such(self):
        layout = classify()
        headers = layout.with_role(CellRole.COLUMN_HEADER)

        assert len(headers) == len(COL_X)
        assert {h.bbox.y for h in headers} == {300}

    def test_every_data_row_gets_the_six_column_roles(self):
        layout = classify()

        for role in (
            CellRole.NOTES,
            CellRole.LINE_TOTAL,
            CellRole.UNIT_PRICE,
            CellRole.QUANTITY,
            CellRole.PRODUCT_NAME,
            CellRole.LINE_NUMBER,
        ):
            assert len(layout.with_role(role)) == 10, role

    def test_column_roles_follow_the_configured_order(self):
        layout = classify()

        for role, x in zip(
            (
                CellRole.NOTES,
                CellRole.LINE_TOTAL,
                CellRole.UNIT_PRICE,
                CellRole.QUANTITY,
                CellRole.PRODUCT_NAME,
                CellRole.LINE_NUMBER,
            ),
            COL_X,
        ):
            assert {r.bbox.x for r in layout.with_role(role)} == {x}, role

    def test_the_totals_strip_is_labelled(self):
        layout = classify()

        assert len(layout.with_role(CellRole.TOTAL_AMOUNT)) == 1
        assert len(layout.with_role(CellRole.TOTAL_IN_FIGURES)) == 1
        assert len(layout.with_role(CellRole.TOTAL_IN_WORDS)) == 1
        assert len(layout.with_role(CellRole.LABEL)) == 2

    def test_zones_split_around_the_table(self):
        layout = classify()

        assert {r.bbox.y for r in layout.header_regions} == {40, 170}
        assert all(r.bbox.y >= 300 for r in layout.table_regions)
        assert all(r.bbox.y > 300 for r in layout.footer_regions)

    def test_content_kind_follows_the_role(self):
        """The seam that keeps the recognizer generic: it selects a prompt from
        the kind and knows nothing about invoices."""
        layout = classify()

        assert layout.first_with_role(CellRole.QUANTITY).content_kind is ContentKind.NUMBER
        assert (
            layout.first_with_role(CellRole.PRODUCT_NAME).content_kind
            is ContentKind.ARABIC_TEXT
        )
        # A date is not a number: the number prompt strips its separators.
        assert layout.first_with_role(CellRole.INVOICE_DATE).content_kind is ContentKind.DATE

    def test_a_different_form_is_a_config_change(self):
        """Nothing is hardcoded to six columns or ten rows."""
        four_cols = [60, 260, 460, 660]
        cells = [
            cell(x, 300 + r * 70, 190, 60, r, c)
            for r in range(4)
            for c, x in enumerate(four_cols)
        ]
        table = TableExtractionResult(
            tables=[
                TableStructure(
                    table_id=TABLE_ID,
                    bbox=BoundingBox(x=60, y=300, w=790, h=280),
                    cells=cells,
                )
            ],
            extractor_name="test",
        )

        layout = BillLayoutClassifier(
            table_row_count=3,
            table_col_count=4,
            column_roles=["line_total", "unit_price", "quantity", "product_name"],
        ).classify(table, (1000, 1000))

        assert len(layout.with_role(CellRole.PRODUCT_NAME)) == 3
        assert len(layout.with_role(CellRole.NOTES)) == 0

    def test_an_unrecognizable_page_labels_nothing_rather_than_guessing(self):
        """Better than confident-looking roles on arbitrary cells: the parser
        sees `has_roles` is False and falls back to its heuristics."""
        cells = [cell(60, 40 + i * 80, 200, 60, i, 0) for i in range(3)]
        table = TableExtractionResult(
            tables=[
                TableStructure(
                    table_id=TABLE_ID,
                    bbox=BoundingBox(x=60, y=40, w=200, h=220),
                    cells=cells,
                )
            ],
            extractor_name="test",
        )

        layout = BillLayoutClassifier().classify(table, (1000, 1000))

        assert layout.regions
        assert not layout.with_role(CellRole.PRODUCT_NAME)

    def test_an_empty_table_yields_an_empty_layout(self):
        layout = BillLayoutClassifier().classify(
            TableExtractionResult(tables=[], extractor_name="test"), (1000, 1000)
        )

        assert not layout
        assert not layout.has_roles


class TestPassthrough:
    def test_labels_nothing_but_keeps_every_box(self):
        table = standard_form()
        layout = PassthroughLayoutClassifier().classify(table, (1300, 1400))

        assert len(layout.regions) == len(table.tables[0].cells)
        assert not layout.has_roles
        assert all(r.role is CellRole.UNKNOWN for r in layout.regions)

    def test_zones_still_come_from_geometry(self):
        """Position needs no knowledge of the form, so even the classifier that
        labels nothing can say what is above and below the table."""
        layout = PassthroughLayoutClassifier().classify(standard_form(), (1300, 1400))

        assert layout.header_regions
        assert layout.table_regions
        assert layout.footer_regions
        assert all(r.zone is not Zone.UNKNOWN for r in layout.regions)
