"""Labels the cells of the standard Arabic sales-invoice form.

Ported from `classify_bill_cells` in `temp/table_det.py`, which was tuned
against this project's own sample invoices.

The form it knows (right-to-left, as printed):

    [ رقم الفاتورة : 00008 ]          <- above the table, top-left
    [ الاسم : ... ] [ التاريخ : ... ] [ المدينة : ... ]

    | ملاحظات | السعر الإجمالي | السعر الإفرادي | العدد | اسم المنتج | # |
    |   ...   |      ...       |      ...       |  ...  |    ...     | 1 |
                              ... 10 rows ...
    [ السعر الإجمالي ] [ رقماً : ... ] [ كتابة : ... ]   <- below the table

Nothing here is hardcoded to those six columns or ten rows: `column_roles`,
`table_row_count` and `table_col_count` are parameters, because they are
properties of one supplier's printed form. A second form is a config entry, not
a code change -- which is the whole reason classification is a separate,
registered family rather than part of the extractor.

Everything is measured **relative to the detected table body**, never in
absolute page coordinates. On a photographed page the table's position varies by
hundreds of pixels; its position relative to the header does not.
"""

from __future__ import annotations

import numpy as np

from core.domain.layout import InvoiceLayout, LayoutRegion
from core.domain.roles import CellRole, Zone, content_kind_for
from core.domain.table import TableCell, TableExtractionResult
from table_extraction.classifiers.registry import classifier_registry

# Left to right, which is how the columns of this RTL form appear in pixel
# order. Named as roles so nothing has to translate a string twice.
DEFAULT_COLUMN_ROLES: tuple[str, ...] = (
    CellRole.NOTES.value,
    CellRole.LINE_TOTAL.value,
    CellRole.UNIT_PRICE.value,
    CellRole.QUANTITY.value,
    CellRole.PRODUCT_NAME.value,
    CellRole.LINE_NUMBER.value,
)

# The three boxes beside the invoice number, left to right on the printed form.
DEFAULT_HEADER_ROLES: tuple[str, ...] = (
    CellRole.CITY.value,
    CellRole.INVOICE_DATE.value,
    CellRole.MERCHANT_NAME.value,
)

# The totals strip below the table, left to right. Two of the five are printed
# captions rather than values; they are labelled so the overlay can show them,
# and LABEL is never read as a field.
DEFAULT_FOOTER_ROLES: tuple[str, ...] = (
    CellRole.LABEL.value,
    CellRole.TOTAL_IN_WORDS.value,
    CellRole.LABEL.value,
    CellRole.TOTAL_IN_FIGURES.value,
    CellRole.TOTAL_AMOUNT.value,
)


class _Cell:
    """A TableCell with the centres the geometry rules need.

    TableCell is frozen -- correctly, it is a domain value -- so the working
    state of classification lives here instead of being written back onto it.
    """

    __slots__ = ("cell", "x_center", "y_center", "role")

    def __init__(self, cell: TableCell):
        self.cell = cell
        self.x_center = cell.bbox.x + cell.bbox.w / 2.0
        self.y_center = cell.bbox.y + cell.bbox.h / 2.0
        self.role = CellRole.UNKNOWN


def _cluster_rows(cells: list[_Cell], y_proximity: float | None = None) -> list[list[_Cell]]:
    """Group cells into printed rows, top to bottom, each sorted left to right.

    The comparison is against the running *mean* of the row so far rather than
    the previous cell, so one tall cell cannot drag the reference point far
    enough to split a printed row in two. `y_proximity` defaults to a fraction
    of the median cell height, so this works at any scan resolution.
    """
    if not cells:
        return []

    if y_proximity is None:
        median_h = float(np.median([c.cell.bbox.h for c in cells]))
        y_proximity = max(10.0, median_h * 0.6)

    ordered = sorted(cells, key=lambda c: c.y_center)

    rows: list[list[_Cell]] = []
    current = [ordered[0]]
    for c in ordered[1:]:
        row_y = float(np.mean([r.y_center for r in current]))
        if abs(c.y_center - row_y) <= y_proximity:
            current.append(c)
        else:
            rows.append(current)
            current = [c]
    rows.append(current)

    for row in rows:
        row.sort(key=lambda c: c.x_center)
    return rows


def _nearest_slot(
    x_center: float,
    slot_centers: list[float],
    avg_col_width: float | None,
    tol_ratio: float = 0.4,
) -> int | None:
    """Index of the closest expected column, or None when nothing is close.

    Returning None matters: a cell that lines up with no column is a stray
    detection, and assigning it to the nearest one anyway would shift a real
    value out of its column.
    """
    if not slot_centers:
        return None

    if avg_col_width is None:
        spread = max(slot_centers) - min(slot_centers)
        avg_col_width = spread / max(1, len(slot_centers) - 1)

    tol = avg_col_width * tol_ratio
    best_idx, best_dist = None, float("inf")
    for i, sc in enumerate(slot_centers):
        d = abs(x_center - sc)
        if d < best_dist:
            best_dist, best_idx = d, i

    return best_idx if best_dist <= tol else None


@classifier_registry.register("bill_layout")
class BillLayoutClassifier:
    name = "bill_layout"

    def __init__(
        self,
        table_row_count: int = 10,
        table_col_count: int = 6,
        column_roles: list[str] | tuple[str, ...] = DEFAULT_COLUMN_ROLES,
        header_roles: list[str] | tuple[str, ...] = DEFAULT_HEADER_ROLES,
        footer_roles: list[str] | tuple[str, ...] = DEFAULT_FOOTER_ROLES,
        zone_tolerance: int = 10,
        slot_tolerance_ratio: float = 0.4,
        ragged_row_slack: int = 5,
        **params,
    ):
        self.table_row_count = table_row_count
        self.table_col_count = table_col_count
        self.column_roles = [CellRole(r) for r in column_roles]
        self.header_roles = [CellRole(r) for r in header_roles]
        self.footer_roles = [CellRole(r) for r in footer_roles]

        if len(self.column_roles) != table_col_count:
            raise ValueError(
                f"bill_layout: column_roles has {len(self.column_roles)} entries but "
                f"table_col_count is {table_col_count}; they describe the same columns."
            )

        # Cells routinely bleed a few pixels past the table's outer rule.
        self.zone_tolerance = zone_tolerance
        self.slot_tolerance_ratio = slot_tolerance_ratio

        # A data row can carry MORE cells than there are columns: a handwritten
        # figure broken across a printed dot pattern closes as two contours. The
        # run detector tolerates that, and _nearest_slot puts both halves in the
        # right column.
        self.ragged_row_slack = ragged_row_slack

        self.params = params

    # ------------------------------------------------------------------

    def classify(
        self, table: TableExtractionResult, image_size: tuple[int, int]
    ) -> InvoiceLayout:
        structure = table.tables[0] if table.tables else None
        if structure is None or not structure.cells:
            return InvoiceLayout(source=f"{table.extractor_name}+{self.name}")

        cells = [_Cell(c) for c in structure.cells]

        top = structure.bbox.y
        bottom = structure.bbox.y + structure.bbox.h
        tol = self.zone_tolerance

        above = [c for c in cells if c.y_center < top - tol]
        inside = [c for c in cells if top - tol <= c.y_center <= bottom + tol]
        below = [c for c in cells if c.y_center > bottom + tol]

        rows = _cluster_rows(inside)
        totals_row = self._label_table(rows, below)
        self._label_header(above)
        self._label_footer(totals_row)

        regions = [
            LayoutRegion(
                id=f"{c.cell.table_id}:r{c.cell.row}c{c.cell.col}",
                bbox=c.cell.bbox,
                role=c.role,
                zone=self._zone_of(c, top, bottom),
                content_kind=content_kind_for(c.role),
                row=c.cell.row,
                col=c.cell.col,
                row_span=c.cell.row_span,
                col_span=c.cell.col_span,
                table_id=c.cell.table_id,
            )
            for c in cells
        ]

        return InvoiceLayout(
            regions=regions,
            table_bbox=structure.bbox,
            table_id=structure.table_id,
            source=f"{table.extractor_name}+{self.name}",
            raw_output=table.raw_extractor_output,
        )

    def _zone_of(self, c: _Cell, top: int, bottom: int) -> Zone:
        if c.y_center < top - self.zone_tolerance:
            return Zone.HEADER
        if c.y_center > bottom + self.zone_tolerance:
            return Zone.FOOTER
        return Zone.TABLE

    # ------------------------------------------------------------------
    # The item table
    # ------------------------------------------------------------------

    def _label_table(self, rows: list[list[_Cell]], below: list[_Cell]) -> list[_Cell]:
        """Label the column-header row and the data rows. Returns the totals row.

        The table block is found as the longest **run** of rows holding about
        the right number of cells, rather than by counting from the top. A stray
        detection above or below the table would break counting; it cannot break
        a run.
        """
        run_start, run_len = self._longest_full_run(rows)
        if run_start is None:
            # No recognizable product table. Everything stays UNKNOWN and the
            # parser falls back to its heuristics -- which is the correct
            # outcome for a form this classifier does not actually fit, and far
            # better than labelling arbitrary cells with confident-looking roles.
            return self._totals_from_below(below)

        table_rows = rows[run_start : run_start + run_len]
        header_row, data_rows = table_rows[0], table_rows[1 : 1 + self.table_row_count]

        slot_centers, avg_col_width = self._column_template(table_rows)
        valid = [c for c in slot_centers if c is not None]

        for row, is_header in [(header_row, True)] + [(r, False) for r in data_rows]:
            for cell in row:
                slot = _nearest_slot(
                    cell.x_center, valid, avg_col_width, self.slot_tolerance_ratio
                )
                if slot is None:
                    continue
                col_index = slot_centers.index(valid[slot])
                cell.role = (
                    CellRole.COLUMN_HEADER if is_header else self.column_roles[col_index]
                )

        # The totals strip is the row straight after the data rows when the
        # table's outer rule enclosed it, and a separate group below when it did
        # not. Both happen on real scans depending on how the form is printed.
        after_data = run_start + 1 + len(data_rows)
        after_run = run_start + run_len

        if after_data < len(rows):
            return rows[after_data]
        if after_run < len(rows):
            return rows[after_run]
        return self._totals_from_below(below)

    def _longest_full_run(self, rows: list[list[_Cell]]) -> tuple[int | None, int]:
        low, high = self.table_col_count, self.table_col_count + self.ragged_row_slack

        best_start, best_len = None, 0
        cur_start, cur_len = None, 0

        for i, row in enumerate(rows):
            if low <= len(row) <= high:
                if cur_start is None:
                    cur_start = i
                cur_len += 1
            else:
                if cur_len > best_len:
                    best_start, best_len = cur_start, cur_len
                cur_start, cur_len = None, 0

        if cur_len > best_len:
            best_start, best_len = cur_start, cur_len

        return best_start, best_len

    def _column_template(
        self, table_rows: list[list[_Cell]]
    ) -> tuple[list[float | None], float | None]:
        """Where each column sits, as the median x-centre over clean rows.

        Only rows holding exactly `table_col_count` cells vote: a row whose
        figure split into two contours has its cells shifted, and letting it
        vote would drag the whole template sideways.
        """
        clean = [r for r in table_rows if len(r) == self.table_col_count] or table_rows

        centers: list[float | None] = []
        for col in range(self.table_col_count):
            xs = [r[col].x_center for r in clean if len(r) > col]
            centers.append(float(np.median(xs)) if xs else None)

        valid = [c for c in centers if c is not None]
        avg_col_width = (
            (max(valid) - min(valid)) / (self.table_col_count - 1)
            if len(valid) >= 2 and self.table_col_count > 1
            else None
        )
        return centers, avg_col_width

    @staticmethod
    def _totals_from_below(below: list[_Cell]) -> list[_Cell]:
        rows = _cluster_rows(below)
        return rows[0] if rows else []

    # ------------------------------------------------------------------
    # Header and footer
    # ------------------------------------------------------------------

    def _label_header(self, above: list[_Cell]) -> None:
        """Invoice number, then city / date / merchant across the row below it.

        The invoice number is the box nearest the top-left corner of the header
        block, scored on normalized x + y so neither axis dominates on a page of
        any aspect ratio. It is scored across **the whole header block**, not
        within its topmost row: a row holding a single cell normalizes to a
        score of zero on both axes -- `max(range, 1.0)` divided into a spread of
        zero -- so whatever sat in it won by default and its position was never
        consulted at all. A page-edge sliver above the real header row is exactly
        that case, and it took the invoice number on a real scan while the box
        actually reading "رقم الفاتورة : 00010" was left unlabelled.

        City / date / merchant are bucketed across **the header row nearest the
        table**, not across every cell above it. Bucketing everything is what the
        original did, and on a real scan it mislabels: the letterhead's phone
        numbers close as their own small boxes, land in the rightmost bucket,
        and win the merchant field over the box that actually says الاسم. The
        three fields are printed on one row directly above the table, and that
        row is the only place they can be.
        """
        if not above or not self.header_roles:
            return

        rows = _cluster_rows(above)

        xs = [c.x_center for c in above]
        x_min, x_range = min(xs), max(max(xs) - min(xs), 1.0)
        ys = [c.y_center for c in above]
        y_min, y_range = min(ys), max(max(ys) - min(ys), 1.0)

        invoice_number = min(
            above,
            key=lambda c: (c.x_center - x_min) / x_range + (c.y_center - y_min) / y_range,
        )
        invoice_number.role = CellRole.INVOICE_NUMBER

        # The row immediately above the table. With only one header row it is
        # the same row the invoice number came from, and that cell is skipped.
        field_row = [c for c in rows[-1] if c is not invoice_number]
        if not field_row:
            return

        r_xs = [c.x_center for c in field_row]
        r_min = min(r_xs)
        width = max(max(r_xs) - r_min, 1.0)
        buckets = len(self.header_roles)

        for c in field_row:
            frac = (c.x_center - r_min) / width
            c.role = self.header_roles[min(buckets - 1, int(frac * buckets))]

    def _label_footer(self, totals_row: list[_Cell]) -> None:
        if not totals_row or not self.footer_roles:
            return

        ordered = sorted(totals_row, key=lambda c: c.x_center)

        if len(ordered) == len(self.footer_roles):
            for cell, role in zip(ordered, self.footer_roles):
                cell.role = role
            return

        # A different cell count means the strip merged or split somewhere.
        # Bucketing by position degrades gracefully instead of shifting every
        # label by one, which is what zipping a short list would do.
        x_min, x_max = ordered[0].x_center, ordered[-1].x_center
        width = max(x_max - x_min, 1.0)
        buckets = len(self.footer_roles)

        for c in ordered:
            frac = (c.x_center - x_min) / width
            c.role = self.footer_roles[min(buckets - 1, int(frac * buckets))]
