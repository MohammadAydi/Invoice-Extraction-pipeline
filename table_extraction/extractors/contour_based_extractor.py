"""Contour-based cell detection.

Ported from `temp/table_det.py`, which was developed against this project's own
sample invoices and beats the line-based `grid_line` extractor on them. See
`contour_utils.py` for why going after the closed regions is more robust than
reconstructing the grid from its lines, and for what the stack filter does.

This extractor answers *where the boxes are* and nothing else. What each box
means is a separate stage -- see `table_extraction/classifiers/` -- because
"find the ruled regions on a photograph" is the same problem for every form,
while "the box above the table on the left is the invoice number" is knowledge
of one printed form.

One convention worth stating: the returned `TableStructure` carries **every**
ruled box on the page in `cells` -- the header captions and the totals row
included -- while its `bbox` is the item table alone, found by union-find as the
largest group of touching boxes. That asymmetry is deliberate. The header and
totals boxes are exactly what the classifier has to label, so dropping them here
would throw away the answer; and `bbox` is the reference line the classifier
measures "above" and "below" against.
"""

from __future__ import annotations

import uuid

import cv2

from core.domain.geometry import BoundingBox
from core.domain.image_payload import ImagePayload
from core.domain.table import TableCell, TableExtractionResult, TableStructure
from table_extraction.extractors.binary_input import require_binary_ink_white
from table_extraction.extractors.contour_utils import (
    cluster_cells_into_rows,
    dedup_cells,
    find_table_bounds,
    get_table_cell_bounds,
    remove_stacked_text_blobs,
)
from table_extraction.extractors.morphology import build_line_mask
from table_extraction.registry import extractor_registry


@extractor_registry.register("contour_based")
class ContourBasedTableExtractor:
    name = "contour_based"

    def __init__(
        self,
        dot_bridge_scale: int = 150,
        main_kernel_scale: int = 30,
        use_dot_bridging: bool = True,
        use_stack_filter: bool = True,
        stack_y_proximity: int = 25,
        stack_length_similarity: float = 0.35,
        stack_min_group_size: int = 2,
        require_no_vertical_intersection: bool = True,
        stack_intersection_band: int = 4,
        cell_min_size: int = 15,
        cell_min_height: int = 20,
        cell_max_area_ratio: float = 0.20,
        cell_max_aspect_ratio: float = 0.3,
        cell_max_width_ratio: float = 0.6,
        table_proximity: int = 3,
        dedup_tolerance: int = 5,
        **extractor_params,
    ):
        self.dot_bridge_scale = dot_bridge_scale
        self.main_kernel_scale = main_kernel_scale
        self.use_dot_bridging = use_dot_bridging

        self.use_stack_filter = use_stack_filter
        self.stack_y_proximity = stack_y_proximity
        self.stack_length_similarity = stack_length_similarity
        self.stack_min_group_size = stack_min_group_size
        self.require_no_vertical_intersection = require_no_vertical_intersection
        self.stack_intersection_band = stack_intersection_band

        self.cell_min_size = cell_min_size
        self.cell_min_height = cell_min_height
        self.cell_max_area_ratio = cell_max_area_ratio
        self.cell_max_aspect_ratio = cell_max_aspect_ratio
        self.cell_max_width_ratio = cell_max_width_ratio

        self.table_proximity = table_proximity
        self.dedup_tolerance = dedup_tolerance
        self.extractor_params = extractor_params

    def extract(self, image: ImagePayload) -> TableExtractionResult:
        binary = require_binary_ink_white(image, self.name)
        h, w = binary.shape[:2]

        horizontal_mask = build_line_mask(
            binary,
            is_horizontal=True,
            w=w,
            h=h,
            dot_bridge_scale=self.dot_bridge_scale,
            main_kernel_scale=self.main_kernel_scale,
            use_dot_bridging=self.use_dot_bridging,
        )
        vertical_mask = build_line_mask(
            binary,
            is_horizontal=False,
            w=w,
            h=h,
            dot_bridge_scale=self.dot_bridge_scale,
            main_kernel_scale=self.main_kernel_scale,
            use_dot_bridging=self.use_dot_bridging,
        )

        removed_mask = None
        if self.use_stack_filter:
            horizontal_mask, removed_mask = remove_stacked_text_blobs(
                horizontal_mask,
                is_horizontal=True,
                y_proximity=self.stack_y_proximity,
                length_similarity=self.stack_length_similarity,
                min_group_size=self.stack_min_group_size,
                vertical_mask=(
                    vertical_mask if self.require_no_vertical_intersection else None
                ),
                intersection_band=self.stack_intersection_band,
            )

        grid_mask = cv2.bitwise_or(horizontal_mask, vertical_mask)

        raw_cells = dedup_cells(
            get_table_cell_bounds(
                grid_mask,
                min_size=self.cell_min_size,
                min_height=self.cell_min_height,
                max_area_ratio=self.cell_max_area_ratio,
                max_aspect_ratio=self.cell_max_aspect_ratio,
                max_width_ratio=self.cell_max_width_ratio,
            ),
            tol=self.dedup_tolerance,
        )

        debug = {
            "grid_mask": grid_mask,
            "horizontal_mask": horizontal_mask,
            "vertical_mask": vertical_mask,
            "removed_as_text_mask": removed_mask,
            "raw_cells": raw_cells,
            "table_bounds": None,
        }

        if not raw_cells:
            return TableExtractionResult(
                tables=[], extractor_name=self.name, raw_extractor_output=debug
            )

        table_bounds = find_table_bounds(raw_cells, proximity=self.table_proximity)
        debug["table_bounds"] = table_bounds

        # Row/column indices come from clustering every detected cell, not only
        # the ones inside the table: the header captions and the totals row are
        # cells too, and the classifier needs them indexed in the same scheme to
        # reason about what sits above and below the table.
        rows = cluster_cells_into_rows(raw_cells)
        table_id = str(uuid.uuid4())

        cells: list[TableCell] = []
        for row_index, row in enumerate(rows):
            for col_index, c in enumerate(row):
                cells.append(
                    TableCell(
                        bbox=BoundingBox(
                            x=int(c["x"]),
                            y=int(c["y"]),
                            w=int(c["width"]),
                            h=int(c["height"]),
                        ),
                        row=row_index,
                        col=col_index,
                        table_id=table_id,
                    )
                )

        if table_bounds is not None:
            table_bbox = BoundingBox(
                x=int(table_bounds["x1"]),
                y=int(table_bounds["y1"]),
                w=int(table_bounds["x2"] - table_bounds["x1"]),
                h=int(table_bounds["y2"] - table_bounds["y1"]),
            )
        else:
            xs = [c.bbox.x for c in cells]
            ys = [c.bbox.y for c in cells]
            table_bbox = BoundingBox(
                x=min(xs),
                y=min(ys),
                w=max(c.bbox.x + c.bbox.w for c in cells) - min(xs),
                h=max(c.bbox.y + c.bbox.h for c in cells) - min(ys),
            )

        return TableExtractionResult(
            tables=[TableStructure(table_id=table_id, bbox=table_bbox, cells=cells)],
            extractor_name=self.name,
            raw_extractor_output=debug,
        )
