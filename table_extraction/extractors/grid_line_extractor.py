# table_extraction/extractors/grid_line_extractor.py
from __future__ import annotations

import uuid

import cv2
import numpy as np

from core.domain.geometry import BoundingBox
from core.domain.image_payload import ImagePayload
from core.domain.table import TableCell, TableExtractionResult, TableStructure
from table_extraction.extractors.binary_input import require_binary_ink_white
from table_extraction.extractors.grid_utils import (
    extract_cells_with_spans,
    extend_lines_to_table_bounds,
    extract_lines_from_mask,
    filter_by_intersection,
    merge_close_lines,
)
from table_extraction.extractors.morphology import build_line_mask
from table_extraction.registry import extractor_registry


@extractor_registry.register("grid_line")
class GridLineTableExtractor:
    """Ruled-line table extractor: morphological line reconstruction, then
    intersect the surviving lines into a cell grid, span-aware.

    Contrast with `contour_based`, which takes the closed regions directly. This
    one is the better choice when the rules are clean and complete, since it
    recovers the exact grid topology (including spans); `contour_based` copes
    better with broken rules and with handwriting crossing them.

    It does no preprocessing of its own -- see `binary_input.py` for why that
    matters.
    """

    name = "grid_line"

    def __init__(
        self,
        dot_bridge_scale: int = 150,
        main_kernel_scale: int = 30,
        use_dot_bridging: bool = True,
        min_line_length_ratio: float = 0.05,
        intersection_tolerance: int = 15,
        merge_proximity: int = 15,
        min_extend_length_ratio: float = 0.5,
        coverage_ratio: float = 0.5,
        **extractor_params,
    ):
        self.dot_bridge_scale = dot_bridge_scale
        self.main_kernel_scale = main_kernel_scale
        self.use_dot_bridging = use_dot_bridging
        self.min_line_length_ratio = min_line_length_ratio
        self.intersection_tolerance = intersection_tolerance
        self.merge_proximity = merge_proximity
        self.min_extend_length_ratio = min_extend_length_ratio
        self.coverage_ratio = coverage_ratio
        self.extractor_params = extractor_params

    def extract(self, image: ImagePayload) -> TableExtractionResult:
        binary = require_binary_ink_white(image, self.name)
        h, w = binary.shape[:2]

        # --- line detection ---
        h_mask = build_line_mask(
            binary, is_horizontal=True, w=w, h=h,
            dot_bridge_scale=self.dot_bridge_scale,
            main_kernel_scale=self.main_kernel_scale,
            use_dot_bridging=self.use_dot_bridging,
        )
        v_mask = build_line_mask(
            binary, is_horizontal=False, w=w, h=h,
            dot_bridge_scale=self.dot_bridge_scale,
            main_kernel_scale=self.main_kernel_scale,
            use_dot_bridging=self.use_dot_bridging,
        )

        min_len_h = int(w * self.min_line_length_ratio)
        min_len_v = int(h * self.min_line_length_ratio)

        h_lines = extract_lines_from_mask(h_mask, True,  min_length=min_len_h)
        v_lines = extract_lines_from_mask(v_mask, False, min_length=min_len_v)

        h_lines = merge_close_lines(h_lines, is_horizontal=True,
                                    proximity_thresh=self.merge_proximity)
        v_lines = merge_close_lines(v_lines, is_horizontal=False,
                                    proximity_thresh=self.merge_proximity)

        h_lines, v_lines = filter_by_intersection(
            h_lines, v_lines, tolerance=self.intersection_tolerance
        )
        h_lines, v_lines = extend_lines_to_table_bounds(
            h_lines, v_lines,
            min_extend_length_ratio=self.min_extend_length_ratio,
        )

        # --- cell extraction (span-aware) ---
        raw_cells = extract_cells_with_spans(
            h_lines, v_lines, v_mask, h_mask,
            coverage_ratio=self.coverage_ratio,
        )

        # --- convert to domain objects ---
        if not raw_cells:
            return TableExtractionResult(
                tables=[], extractor_name=self.name, raw_extractor_output=raw_cells
            )

        # Compute the table's outer bounding box from the union of all cells
        x1 = min(c["x1"] for c in raw_cells)
        y1 = min(c["y1"] for c in raw_cells)
        x2 = max(c["x2"] for c in raw_cells)
        y2 = max(c["y2"] for c in raw_cells)
        table_id = str(uuid.uuid4())

        cells = [
            TableCell(
                bbox=BoundingBox(
                    x=c["x1"],
                    y=c["y1"],
                    w=c["x2"] - c["x1"],
                    h=c["y2"] - c["y1"],
                ),
                row=c["row"],
                col=c["col"],
                table_id=table_id,
                # Spans were computed and then dropped on the floor. A totals
                # row with no vertical dividers is one wide cell, and saying so
                # is what stops the mapper treating it as several narrow ones.
                row_span=c.get("rowspan", 1),
                col_span=c.get("colspan", 1),
            )
            for c in raw_cells
        ]

        table = TableStructure(
            table_id=table_id,
            bbox=BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1),
            cells=cells,
        )

        return TableExtractionResult(
            tables=[table],
            extractor_name=self.name,
            raw_extractor_output=raw_cells,
        )