"""Stub. This is where lessons 10-11 (ruled line detection, region
splitting) land: detect horizontal/vertical ruling lines (extending the
HoughLinesP work already used for deskew), intersect them into a cell
grid, and wrap the result as TableStructure/TableCell.
"""

from __future__ import annotations

from core.domain.image_payload import ImagePayload
from core.domain.table import TableExtractionResult
from table_extraction.registry import extractor_registry


@extractor_registry.register("contour_based")
class ContourBasedTableExtractor:
    def __init__(self, **extractor_params):
        self.extractor_params = extractor_params

    def extract(self, image: ImagePayload) -> TableExtractionResult:
        raise NotImplementedError(
            "Detect table + cell boundaries here (ruled-line detection + region "
            "splitting). Return a TableExtractionResult."
        )
