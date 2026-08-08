"""Maps raw OCR fragments + table structure into a StructuredDocument.

Because geometric correction runs once, upstream of both the OCR and
table branches (see orchestration.PipelineOrchestrator), every bbox
coming out of OCRResult and TableExtractionResult is already in the same
coordinate space -- the geometrically corrected display image. No
coordinate remapping happens here, only containment/assignment logic.
"""

from __future__ import annotations

from core.domain.document import StructuredDocument
from core.domain.ocr import OCRResult
from core.domain.table import TableExtractionResult


class CellMapper:
    def map(
        self, ocr_result: OCRResult, table_result: TableExtractionResult
    ) -> StructuredDocument:
        raise NotImplementedError(
            "Assign each OCRFragment to the TableCell whose bbox contains its "
            "centroid (merge multiple fragments per cell into "
            "TableCellElement.merged_text). Fragments assigned to no cell "
            "become FreeFieldElements."
        )
