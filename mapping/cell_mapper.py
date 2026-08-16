"""Maps raw OCR fragments + table structure into a StructuredDocument.

Because geometric correction runs once, upstream of both the OCR and
table branches (see orchestration.PipelineOrchestrator), every bbox
coming out of OCRResult and TableExtractionResult is already in the same
coordinate space -- the geometrically corrected display image. No
coordinate remapping happens here, only containment/assignment logic.

Assignment is by centroid rather than by overlap: a fragment whose box clips the
printed cell divider still belongs to the cell its ink sits in, and centroid
containment says so without needing an overlap threshold nobody can tune.
"""

from __future__ import annotations

from core.domain.document import (
    DocumentElement,
    FreeFieldElement,
    StructuredDocument,
    TableCellElement,
)
from core.domain.geometry import BoundingBox
from core.domain.ocr import OCRFragment, OCRResult
from core.domain.table import TableCell, TableExtractionResult


def _centroid(bbox: BoundingBox) -> tuple[float, float]:
    return bbox.x + bbox.w / 2.0, bbox.y + bbox.h / 2.0


def _contains(cell: BoundingBox, x: float, y: float) -> bool:
    return cell.x <= x <= cell.x + cell.w and cell.y <= y <= cell.y + cell.h


def _merge_text(fragments: list[OCRFragment]) -> str:
    """Join a cell's fragments in Arabic reading order: top to bottom, right to left.

    Reading order matters even inside one cell: a two-line description merged
    left-to-right would come back with its halves swapped, and the string
    matcher would then score it against the catalog as a different product.
    """
    if not fragments:
        return ""

    ordered = sorted(
        fragments,
        key=lambda f: (round(f.bbox.y + f.bbox.h / 2.0), -(f.bbox.x + f.bbox.w)),
    )
    return " ".join(f.text.strip() for f in ordered if f.text and f.text.strip()).strip()


def _hull(fragments: list[OCRFragment]) -> BoundingBox:
    """Smallest box covering every fragment, used when a cell has no box of its own."""
    x1 = min(f.bbox.x for f in fragments)
    y1 = min(f.bbox.y for f in fragments)
    x2 = max(f.bbox.x + f.bbox.w for f in fragments)
    y2 = max(f.bbox.y + f.bbox.h for f in fragments)
    return BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)


class CellMapper:
    """Turns (OCR fragments, table cells) into one flat list of UI elements."""

    def map(
        self, ocr_result: OCRResult, table_result: TableExtractionResult
    ) -> StructuredDocument:
        cells: list[TableCell] = [
            cell for table in table_result.tables for cell in table.cells
        ]

        # Smallest cell first. Grid extraction can emit a whole-table box
        # alongside its cells, and a fragment inside both belongs to the cell.
        cells.sort(key=lambda c: c.bbox.w * c.bbox.h)

        per_cell: dict[int, list[OCRFragment]] = {}
        unassigned: list[OCRFragment] = []

        for fragment in ocr_result.fragments:
            if not (fragment.text or "").strip():
                continue

            x, y = _centroid(fragment.bbox)
            index = next(
                (i for i, cell in enumerate(cells) if _contains(cell.bbox, x, y)),
                None,
            )

            if index is None:
                unassigned.append(fragment)
            else:
                per_cell.setdefault(index, []).append(fragment)

        elements: list[DocumentElement] = []

        # Every detected cell becomes an element, including the empty ones: the
        # verification grid needs a box to click on for a quantity the engine
        # failed to read, and a missing element would silently shift the row.
        for index, cell in enumerate(cells):
            fragments = per_cell.get(index, [])
            elements.append(
                TableCellElement(
                    id=f"{cell.table_id}:r{cell.row}c{cell.col}",
                    table_id=cell.table_id,
                    row=cell.row,
                    col=cell.col,
                    bbox=cell.bbox,
                    fragments=fragments,
                    merged_text=_merge_text(fragments),
                )
            )

        for index, fragment in enumerate(unassigned):
            elements.append(
                FreeFieldElement(
                    id=f"free:{index}",
                    bbox=fragment.bbox,
                    fragments=[fragment],
                    merged_text=fragment.text.strip(),
                )
            )

        return StructuredDocument(elements=elements)

    @staticmethod
    def bounds_of(fragments: list[OCRFragment]) -> BoundingBox | None:
        """Public helper for callers assembling their own elements from fragments."""
        return _hull(fragments) if fragments else None
