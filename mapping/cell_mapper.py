"""Maps raw OCR fragments onto the page's labelled regions.

Because geometric correction runs once, upstream of both the OCR and table
branches (see orchestration.PipelineOrchestrator), every bbox coming out of
OCRResult and InvoiceLayout is already in the same coordinate space -- the
geometrically corrected display image. No coordinate remapping happens here,
only assignment.

Two things changed from the version that only knew about table cells:

* It takes an :class:`~core.domain.layout.InvoiceLayout`, not a
  `TableExtractionResult`, so it maps the **whole page**: the invoice number,
  city, date and merchant boxes above the table and the totals strip below it
  are regions like any other. Leaving them out was why header fields had to be
  re-derived downstream from reconstructed text lines, which cannot see that a
  value sits inside a ruled box under a printed caption.

* Assignment is by **identity** when the fragment carries a `source_id`. In the
  layout-driven flow the crop was cut from a specific region, so which region it
  belongs to is known exactly -- re-deriving it from geometry would be throwing
  away a fact in order to guess at it.

Fragments with no `source_id` fall back to centroid containment: a fragment
whose box clips a printed cell divider still belongs to the cell its ink sits
in, and centroid containment says so without needing an overlap threshold nobody
can tune.
"""

from __future__ import annotations

from core.domain.document import (
    DocumentElement,
    FreeFieldElement,
    StructuredDocument,
    TableCellElement,
)
from core.domain.geometry import BoundingBox
from core.domain.layout import InvoiceLayout, LayoutRegion
from core.domain.ocr import OCRFragment, OCRResult
from core.domain.roles import CellRole, Zone


def _centroid(bbox: BoundingBox) -> tuple[float, float]:
    return bbox.x + bbox.w / 2.0, bbox.y + bbox.h / 2.0


def _contains(box: BoundingBox, x: float, y: float) -> bool:
    return box.x <= x <= box.x + box.w and box.y <= y <= box.y + box.h


def _merge_text(fragments: list[OCRFragment]) -> str:
    """Join a region's fragments in Arabic reading order: top to bottom, right to left.

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
    """Smallest box covering every fragment, used when a region has no box of its own."""
    x1 = min(f.bbox.x for f in fragments)
    y1 = min(f.bbox.y for f in fragments)
    x2 = max(f.bbox.x + f.bbox.w for f in fragments)
    y2 = max(f.bbox.y + f.bbox.h for f in fragments)
    return BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)


class CellMapper:
    """Turns (OCR fragments, page layout) into one flat list of UI elements."""

    def map(self, ocr_result: OCRResult, layout: InvoiceLayout) -> StructuredDocument:
        regions = list(layout.regions)

        # Smallest region first. A layout can carry a whole-table box alongside
        # its cells, and a fragment inside both belongs to the cell.
        order = sorted(range(len(regions)), key=lambda i: regions[i].area)
        by_id = {region.id: i for i, region in enumerate(regions)}

        per_region: dict[int, list[OCRFragment]] = {}
        unassigned: list[OCRFragment] = []

        for fragment in ocr_result.fragments:
            if not (fragment.text or "").strip():
                continue

            index = self._assign(fragment, regions, order, by_id)
            if index is None:
                unassigned.append(fragment)
            else:
                per_region.setdefault(index, []).append(fragment)

        elements: list[DocumentElement] = []

        # Every region becomes an element, including the ones nothing was read
        # in: the verification grid needs a box to click on for a quantity the
        # engine failed to read, and a missing element would silently shift the
        # row.
        for index, region in enumerate(regions):
            elements.append(self._element(region, per_region.get(index, [])))

        for index, fragment in enumerate(unassigned):
            elements.append(
                FreeFieldElement(
                    id=f"free:{index}",
                    bbox=fragment.bbox,
                    fragments=[fragment],
                    merged_text=fragment.text.strip(),
                    role=CellRole.UNKNOWN,
                    zone=Zone.UNKNOWN,
                )
            )

        return StructuredDocument(elements=elements)

    # ------------------------------------------------------------------

    @staticmethod
    def _assign(
        fragment: OCRFragment,
        regions: list[LayoutRegion],
        order: list[int],
        by_id: dict[str, int],
    ) -> int | None:
        """Which region this fragment belongs to, by identity where known."""
        if fragment.source_id is not None:
            # The crop was cut from this region. If the id is unknown the layout
            # and the fragments came from different runs, which is a bug, not
            # something to paper over with a geometric guess.
            return by_id.get(fragment.source_id)

        x, y = _centroid(fragment.bbox)
        return next(
            (i for i in order if _contains(regions[i].bbox, x, y)),
            None,
        )

    @staticmethod
    def _element(region: LayoutRegion, fragments: list[OCRFragment]) -> DocumentElement:
        merged = _merge_text(fragments)

        if region.row is None or region.col is None or region.table_id is None:
            return FreeFieldElement(
                id=region.id,
                bbox=region.bbox,
                fragments=fragments,
                merged_text=merged,
                role=region.role,
                zone=region.zone,
            )

        return TableCellElement(
            id=region.id,
            table_id=region.table_id,
            row=region.row,
            col=region.col,
            bbox=region.bbox,
            fragments=fragments,
            merged_text=merged,
            role=region.role,
            zone=region.zone,
            row_span=region.row_span,
            col_span=region.col_span,
        )

    @staticmethod
    def bounds_of(fragments: list[OCRFragment]) -> BoundingBox | None:
        """Public helper for callers assembling their own elements from fragments."""
        return _hull(fragments) if fragments else None
