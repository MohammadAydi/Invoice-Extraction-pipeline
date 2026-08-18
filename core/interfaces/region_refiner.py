from __future__ import annotations

from typing import Protocol, Sequence

from core.domain.ocr import TextRegion


class RegionRefiner(Protocol):
    """Adjusts a detector's boxes before they are cropped.

    A text detector boxes ink, not fields: on a ruled invoice it happily draws
    one box across three columns because the handwriting runs together. Refining
    is the step that cuts such a box at the column boundaries and invents boxes
    for cells the detector missed entirely.

    Only the detector-driven flow needs this. The layout-driven flow takes its
    regions from the printed grid, which already has the real boundaries, so it
    has nothing to refine.
    """

    def refine(
        self, regions: Sequence[TextRegion], width: int, height: int
    ) -> list[TextRegion]: ...
