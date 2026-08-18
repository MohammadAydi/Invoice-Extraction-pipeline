"""The refiner that refines nothing.

The default, so a detector-driven configuration that has not declared a column
layout passes its boxes through untouched instead of needing a special case in
the flow.
"""

from __future__ import annotations

from typing import Sequence

from core.domain.ocr import TextRegion
from ocr.refiners.registry import refiner_registry


@refiner_registry.register("noop")
class NoopRegionRefiner:
    name = "noop"

    def __init__(self, **params):
        self.params = params

    def refine(
        self, regions: Sequence[TextRegion], width: int, height: int
    ) -> list[TextRegion]:
        return list(regions)
