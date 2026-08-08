"""Stub(s). Reference material: thresholding.py (fixed-T sweep, Otsu via
THRESH_BINARY + THRESH_OTSU, histogram inspection, and adaptiveThreshold
with ADAPTIVE_THRESH_GAUSSIAN_C across several blockSize values).

Two strategies are registered to show the Strategy pattern in action within
one "family" -- config picks which one runs via `name`, and both can coexist
in the registry without either implementation knowing about the other.
"""

from __future__ import annotations

from core.domain.image_payload import PipelineContext
from preprocessing.steps.registry import step_registry


@step_registry.register("adaptive_threshold")
class AdaptiveThresholdStep:
    name = "adaptive_threshold"

    def __init__(self, block_size: int = 51, c: float = 35, **params):
        self.block_size = block_size
        self.c = c
        self.params = params

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        raise NotImplementedError("Implement adaptive thresholding (see docstring).")


@step_registry.register("otsu_threshold")
class OtsuThresholdStep:
    name = "otsu_threshold"

    def __init__(self, **params):
        self.params = params

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        raise NotImplementedError("Implement Otsu thresholding (see docstring).")
