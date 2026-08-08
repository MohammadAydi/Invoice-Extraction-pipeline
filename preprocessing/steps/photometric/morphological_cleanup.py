"""Stub. Reference material: morphology.py (invert since ink=0/bg=255,
erode/dilate/open/close with a small kernel, connectedComponentsWithStats
to tell specks from real strokes, then invert back)."""

from __future__ import annotations

from core.domain.image_payload import PipelineContext
from preprocessing.steps.registry import step_registry


@step_registry.register("morphological_cleanup")
class MorphologicalCleanupStep:
    name = "morphological_cleanup"

    def __init__(self, operation: str = "open", kernel_size: int = 2, **params):
        self.operation = operation
        self.kernel_size = kernel_size
        self.params = params

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        raise NotImplementedError("Implement morphological cleanup (see docstring).")
