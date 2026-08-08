"""Stub. Reference material: surface_normalization.py, steps 10-12
(large-kernel Gaussian blur as a background estimate, float32 division of
original by background, rescale/clip back to uint8).
"""

from __future__ import annotations

from core.domain.image_payload import PipelineContext
from preprocessing.steps.registry import step_registry


@step_registry.register("illumination_normalization")
class IlluminationNormalizationStep:
    name = "illumination_normalization"

    def __init__(self, blur_kernel: int = 95, **params):
        self.blur_kernel = blur_kernel
        self.params = params

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        raise NotImplementedError("Implement illumination normalization (see docstring).")
