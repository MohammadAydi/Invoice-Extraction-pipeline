"""Stub. Reference material: surface_normalization.py (cv2.bilateralFilter
with d / sigmaColor / sigmaSpace, compared against median blur, Gaussian
blur, and fastNlMeansDenoising)."""

from __future__ import annotations

from core.domain.image_payload import PipelineContext
from preprocessing.steps.registry import step_registry


@step_registry.register("bilateral_filter")
class BilateralFilterStep:
    name = "bilateral_filter"

    def __init__(self, d: int = 20, sigma_color: float = 25, sigma_space: float = 50, **params):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.params = params

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        raise NotImplementedError("Implement bilateral filtering (see docstring).")
