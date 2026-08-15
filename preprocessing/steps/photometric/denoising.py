"""Denoise strategies, all ported from surface_normalization.py's
comparison of median blur, Gaussian blur, non-local means (NLM), and
bilateral filtering.
"""

from __future__ import annotations

import cv2

from core.domain.image_payload import PipelineContext
from preprocessing.steps.registry import step_registry


def _require_single_channel(image, step_name: str) -> None:
    if image is None or image.size == 0:
        raise ValueError(f"{step_name} received an empty image.")
    if image.ndim != 2:
        raise ValueError(
            f"{step_name} expects a single-channel (grayscale) image; "
            "run channel_selection first."
        )


@step_registry.register("median_blur")
class MedianBlurStep:
    name = "median_blur"

    def __init__(self, k: int = 5, **params):
        if k <= 0 or k % 2 == 0:
            raise ValueError(f"median_blur k must be a positive odd integer, got {k}.")
        self.k = k
        self.params = params

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        gray = ctx.payload.image
        _require_single_channel(gray, self.name)
        ctx.payload.image = cv2.medianBlur(gray, self.k)
        return ctx


@step_registry.register("gaussian_blur")
class GaussianBlurStep:
    name = "gaussian_blur"

    def __init__(self, ksize: tuple = (5, 5), sigma: float = 0, **params):
        self.ksize = tuple(ksize)
        self.sigma = sigma
        self.params = params

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        gray = ctx.payload.image
        _require_single_channel(gray, self.name)
        ctx.payload.image = cv2.GaussianBlur(gray, self.ksize, self.sigma)
        return ctx


@step_registry.register("nlm_denoise")
class NLMDenoiseStep:
    name = "nlm_denoise"

    def __init__(
        self,
        h: float = 10,
        template_window_size: int = 7,
        search_window_size: int = 21,
        **params,
    ):
        self.h = h
        self.template_window_size = template_window_size
        self.search_window_size = search_window_size
        self.params = params

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        gray = ctx.payload.image
        _require_single_channel(gray, self.name)
        ctx.payload.image = cv2.fastNlMeansDenoising(
            gray,
            h=self.h,
            templateWindowSize=self.template_window_size,
            searchWindowSize=self.search_window_size,
        )
        return ctx


@step_registry.register("bilateral_filter")
class BilateralFilterStep:
    name = "bilateral_filter"

    def __init__(self, d: int = 20, sigma_color: float = 25, sigma_space: float = 50, **params):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.params = params

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        gray = ctx.payload.image
        _require_single_channel(gray, self.name)
        ctx.payload.image = cv2.bilateralFilter(
            gray, d=self.d, sigmaColor=self.sigma_color, sigmaSpace=self.sigma_space
        )
        return ctx
