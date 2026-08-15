"""Illumination normalization strategies. `illumination_normalization_blur_divide`
is ported from surface_normalization.py steps 10-12 (large-kernel Gaussian
blur as a background estimate, float32 division of original by background,
rescale/clip back to uint8). `illumination_normalization_blackhat`
(morphological black-hat/top-hat filtering) is deliberately not
implemented -- see NotImplementedStrategyError.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.domain.image_payload import PipelineContext
from core.exceptions import NotImplementedStrategyError
from preprocessing.steps.registry import step_registry


def _require_single_channel(image, step_name: str) -> None:
    if image is None or image.size == 0:
        raise ValueError(f"{step_name} received an empty image.")
    if image.ndim != 2:
        raise ValueError(
            f"{step_name} expects a single-channel (grayscale) image; "
            "run channel_selection first."
        )


@step_registry.register("illumination_normalization_blur_divide")
class BlurDivideIlluminationStep:
    name = "illumination_normalization_blur_divide"

    def __init__(self, blur_kernel: int = 95, **params):
        if blur_kernel <= 0 or blur_kernel % 2 == 0:
            raise ValueError(
                f"illumination_normalization_blur_divide blur_kernel must be "
                f"a positive odd integer, got {blur_kernel}."
            )
        self.blur_kernel = blur_kernel
        self.params = params

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        gray = ctx.payload.image
        _require_single_channel(gray, self.name)

        background = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)

        gray_f = gray.astype(np.float32)
        bg_f = background.astype(np.float32)
        normalized = (gray_f / (bg_f + 1e-6)) * 255
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        heatmap_before = cv2.applyColorMap(
            cv2.GaussianBlur(gray, (101, 101), 0), cv2.COLORMAP_JET
        )
        heatmap_after = cv2.applyColorMap(
            cv2.GaussianBlur(normalized, (101, 101), 0), cv2.COLORMAP_JET
        )

        ctx.payload.image = normalized
        ctx.payload.debug_images.append(("background_estimate", background))
        ctx.payload.debug_images.append(("illumination_heatmap_before", heatmap_before))
        ctx.payload.debug_images.append(("illumination_heatmap_after", heatmap_after))

        return ctx


@step_registry.register("illumination_normalization_blackhat")
class BlackHatIlluminationStep:
    name = "illumination_normalization_blackhat"

    def __init__(self, **params):
        self.params = params

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        raise NotImplementedStrategyError(
            self.name,
            "illumination_normalization_blackhat",
            "Use 'illumination_normalization_blur_divide' instead.",
        )
