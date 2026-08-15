"""Contrast enhancement strategies, both ported from surface_normalization.py:
`clahe` (cv2.createCLAHE with clipLimit/tileGridSize) and
`plain_equalization` (cv2.equalizeHist), kept side by side to show WHY the
"limited" part of CLAHE matters.
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


@step_registry.register("clahe")
class CLAHEStep:
    name = "clahe"

    def __init__(self, clip_limit: float = 2.5, tile_grid_size: tuple = (8, 8), **params):
        self.clip_limit = clip_limit
        self.tile_grid_size = tuple(tile_grid_size)
        self.params = params

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        gray = ctx.payload.image
        _require_single_channel(gray, self.name)

        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        ctx.payload.image = clahe.apply(gray)
        return ctx


@step_registry.register("plain_equalization")
class PlainEqualizationStep:
    name = "plain_equalization"

    def __init__(self, **params):
        self.params = params

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        gray = ctx.payload.image
        _require_single_channel(gray, self.name)

        ctx.payload.image = cv2.equalizeHist(gray)
        return ctx
