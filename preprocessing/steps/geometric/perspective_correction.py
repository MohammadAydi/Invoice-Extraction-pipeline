"""Stub. Reference material: contour_detection.py / perspective_warp.py
(paper-contour detection via Canny + dilate + findContours, minAreaRect,
consistent corner ordering, getPerspectiveTransform + warpPerspective).

Whoever implements this must:
  1. Replace ctx.payload.image with the warped image.
  2. Compose the resulting perspective Transform into ctx.payload.transform
     via `ctx.payload.transform = ctx.payload.transform.then(new_transform)`.
"""

from __future__ import annotations

from core.domain.image_payload import PipelineContext
from preprocessing.steps.registry import step_registry


@step_registry.register("perspective_correction")
class PerspectiveCorrectionStep:
    name = "perspective_correction"

    def __init__(self, **params):
        self.params = params

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        raise NotImplementedError("Implement perspective correction (see docstring).")
