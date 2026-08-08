"""Stub. Reference material: contour_detection.py (HoughLinesP on the
warped image, filter near-horizontal segments, median angle,
getRotationMatrix2D + warpAffine with a white borderValue).

Must compose its rotation into ctx.payload.transform, same as
perspective_correction.
"""

from __future__ import annotations

from core.domain.image_payload import PipelineContext
from preprocessing.steps.registry import step_registry


@step_registry.register("deskew")
class DeskewStep:
    name = "deskew"

    def __init__(self, **params):
        self.params = params

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        raise NotImplementedError("Implement deskew (see docstring).")
