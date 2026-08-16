"""Morphological cleanup, ported from morphology.py: invert since ink=0/bg=255,
erode/dilate/open/close with a small kernel, then invert back. The
operation is chosen via the `operation` param (open/close/erode/dilate),
not a separate registry name per operation -- config can already switch
between all four with a one-line YAML edit.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.domain.image_payload import PipelineContext
from preprocessing.steps.registry import step_registry

_VALID_OPERATIONS = {"open", "close", "erode", "dilate"}


@step_registry.register("morphological_cleanup")
class MorphologicalCleanupStep:
    name = "morphological_cleanup"

    def __init__(self, operation: str = "open", kernel_size: int = 2, **params):
        if operation not in _VALID_OPERATIONS:
            raise ValueError(
                f"Unknown morphology operation '{operation}'; expected one of "
                f"{sorted(_VALID_OPERATIONS)}."
            )
        if kernel_size <= 0:
            raise ValueError(
                f"morphological_cleanup kernel_size must be positive, got {kernel_size}."
            )
        self.operation = operation
        self.kernel_size = kernel_size
        self.params = params

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        binary = ctx.payload.image
        if binary is None or binary.size == 0:
            raise ValueError("morphological_cleanup received an empty image.")
        if binary.ndim != 2:
            raise ValueError(
                "morphological_cleanup expects a single-channel (binary) image."
            )

        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        inv = cv2.bitwise_not(binary)

        if self.operation == "erode":
            processed_inv = cv2.erode(inv, kernel, iterations=1)
        elif self.operation == "dilate":
            processed_inv = cv2.dilate(inv, kernel, iterations=1)
        elif self.operation == "open":
            processed_inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel)
        else:  # close
            processed_inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel)

        ctx.payload.image = cv2.bitwise_not(processed_inv)
        return ctx
