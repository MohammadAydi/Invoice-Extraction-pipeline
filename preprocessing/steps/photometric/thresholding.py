"""Thresholding strategies, all ported from thresholding.py: `fixed_threshold`
(a single fixed T), `otsu_threshold` (THRESH_BINARY + THRESH_OTSU), and
`adaptive_threshold` (ADAPTIVE_THRESH_GAUSSIAN_C).

Three strategies are registered to show the Strategy pattern in action
within one "family" -- config picks which one runs via `name`, and all can
coexist in the registry without any of them knowing about the others.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from core.domain.image_payload import PipelineContext
from preprocessing.steps.registry import step_registry

logger = logging.getLogger(__name__)


def _require_single_channel(image, step_name: str) -> None:
    if image is None or image.size == 0:
        raise ValueError(f"{step_name} received an empty image.")
    if image.ndim != 2:
        raise ValueError(
            f"{step_name} expects a single-channel (grayscale) image; "
            "run channel_selection first."
        )


def _add_histogram(ctx: PipelineContext, gray: np.ndarray) -> None:
    """Queue a brightness histogram alongside this step's output image.

    Only when someone is going to look at it. Rendering a matplotlib figure
    costs more than the thresholding itself, and under the HTTP service -- which
    runs with debug output off -- every one of them would be drawn and thrown
    away, once per invoice.
    """
    if not ctx.payload.metadata.get("collect_debug_images"):
        return

    try:
        # Imported here, not at module scope: matplotlib is a heavyweight
        # dependency for a diagnostic picture, and the service must start on a
        # machine that has not installed it.
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError:
        logger.info("matplotlib is not installed; skipping the threshold histogram.")
        return

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    fig = Figure(figsize=(6, 3))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot(hist)
    ax.set_xlim([0, 255])
    ax.set_xlabel("Brightness")
    ax.set_ylabel("Number of pixels")
    fig.tight_layout()
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())

    ctx.payload.debug_images.append(("histogram", cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)))


@step_registry.register("fixed_threshold")
class FixedThresholdStep:
    name = "fixed_threshold"

    def __init__(self, t: int = 127, **params):
        self.t = t
        self.params = params

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        gray = ctx.payload.image
        _require_single_channel(gray, self.name)

        _, out = cv2.threshold(gray, self.t, 255, cv2.THRESH_BINARY)

        ctx.payload.image = out
        _add_histogram(ctx, gray)
        return ctx


@step_registry.register("otsu_threshold")
class OtsuThresholdStep:
    name = "otsu_threshold"

    def __init__(self, **params):
        self.params = params

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        gray = ctx.payload.image
        _require_single_channel(gray, self.name)

        otsu_thresh, out = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        ctx.payload.image = out
        ctx.payload.metadata["otsu_threshold"] = {"chosen_threshold": otsu_thresh}
        _add_histogram(ctx, gray)
        return ctx


@step_registry.register("adaptive_threshold")
class AdaptiveThresholdStep:
    name = "adaptive_threshold"

    def __init__(self, block_size: int = 51, c: float = 35, **params):
        if block_size < 3 or block_size % 2 == 0:
            raise ValueError(
                f"adaptive_threshold block_size must be an odd integer >= 3, "
                f"got {block_size}."
            )
        self.block_size = block_size
        self.c = c
        self.params = params

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        gray = ctx.payload.image
        _require_single_channel(gray, self.name)

        out = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=self.block_size,
            C=self.c,
        )

        ctx.payload.image = out
        _add_histogram(ctx, gray)
        return ctx
