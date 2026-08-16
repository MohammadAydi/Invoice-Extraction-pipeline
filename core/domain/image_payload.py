"""The unit of data passed between preprocessing steps."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.domain.geometry import Transform


@dataclass
class ImagePayload:
    image: np.ndarray
    transform: Transform = field(default_factory=Transform.identity)
    applied_steps: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    # Extra diagnostic images a step wants preserved alongside its main
    # output (e.g. edge maps, heatmaps, overlays, histograms) -- not part
    # of the image/transform flow, just queued for PreprocessingPipeline
    # to write to disk when a debug_dir is set, then cleared per step.
    debug_images: list[tuple[str, np.ndarray]] = field(default_factory=list)


@dataclass
class PipelineContext:
    """Mutable context threaded through a single PreprocessingPipeline run.

    Steps read `payload`, do their work, and return a (possibly new)
    PipelineContext. The pipeline itself doesn't care whether a step
    mutates and returns `self` or builds a fresh context -- it only calls
    `step.apply(ctx)` and uses the return value.
    """
    
    payload: ImagePayload
