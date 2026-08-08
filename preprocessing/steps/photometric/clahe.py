"""Stub. Reference material: surface_normalization.py (cv2.createCLAHE
with clipLimit / tileGridSize, contrasted against plain equalizeHist)."""

from __future__ import annotations

from core.domain.image_payload import PipelineContext
from preprocessing.steps.registry import step_registry


@step_registry.register("clahe")
class CLAHEStep:
    name = "clahe"

    def __init__(self, clip_limit: float = 2.5, tile_grid_size: tuple = (8, 8), **params):
        self.clip_limit = clip_limit
        self.tile_grid_size = tuple(tile_grid_size)
        self.params = params

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        raise NotImplementedError("Implement CLAHE (see docstring).")
