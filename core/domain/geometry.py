"""Geometric primitives shared across the pipeline.

Only two preprocessing steps in this project ever change pixel coordinates:
perspective correction and deskew (rotation). Every other preprocessing step
(channel selection, illumination normalization, CLAHE, bilateral filtering,
thresholding, morphology) is purely photometric and preserves coordinates
exactly.

Because of this, geometric correction runs ONCE, upstream of the OCR and
table-extraction photometric branches (see `orchestration.PipelineOrchestrator`).
Both branches -- and therefore the OCR fragments and table cells they
produce -- share a single coordinate space: that of the geometrically
corrected image, which is also the exact image the UI displays on both the
left (plain) and right (with overlays). No bounding box ever needs to be
re-mapped between spaces.

Transform composition is still tracked for provenance/debugging and for the
audit-friendly config/run snapshot, even though downstream stages don't need
to invert it under current UI requirements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Transform:
    """An invertible 3x3 homogeneous transform (covers affine + perspective)."""

    matrix: np.ndarray  # shape (3, 3)

    @staticmethod
    def identity() -> "Transform":
        return Transform(matrix=np.eye(3, dtype=np.float64))

    def then(self, other: "Transform") -> "Transform":
        """Compose: apply `self` first, then `other`."""
        return Transform(matrix=other.matrix @ self.matrix)

    def apply_to_point(self, x: float, y: float) -> tuple[float, float]:
        vec = self.matrix @ np.array([x, y, 1.0])
        return float(vec[0] / vec[2]), float(vec[1] / vec[2])

    def invert(self) -> "Transform":
        return Transform(matrix=np.linalg.inv(self.matrix))


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in the single canonical coordinate space: the
    geometrically corrected (perspective-warped + deskewed) display image.
    """

    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)
