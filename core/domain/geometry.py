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
    """Axis-aligned box in the single canonical coordinate space."""
    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    # --- الخصائص الجديدة المطلوبة لخوارزميات الـ Mapping ---
    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def diagonal(self) -> float:
        return (self.w ** 2 + self.h ** 2) ** 0.5

    def intersection_area(self, other: "BoundingBox") -> float:
        ix1, iy1 = max(self.x, other.x), max(self.y, other.y)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return (ix2 - ix1) * (iy2 - iy1)

    def iou(self, other: "BoundingBox") -> float:
        inter = self.intersection_area(other)
        if inter == 0.0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def ioa(self, other: "BoundingBox") -> float:
        """Fraction of THIS box's area covered by `other`."""
        if self.area == 0:
            return 0.0
        return self.intersection_area(other) / self.area

    def center_distance(self, other: "BoundingBox") -> float:
        cx1, cy1 = self.center
        cx2, cy2 = other.center
        return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

    def center_in(self, other: "BoundingBox") -> bool:
        cx, cy = self.center
        return other.x <= cx <= other.x2 and other.y <= cy <= other.y2
