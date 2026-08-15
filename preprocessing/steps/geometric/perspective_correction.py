"""Paper-contour detection via Canny + dilate + findContours, minAreaRect,
consistent corner ordering, getPerspectiveTransform + warpPerspective.
Ported from contour_detection.py.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.domain.geometry import Transform
from core.domain.image_payload import PipelineContext
from preprocessing.steps.registry import step_registry


def _order_points(pts: np.ndarray) -> np.ndarray:
    """getPerspectiveTransform needs corners in a CONSISTENT order:
    top-left, top-right, bottom-right, bottom-left. minAreaRect/boxPoints
    doesn't guarantee this order, so sort manually.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left: smallest x+y
    rect[2] = pts[np.argmax(s)]  # bottom-right: largest x+y
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right: smallest y-x
    rect[3] = pts[np.argmax(diff)]  # bottom-left: largest y-x
    return rect


@step_registry.register("perspective_correction")
class PerspectiveCorrectionStep:
    name = "perspective_correction"

    def __init__(self, canny_low: int = 30, canny_high: int = 150, **params):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.params = params

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        image = ctx.payload.image
        if image is None or image.size == 0:
            raise ValueError("perspective_correction received an empty image.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            raise ValueError("No paper contour detected in image.")

        paper_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(paper_contour)
        box = cv2.boxPoints(rect).astype("float32")

        ordered = _order_points(box)
        (tl, tr, br, bl) = ordered

        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = int(max(width_a, width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = int(max(height_a, height_b))

        if max_width <= 0 or max_height <= 0:
            raise ValueError(
                "Detected paper contour has zero width/height; cannot warp."
            )

        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype="float32",
        )

        matrix = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

        debug_img = image.copy()
        cv2.drawContours(debug_img, [paper_contour], -1, (0, 255, 0), 2)
        cv2.drawContours(debug_img, [box.astype(int)], -1, (0, 0, 255), 3)
        for (x, y) in box:
            cv2.circle(debug_img, (int(x), int(y)), 8, (255, 0, 0), -1)

        ctx.payload.image = warped
        ctx.payload.transform = ctx.payload.transform.then(
            Transform(matrix=matrix.astype(np.float64))
        )
        ctx.payload.metadata["perspective_correction"] = {
            "corners": ordered.tolist(),
            "rect_angle": rect[2],
        }
        ctx.payload.debug_images.append(("edges", edges))
        ctx.payload.debug_images.append(("edges_dilated", edges_dilated))
        ctx.payload.debug_images.append(("contour_overlay", debug_img))

        return ctx
