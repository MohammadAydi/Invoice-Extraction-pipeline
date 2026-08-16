"""Deskew strategies. `deskew_hough` is ported from contour_detection.py
(HoughLinesP on the warped image, filter near-horizontal segments, median
angle, getRotationMatrix2D + warpAffine with a white borderValue).
`deskew_min_area_rect` (minAreaRect over ink/text pixels) is deliberately
not implemented -- see NotImplementedStrategyError.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.domain.geometry import Transform
from core.domain.image_payload import PipelineContext
from core.exceptions import NotImplementedStrategyError
from preprocessing.steps.registry import step_registry


@step_registry.register("deskew_hough")
class HoughDeskewStep:
    name = "deskew_hough"

    def __init__(
        self,
        hough_threshold: int = 100,
        min_line_length: int = 150,
        max_line_gap: int = 10,
        max_angle_deg: float = 20.0,
        **params,
    ):
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.max_angle_deg = max_angle_deg
        self.params = params

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        image = ctx.payload.image
        if image is None or image.size == 0:
            raise ValueError("deskew_hough received an empty image.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )
        angles = []
        debug_img = image.copy()
        for line in lines if lines is not None else []:
            x1, y1, x2, y2 = line.reshape(-1)[:4]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < self.max_angle_deg:
                angles.append(angle)
                cv2.line(debug_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        if not angles:
            # Nothing measurable to rotate by. A borderless receipt, or a page of
            # loose handwriting with no ruled lines, simply has no long straight
            # edges -- that is a normal invoice, not a failure, and refusing it
            # here would fail the whole extraction over a step that had nothing
            # to do. Pass the image through untouched and say so in the metadata.
            ctx.payload.metadata[self.name] = {
                "median_angle_deg": 0.0,
                "num_lines_used": 0,
                "skipped": "no near-horizontal line segments detected",
            }
            return ctx

        median_angle = float(np.median(angles))

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)

        border_value = (255, 255, 255) if image.ndim == 3 else 255
        deskewed = cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )

        ctx.payload.image = deskewed
        homogeneous = np.vstack([rotation_matrix, [0.0, 0.0, 1.0]]).astype(np.float64)
        ctx.payload.transform = ctx.payload.transform.then(Transform(matrix=homogeneous))
        ctx.payload.metadata["deskew_hough"] = {
            "median_angle_deg": median_angle,
            "num_lines_used": len(angles),
        }
        ctx.payload.debug_images.append(("hough_lines", debug_img))

        return ctx


@step_registry.register("deskew_min_area_rect")
class MinAreaRectDeskewStep:
    name = "deskew_min_area_rect"

    def __init__(self, **params):
        self.params = params

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        raise NotImplementedStrategyError(
            self.name, "deskew_min_area_rect", "Use 'deskew_hough' instead."
        )
