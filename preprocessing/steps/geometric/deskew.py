"""Deskew strategies.

`deskew_hough` is the single deskew implementation in this project. It used to
have two rivals -- `estimate_skew_angle`/`deskew_image` in
`table_extraction/extractors/grid_utils.py`, and a third copy in
`temp/table_det.py` -- and the extractor's copy ran a SECOND rotation on an
image this step had already corrected, which put every table bbox in a
different coordinate space than every OCR bbox. Both copies are gone; this is
the only place a pixel is ever rotated.

Two behaviours were taken from the table-detection copy because they measurably
beat what was here:

* **Angle folding (`fold_to_90`).** Voting on the raw Hough angle and throwing
  away anything past `max_angle_deg` discards every *vertical* line -- which on
  a ruled invoice is half the evidence, and on some forms the better half.
  Folding each angle into (-45, 45] by `angle % 90` lets a vertical rule vote
  for the same tilt its horizontal neighbours do.

* **Canvas expansion (`expand_canvas`).** Rotating into the original frame
  clips the corners, and on a page that fills the photo the clipped corner is
  where the table's outer rule lives.

The border fill is deliberately NOT `BORDER_REPLICATE`, which is what
`table_det.py` used: replicating a dark page edge smears a long straight streak
across the new corner, and the grid detector downstream reads exactly that kind
of streak as a table rule.

`deskew_min_area_rect` (minAreaRect over ink/text pixels) is deliberately not
implemented -- see NotImplementedStrategyError.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.domain.geometry import Transform
from core.domain.image_payload import PipelineContext
from core.exceptions import NotImplementedStrategyError
from preprocessing.steps.registry import step_registry


def _fold_angle(angle: float) -> float:
    """Fold a line's angle into (-45, 45].

    A table's vertical rules come back from HoughLinesP at ~90 degrees. They
    describe the same page tilt as the horizontal ones, just measured off the
    other axis, so folding modulo 90 turns them into votes instead of noise.
    """
    folded = angle % 90.0
    if folded > 45.0:
        folded -= 90.0
    return folded


@step_registry.register("deskew_hough")
class HoughDeskewStep:
    name = "deskew_hough"

    def __init__(
        self,
        hough_threshold: int = 100,
        min_line_length: int = 150,
        max_line_gap: int = 10,
        max_angle_deg: float = 20.0,
        blur_kernel: int = 3,
        canny_low: int = 50,
        canny_high: int = 150,
        fold_to_90: bool = True,
        expand_canvas: bool = True,
        min_angle_deg: float = 0.1,
        **params,
    ):
        if blur_kernel < 1 or blur_kernel % 2 == 0:
            raise ValueError(
                f"deskew_hough blur_kernel must be an odd positive integer, got {blur_kernel}."
            )

        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

        # A sanity clamp, applied to the *median* rather than to each line. A
        # page is photographed askew by a few degrees; a median past this is a
        # measurement failure, not a tilt, and acting on it would destroy the
        # page rather than straighten it.
        self.max_angle_deg = max_angle_deg

        self.blur_kernel = blur_kernel
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.fold_to_90 = fold_to_90
        self.expand_canvas = expand_canvas

        # Below this there is nothing to gain and a resample to lose.
        self.min_angle_deg = min_angle_deg

        self.params = params

    # ------------------------------------------------------------------

    def estimate_angle(self, image: np.ndarray) -> tuple[float, int, np.ndarray | None]:
        """Median page tilt in degrees, the number of segments that voted, and
        an overlay of those segments (None when debug images are not wanted).

        Public because `tools/compare_deskew.py` measures the angle without
        rotating anything.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high, apertureSize=3)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        angles: list[float] = []
        segments: list[tuple[int, int, int, int]] = []

        for line in lines if lines is not None else []:
            x1, y1, x2, y2 = (int(v) for v in line.reshape(-1)[:4])
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

            if self.fold_to_90:
                angle = _fold_angle(angle)
            elif abs(angle) >= self.max_angle_deg:
                # Legacy behaviour: near-horizontal segments only.
                continue

            angles.append(angle)
            segments.append((x1, y1, x2, y2))

        if not angles:
            return 0.0, 0, None

        median = float(np.median(angles))

        overlay = image.copy()
        if overlay.ndim == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in segments:
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return median, len(angles), overlay

    def rotation_matrix(self, image: np.ndarray, angle: float) -> tuple[np.ndarray, int, int]:
        """The 2x3 affine that levels `image`, plus the output size.

        When `expand_canvas` is on, the translation that keeps the whole rotated
        page inside the new frame is folded into the matrix itself -- which is
        also why the `Transform` recorded on the payload stays exact: the shift
        is part of the matrix, not applied separately afterwards.
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        if not self.expand_canvas:
            return matrix, w, h

        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        matrix[0, 2] += (new_w / 2.0) - center[0]
        matrix[1, 2] += (new_h / 2.0) - center[1]
        return matrix, new_w, new_h

    def rotate(self, image: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
        """Level `image` by `angle`. Returns (rotated image, 2x3 matrix used)."""
        matrix, out_w, out_h = self.rotation_matrix(image, angle)
        border_value = (255, 255, 255) if image.ndim == 3 else 255
        rotated = cv2.warpAffine(
            image,
            matrix,
            (out_w, out_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )
        return rotated, matrix

    # ------------------------------------------------------------------

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        image = ctx.payload.image
        if image is None or image.size == 0:
            raise ValueError("deskew_hough received an empty image.")

        median_angle, num_lines, overlay = self.estimate_angle(image)

        if num_lines == 0:
            # Nothing measurable to rotate by. A borderless receipt, or a page of
            # loose handwriting with no ruled lines, simply has no long straight
            # edges -- that is a normal invoice, not a failure, and refusing it
            # here would fail the whole extraction over a step that had nothing
            # to do. Pass the image through untouched and say so in the metadata.
            ctx.payload.metadata[self.name] = {
                "median_angle_deg": 0.0,
                "num_lines_used": 0,
                "skipped": "no line segments detected",
            }
            return ctx

        if abs(median_angle) > self.max_angle_deg:
            # See max_angle_deg in __init__: a median this large is a bad
            # measurement, and rotating by it would wreck a page that was
            # probably fine.
            ctx.payload.metadata[self.name] = {
                "median_angle_deg": median_angle,
                "num_lines_used": num_lines,
                "skipped": (
                    f"median angle {median_angle:.2f} exceeds the "
                    f"{self.max_angle_deg} degree limit; image passed through unchanged"
                ),
            }
            return ctx

        if abs(median_angle) < self.min_angle_deg:
            # Straight enough. Resampling would cost sharpness for nothing.
            ctx.payload.metadata[self.name] = {
                "median_angle_deg": median_angle,
                "num_lines_used": num_lines,
                "skipped": f"already within {self.min_angle_deg} degrees of level",
            }
            return ctx

        deskewed, matrix = self.rotate(image, median_angle)

        ctx.payload.image = deskewed
        homogeneous = np.vstack([matrix, [0.0, 0.0, 1.0]]).astype(np.float64)
        ctx.payload.transform = ctx.payload.transform.then(Transform(matrix=homogeneous))
        ctx.payload.metadata[self.name] = {
            "median_angle_deg": median_angle,
            "num_lines_used": num_lines,
            "folded": self.fold_to_90,
            "expanded": self.expand_canvas,
            "output_size": [int(deskewed.shape[1]), int(deskewed.shape[0])],
        }

        if overlay is not None and ctx.payload.metadata.get("collect_debug_images"):
            ctx.payload.debug_images.append(("hough_lines", overlay))

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
