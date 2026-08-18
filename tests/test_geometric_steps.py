"""The two coordinate-changing steps.

These run before everything else and their output *is* the image the UI shows
and every bbox is measured against, so a bad decision here is not recoverable
downstream — and, worse, is invisible: a wrongly cropped page is still a
perfectly valid image.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.domain.image_payload import ImagePayload, PipelineContext
from preprocessing.steps.geometric.deskew import HoughDeskewStep
from preprocessing.steps.geometric.perspective_correction import (
    PerspectiveCorrectionStep,
)


def context(image: np.ndarray) -> PipelineContext:
    return PipelineContext(payload=ImagePayload(image=image))


def blank(height: int = 900, width: int = 700, value: int = 255) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


class TestPerspectiveCorrection:
    def test_crops_a_page_photographed_on_a_dark_desk(self):
        photo = blank(1400, 1000, value=90)
        cv2.rectangle(photo, (70, 90), (930, 1310), (255, 255, 255), -1)

        result = PerspectiveCorrectionStep().apply(context(photo))
        height, width = result.payload.image.shape[:2]

        # The desk is gone; roughly the page remains.
        assert 1150 < height < 1300
        assert 800 < width < 930

    def test_refuses_to_crop_to_a_blob_of_text(self):
        """The guard that matters most.

        On a borderless receipt the largest closed contour is a word, not the
        page edge. Warping to it replaced a 1400x1000 invoice with a 37x57
        thumbnail — and because the result is still a valid image, nothing
        downstream could tell that the invoice had been thrown away.
        """
        page = blank(1400, 1000)
        cv2.putText(page, "receipt", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

        result = PerspectiveCorrectionStep().apply(context(page))

        assert result.payload.image.shape[:2] == (1400, 1000)
        assert "skipped" in result.payload.metadata["perspective_correction"]

    def test_a_blank_page_passes_through(self):
        page = blank()
        result = PerspectiveCorrectionStep().apply(context(page))

        assert result.payload.image.shape[:2] == (900, 700)

    def test_the_floor_is_tunable(self):
        page = blank(1400, 1000)
        cv2.putText(page, "receipt", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

        # A floor of 0 restores the old, unguarded behaviour, which is what
        # makes this a policy rather than a hardcoded rule.
        result = PerspectiveCorrectionStep(min_area_ratio=0.0).apply(context(page))
        assert result.payload.image.shape[:2] != (1400, 1000)

    def test_an_empty_image_is_an_error(self):
        import pytest

        with pytest.raises(ValueError):
            PerspectiveCorrectionStep().apply(context(np.zeros((0, 0, 3), np.uint8)))


class TestDeskew:
    def test_straightens_a_rotated_page(self):
        page = blank(800, 800)
        for y in range(200, 600, 60):
            cv2.line(page, (100, y), (700, y), (0, 0, 0), 3)

        rotated = cv2.warpAffine(
            page,
            cv2.getRotationMatrix2D((400, 400), -5.0, 1.0),
            (800, 800),
            borderValue=(255, 255, 255),
        )

        result = HoughDeskewStep().apply(context(rotated))
        angle = result.payload.metadata["deskew_hough"]["median_angle_deg"]

        # Rotated by -5°, so the correction is about +5° back.
        assert 3.5 < angle < 6.5

    def test_a_page_with_no_straight_lines_passes_through(self):
        """Nothing to measure is not a failure.

        A borderless receipt, or a page of loose handwriting, simply has no long
        straight edges. Raising here would fail the whole extraction over a step
        that had no work to do.
        """
        page = blank(1400, 1000)
        cv2.putText(page, "receipt", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

        result = HoughDeskewStep().apply(context(page))

        assert result.payload.image.shape[:2] == (1400, 1000)
        assert result.payload.metadata["deskew_hough"]["num_lines_used"] == 0

    def test_a_blank_page_passes_through(self):
        result = HoughDeskewStep().apply(context(blank()))
        assert result.payload.image.shape[:2] == (900, 700)

    def test_refuses_to_rotate_beyond_the_maximum_angle(self):
        """The clamp applies to the median, not to each line.

        It used to be per-line, which silently threw away every *vertical* rule
        on a ruled invoice — half the available evidence. Now every segment
        votes and the clamp guards the answer instead: with max_angle_deg at 1°
        the measurement is rejected and the page is left exactly as it came in.
        """
        page = blank(800, 800)
        for x in range(200, 600, 60):
            cv2.line(page, (x, 100), (x + 300, 700), (0, 0, 0), 3)

        result = HoughDeskewStep(max_angle_deg=1.0).apply(context(page))
        meta = result.payload.metadata["deskew_hough"]

        assert result.payload.image.shape[:2] == (800, 800)
        assert "skipped" in meta
        assert meta["num_lines_used"] > 0

    def test_vertical_rules_vote_for_the_same_tilt(self):
        """The reason folding exists.

        A ruled invoice carries as many vertical rules as horizontal ones. The
        old per-line filter discarded every one of them, so a form whose
        verticals are the crisper feature was measured off whichever horizontals
        happened to survive. Folding modulo 90 makes a vertical rule vote for
        the tilt it actually describes.
        """
        page = blank(800, 800)
        for x in range(200, 640, 60):
            cv2.line(page, (x, 120), (x, 680), (0, 0, 0), 3)

        rotated = cv2.warpAffine(
            page,
            cv2.getRotationMatrix2D((400, 400), -4.0, 1.0),
            (800, 800),
            borderValue=(255, 255, 255),
        )

        folded = HoughDeskewStep().apply(context(rotated))
        assert 2.5 < folded.payload.metadata["deskew_hough"]["median_angle_deg"] < 5.5

        # Without folding the same page yields nothing to measure at all.
        unfolded = HoughDeskewStep(fold_to_90=False).apply(context(rotated))
        assert unfolded.payload.metadata["deskew_hough"]["num_lines_used"] == 0

    def test_expanding_the_canvas_keeps_the_corners(self):
        """Rotating into the original frame clips the corners, and on a page
        that fills the photo the clipped corner is where the table's outer rule
        lives."""
        page = blank(800, 600)
        for y in range(150, 500, 50):
            cv2.line(page, (60, y), (540, y), (0, 0, 0), 3)

        rotated = cv2.warpAffine(
            page,
            cv2.getRotationMatrix2D((300, 400), -6.0, 1.0),
            (600, 800),
            borderValue=(255, 255, 255),
        )

        expanded = HoughDeskewStep().apply(context(rotated)).payload.image
        clipped = HoughDeskewStep(expand_canvas=False).apply(context(rotated)).payload.image

        assert expanded.shape[0] > clipped.shape[0]
        assert expanded.shape[1] > clipped.shape[1]
        assert clipped.shape[:2] == (800, 600)

    def test_a_page_already_level_is_not_resampled(self):
        """A sub-degree correction costs sharpness and buys nothing."""
        page = blank(800, 800)
        for y in range(200, 600, 60):
            cv2.line(page, (100, y), (700, y), (0, 0, 0), 3)

        result = HoughDeskewStep().apply(context(page))

        assert np.array_equal(result.payload.image, page)
        assert "skipped" in result.payload.metadata["deskew_hough"]
