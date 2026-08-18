"""Cell detection and the input contract it depends on.

The extractors are where the pipeline's central invariant was being broken:
they used to deskew and threshold the image a second time, which put every cell
bbox in a different coordinate space from every OCR bbox. These tests pin the
contract that replaced that.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.domain.image_payload import ImagePayload
from table_extraction.extractors.contour_based_extractor import ContourBasedTableExtractor
from table_extraction.extractors.contour_utils import (
    find_table_bounds,
    get_table_cell_bounds,
    remove_stacked_text_blobs,
)
from table_extraction.extractors.grid_line_extractor import GridLineTableExtractor


def ruled_grid(
    rows: int = 4,
    cols: int = 3,
    cell_w: int = 160,
    cell_h: int = 70,
    origin: tuple[int, int] = (80, 120),
    canvas: tuple[int, int] = (700, 800),
) -> np.ndarray:
    """A binary page with ink at 255 -- what the table branch produces."""
    page = np.zeros(canvas, np.uint8)
    x0, y0 = origin

    for r in range(rows + 1):
        y = y0 + r * cell_h
        cv2.line(page, (x0, y), (x0 + cols * cell_w, y), 255, 3)
    for c in range(cols + 1):
        x = x0 + c * cell_w
        cv2.line(page, (x, y0), (x, y0 + rows * cell_h), 255, 3)

    return page


def payload(image: np.ndarray, ink_is_white: bool = True) -> ImagePayload:
    p = ImagePayload(image=image)
    p.metadata["binary_ink_is_white"] = ink_is_white
    return p


class TestBinaryInputContract:
    """Extractors state what they need instead of quietly fixing it up.

    Re-thresholding an already-binary image is not a no-op: it produces a
    plausible mask with the polarity and stroke widths subtly wrong, and nothing
    downstream can tell. The fix is always a config change, so the error names
    the config.
    """

    @pytest.mark.parametrize(
        "extractor", [ContourBasedTableExtractor(), GridLineTableExtractor()]
    )
    def test_a_colour_image_is_refused_with_the_config_that_fixes_it(self, extractor):
        colour = np.zeros((200, 200, 3), np.uint8)

        with pytest.raises(ValueError, match="table_photometric_steps"):
            extractor.extract(payload(colour))

    @pytest.mark.parametrize(
        "extractor", [ContourBasedTableExtractor(), GridLineTableExtractor()]
    )
    def test_a_grayscale_image_is_refused(self, extractor):
        gray = np.full((200, 200), 128, np.uint8)
        gray[50:60, :] = 200

        with pytest.raises(ValueError, match="binary"):
            extractor.extract(payload(gray))

    @pytest.mark.parametrize(
        "extractor", [ContourBasedTableExtractor(), GridLineTableExtractor()]
    )
    def test_the_wrong_polarity_is_refused(self, extractor):
        """Ink at 0 would make morphology reconstruct the paper, not the rules."""
        inked = cv2.bitwise_not(ruled_grid())

        with pytest.raises(ValueError, match="ink at 255"):
            extractor.extract(payload(inked, ink_is_white=False))


class TestContourExtractor:
    def test_finds_every_cell_of_a_clean_grid(self):
        result = ContourBasedTableExtractor().extract(payload(ruled_grid(rows=4, cols=3)))

        assert len(result.tables) == 1
        assert len(result.tables[0].cells) == 12

    def test_does_not_move_the_image(self):
        """The invariant the extractor used to break.

        It ran its own deskew, so on any page with measurable tilt every cell it
        reported was in a rotated space that no OCR bbox shared. Cells must land
        inside the image they were measured in.
        """
        page = ruled_grid(canvas=(700, 800))
        result = ContourBasedTableExtractor().extract(payload(page))

        height, width = page.shape[:2]
        for cell in result.tables[0].cells:
            assert 0 <= cell.bbox.x
            assert 0 <= cell.bbox.y
            assert cell.bbox.x + cell.bbox.w <= width
            assert cell.bbox.y + cell.bbox.h <= height

    def test_rows_and_columns_are_indexed_top_to_bottom_left_to_right(self):
        result = ContourBasedTableExtractor().extract(payload(ruled_grid(rows=3, cols=3)))
        cells = result.tables[0].cells

        by_row: dict[int, list] = {}
        for cell in cells:
            by_row.setdefault(cell.row, []).append(cell)

        for row_cells in by_row.values():
            ordered = sorted(row_cells, key=lambda c: c.col)
            xs = [c.bbox.x for c in ordered]
            assert xs == sorted(xs)

        tops = [min(c.bbox.y for c in by_row[r]) for r in sorted(by_row)]
        assert tops == sorted(tops)

    def test_an_empty_page_yields_no_tables(self):
        result = ContourBasedTableExtractor().extract(payload(np.zeros((300, 300), np.uint8)))
        assert result.tables == []


class TestStackFilter:
    """The piece the line-based extractor has no equivalent of.

    Several lines of handwriting stacked above each other -- similar widths,
    overlapping in x -- look exactly like a set of horizontal rules after
    morphological reconstruction.
    """

    def test_stacked_similar_blobs_are_removed(self):
        mask = np.zeros((300, 400), np.uint8)
        for y in range(60, 160, 25):
            cv2.rectangle(mask, (100, y), (260, y + 4), 255, -1)

        clean, removed = remove_stacked_text_blobs(mask, is_horizontal=True)

        assert removed.any()
        assert not clean.any()

    def test_a_blob_crossing_a_vertical_rule_is_kept(self):
        """Positive evidence of a real grid outranks the text pattern.

        Without this, a genuinely short table rule inside a dense block of
        writing would be thrown away with the writing.
        """
        mask = np.zeros((300, 400), np.uint8)
        for y in range(60, 160, 25):
            cv2.rectangle(mask, (100, y), (260, y + 4), 255, -1)

        verticals = np.zeros((300, 400), np.uint8)
        cv2.rectangle(verticals, (170, 40), (174, 200), 255, -1)

        clean, _ = remove_stacked_text_blobs(mask, is_horizontal=True, vertical_mask=verticals)

        assert clean.any()

    def test_vertical_masks_pass_through(self):
        mask = np.zeros((300, 400), np.uint8)
        cv2.rectangle(mask, (100, 40), (104, 260), 255, -1)

        clean, removed = remove_stacked_text_blobs(mask, is_horizontal=False)

        assert clean.any()
        assert not removed.any()


class TestTableBounds:
    def test_picks_the_largest_connected_group(self):
        """An invoice page holds several disconnected groups of ruled boxes --
        header captions, the item table, the totals strip -- and the item table
        is by a wide margin the biggest."""
        page = np.zeros((800, 700), np.uint8)
        cv2.rectangle(page, (60, 40), (300, 90), 255, 3)  # a lone header box

        grid = ruled_grid(rows=4, cols=3, origin=(80, 200), canvas=(800, 700))
        page = cv2.bitwise_or(page, grid)

        cells = get_table_cell_bounds(page)
        bounds = find_table_bounds(cells, proximity=3)

        assert bounds is not None
        assert bounds["y1"] > 150  # the header box is not part of the table
        assert len(bounds["cells"]) == 12

    def test_no_cells_means_no_bounds(self):
        assert find_table_bounds([]) is None
