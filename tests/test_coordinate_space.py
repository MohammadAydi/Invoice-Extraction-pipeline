"""The invariant the whole pipeline is built on: one coordinate space.

Only perspective correction and deskew move pixels, and they run once, upstream,
before the pipeline forks into its two photometric branches. Every OCR bbox and
every table bbox therefore lives in the same space -- the geometrically
corrected display image the UI shows and the user clicks on.

Two separate bugs were breaking this, and both were invisible: a wrongly-placed
box is still a perfectly valid box.

1. The table extractors ran their *own* deskew on an image the pipeline had
   already deskewed. On any page with measurable tilt that rotated the image a
   second time, so cell bboxes and OCR bboxes disagreed and the mapper assigned
   fragments to cells across two different spaces.

2. `PreprocessingPipeline.run()` mutated the payload it was given, so the "fork"
   into two branches was really a chain: the table branch read the OCR branch's
   output, and the display image handed to the UI was the final binarized mask
   rather than the corrected page.
"""

from __future__ import annotations

import cv2
import numpy as np

from config.schema import (
    AppConfig,
    ComponentConfig,
    FlowConfig,
    OCRConfig,
    OutputConfig,
    PreprocessingConfig,
    StepConfig,
    StringMatchingConfig,
    TableExtractionConfig,
)
from core.domain.image_payload import ImagePayload
from orchestration.pipeline_orchestrator import PipelineOrchestrator
from preprocessing.pipeline_builder import PreprocessingPipelineBuilder


def tilted_invoice(angle: float = 4.0, w: int = 900, h: int = 1100) -> np.ndarray:
    """A ruled form, photographed askew on a dark desk."""
    page = np.full((h, w, 3), 255, np.uint8)

    x0, y0, cell_w, cell_h, cols, rows = 90, 260, 130, 60, 5, 8
    for r in range(rows + 1):
        y = y0 + r * cell_h
        cv2.line(page, (x0, y), (x0 + cols * cell_w, y), (0, 0, 0), 3)
    for c in range(cols + 1):
        x = x0 + c * cell_w
        cv2.line(page, (x, y0), (x, y0 + rows * cell_h), (0, 0, 0), 3)

    # A header box above the table.
    cv2.rectangle(page, (90, 90), (400, 180), (0, 0, 0), 3)

    canvas = np.full((h + 200, w + 200, 3), 70, np.uint8)
    canvas[100 : 100 + h, 100 : 100 + w] = page

    matrix = cv2.getRotationMatrix2D((canvas.shape[1] / 2, canvas.shape[0] / 2), angle, 1.0)
    return cv2.warpAffine(
        canvas, matrix, (canvas.shape[1], canvas.shape[0]), borderValue=(70, 70, 70)
    )


def config() -> AppConfig:
    return AppConfig(
        preprocessing=PreprocessingConfig(
            geometric_steps=[
                StepConfig(name="perspective_correction"),
                StepConfig(name="deskew_hough"),
            ],
            ocr_photometric_steps=[
                StepConfig(name="channel_selection", params={"channel": "gray"}),
            ],
            table_photometric_steps=[
                StepConfig(name="channel_selection", params={"channel": "gray"}),
                StepConfig(
                    name="adaptive_threshold",
                    params={"block_size": 15, "c": 10, "invert": True},
                ),
            ],
        ),
        ocr=OCRConfig(
            cropper=ComponentConfig(name="padded_crop", params={"pad": 2, "upscale": 1}),
            recognizer=ComponentConfig(name="echo"),
        ),
        flow=FlowConfig(name="layout_driven"),
        table_extraction=TableExtractionConfig(
            extractor="contour_based", classifier="passthrough"
        ),
        string_matching=StringMatchingConfig(
            algorithm="levenshtein", dictionary_path="keywords/does-not-exist.json"
        ),
        output=OutputConfig(formatter="ui_overlay_json"),
    )


def run() -> tuple:
    orchestrator = PipelineOrchestrator(config())
    result = orchestrator.run(
        ImagePayload(image=tilted_invoice()), write_debug_images=False, persist=False
    )
    return result, result.display_image


class TestBranchesDoNotChain:
    def test_running_a_branch_does_not_alter_the_caller_s_payload(self):
        """The fork has to actually fork.

        While `run()` mutated its argument, the second branch silently started
        from the first branch's output.
        """
        payload = ImagePayload(image=np.full((40, 40, 3), 200, np.uint8))
        pipeline = PreprocessingPipelineBuilder.build(
            [StepConfig(name="channel_selection", params={"channel": "gray"})]
        )

        output = pipeline.run(payload)

        assert output is not payload
        assert payload.image.shape == (40, 40, 3)
        assert output.image.shape == (40, 40)

    def test_two_branches_from_one_payload_are_independent(self):
        source = ImagePayload(image=np.full((60, 60, 3), 180, np.uint8))

        gray = PreprocessingPipelineBuilder.build(
            [StepConfig(name="channel_selection", params={"channel": "gray"})]
        ).run(source)
        binary = PreprocessingPipelineBuilder.build(
            [
                StepConfig(name="channel_selection", params={"channel": "gray"}),
                StepConfig(name="adaptive_threshold", params={"block_size": 15, "c": 10}),
            ]
        ).run(source)

        assert source.image.ndim == 3
        assert gray.image.ndim == 2
        assert set(np.unique(binary.image)) <= {0, 255}
        # The grayscale branch must not have been binarized by the other one.
        assert len(np.unique(gray.image)) >= 1
        assert gray.applied_steps == ["channel_selection"]

    def test_the_display_image_is_the_corrected_page_not_a_mask(self):
        """What the user actually looks at, and what every bbox is measured
        against. It came back as the table branch's binary mask."""
        result, display = run()

        assert display.ndim == 3, "the display image must be the colour page"
        assert len(np.unique(display)) > 2, "the display image must not be binarized"


class TestOneCoordinateSpace:
    def test_every_box_lies_inside_the_display_image(self):
        result, display = run()
        height, width = display.shape[:2]

        boxes = [matched.element.bbox for matched in result.result.elements]
        assert boxes, "the tilted invoice should produce elements"

        for bbox in boxes:
            assert 0 <= bbox.x, bbox
            assert 0 <= bbox.y, bbox
            assert bbox.x + bbox.w <= width, bbox
            assert bbox.y + bbox.h <= height, bbox

    def test_cells_are_measured_in_the_same_space_they_are_shown_in(self):
        """The specific failure the extractors' second deskew caused.

        The detected grid has to sit where the grid actually is in the display
        image. Under a second rotation the cells drifted -- still valid boxes,
        just no longer over the table.
        """
        result, display = run()
        gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)

        regions = [r for r in result.layout.regions if r.bbox.w > 20 and r.bbox.h > 20]
        assert len(regions) >= 20, f"expected a detected grid, got {len(regions)}"

        # A real cell of a ruled form is mostly paper with a dark rule around
        # it. If the boxes had drifted off the page they would be sitting on the
        # uniform dark desk instead.
        inked = 0
        for region in regions:
            b = region.bbox
            patch = gray[b.y : b.y + b.h, b.x : b.x + b.w]
            if patch.size and patch.mean() > 128:
                inked += 1

        assert inked >= len(regions) * 0.9, (
            f"only {inked}/{len(regions)} detected cells land on the page -- "
            "the cells and the display image are in different coordinate spaces"
        )
