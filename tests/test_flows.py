"""The three reading strategies, and the Template Method that keeps them honest.

The point of these tests is the *shape* of each flow, not recognition quality:
that `run()` calls its hooks in one fixed order for all three, that the pieces
are genuinely swappable, and above all that `layout_driven` never touches a text
detector -- which is the guarantee the whole flow exists to provide.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.domain.geometry import BoundingBox
from core.domain.image_payload import ImagePayload
from core.domain.layout import InvoiceLayout, LayoutRegion
from core.domain.ocr import DetectionResult, OCRFragment, OCRResult, TextRegion
from core.domain.roles import CellRole, ContentKind, Zone
from core.domain.table import TableCell, TableExtractionResult, TableStructure
from mapping.cell_mapper import CellMapper
from ocr.cropping.padded_cropper import PaddedRegionCropper
from ocr.refiners.noop_refiner import NoopRegionRefiner
from orchestration.flows.detector_driven import DetectorDrivenFlow
from orchestration.flows.layout_driven import LayoutDrivenFlow
from orchestration.flows.registry import flow_registry
from orchestration.flows.single_engine import SingleEngineFlow

TABLE_ID = "t1"


def page(w: int = 400, h: int = 300) -> ImagePayload:
    return ImagePayload(image=np.full((h, w, 3), 255, np.uint8))


def region(id: str, x: int, y: int, role: CellRole = CellRole.UNKNOWN) -> LayoutRegion:
    return LayoutRegion.from_role(
        id=id,
        bbox=BoundingBox(x=x, y=y, w=80, h=40),
        role=role,
        zone=Zone.TABLE,
        row=0,
        col=0,
        table_id=TABLE_ID,
    )


class FakeExtractor:
    """Records that it ran, so the template's order can be asserted."""

    name = "fake_extractor"

    def __init__(self, calls: list[str], cells: list[TableCell] | None = None):
        self.calls = calls
        self.cells = cells or []

    def extract(self, payload):
        self.calls.append("extract")
        if not self.cells:
            return TableExtractionResult(tables=[], extractor_name=self.name)
        return TableExtractionResult(
            tables=[
                TableStructure(
                    table_id=TABLE_ID,
                    bbox=BoundingBox(x=0, y=0, w=400, h=300),
                    cells=self.cells,
                )
            ],
            extractor_name=self.name,
        )


class FakeClassifier:
    name = "fake_classifier"

    def __init__(self, calls: list[str], regions: list[LayoutRegion] | None = None):
        self.calls = calls
        self.regions = regions or []

    def classify(self, table, image_size):
        self.calls.append("classify")
        return InvoiceLayout(regions=self.regions, source="fake", table_id=TABLE_ID)


class ExplodingDetector:
    """A detector that fails the test if anything calls it.

    This is the whole assertion behind "layout_driven does not use the
    detector" -- stated as an object that cannot be used quietly.
    """

    name = "exploding_detector"

    def detect(self, payload):
        raise AssertionError("the detector must not be called by this flow")


class FakeDetector:
    name = "fake_detector"

    def __init__(self, calls: list[str], regions: list[TextRegion]):
        self.calls = calls
        self.regions = regions

    def detect(self, payload):
        self.calls.append("detect")
        return DetectionResult(regions=self.regions, detector_name=self.name)


class FakeRecognizer:
    name = "fake_recognizer"

    def __init__(self, calls: list[str]):
        self.calls = calls
        self.seen: list[ContentKind] = []

    def read(self, crop):
        return "x"

    def read_all(self, crops):
        self.calls.append("read")
        self.seen.extend(c.content_kind for c in crops)
        return [f"text{i}" for i in range(len(crops))]


class FakeEngine:
    name = "fake_engine"

    def __init__(self, calls: list[str]):
        self.calls = calls

    def recognize(self, payload):
        self.calls.append("recognize")
        return OCRResult(
            fragments=[
                OCRFragment(text="whole page", bbox=BoundingBox(x=5, y=5, w=50, h=20))
            ],
            engine_name=self.name,
        )


def cropper() -> PaddedRegionCropper:
    return PaddedRegionCropper(pad=2, upscale=1, min_side=4)


class TestTemplateMethod:
    """One sequence, three fillings."""

    def test_every_flow_analyzes_layout_before_reading(self):
        for build in (
            lambda calls: SingleEngineFlow(
                FakeExtractor(calls), FakeClassifier(calls), CellMapper(), FakeEngine(calls)
            ),
            lambda calls: LayoutDrivenFlow(
                FakeExtractor(calls),
                FakeClassifier(calls),
                CellMapper(),
                cropper(),
                FakeRecognizer(calls),
            ),
            lambda calls: DetectorDrivenFlow(
                FakeExtractor(calls),
                FakeClassifier(calls),
                CellMapper(),
                FakeDetector(calls, []),
                NoopRegionRefiner(),
                cropper(),
                FakeRecognizer(calls),
            ),
        ):
            calls: list[str] = []
            build(calls).run(page(), page())

            assert calls[:2] == ["extract", "classify"], calls

    def test_all_three_flows_are_registered(self):
        assert set(flow_registry.available()) == {
            "single_engine",
            "detector_driven",
            "layout_driven",
        }

    def test_a_failing_extractor_degrades_instead_of_crashing(self):
        """An unruled receipt has no grid and is still a good invoice."""

        class Broken:
            name = "broken"

            def extract(self, payload):
                raise RuntimeError("no grid here")

        calls: list[str] = []
        flow = SingleEngineFlow(Broken(), FakeClassifier(calls), CellMapper(), FakeEngine(calls))
        outcome = flow.run(page(), page())

        assert not outcome.layout
        assert outcome.ocr_result.fragments


class TestLayoutDriven:
    def test_never_constructs_or_calls_a_detector(self):
        assert "detector" not in LayoutDrivenFlow.REQUIRES

    def test_reads_one_region_per_classified_cell(self):
        calls: list[str] = []
        regions = [region("r1", 10, 10), region("r2", 120, 10)]
        recognizer = FakeRecognizer(calls)

        flow = LayoutDrivenFlow(
            FakeExtractor(calls),
            FakeClassifier(calls, regions),
            CellMapper(),
            cropper(),
            recognizer,
        )
        outcome = flow.run(page(), page())

        assert len(outcome.ocr_result.fragments) == 2

    def test_the_prompt_kind_comes_from_the_cell_role(self):
        """The reason this flow reads better: a quantity cell is read with the
        digits-only prompt because the layout said it is a quantity."""
        calls: list[str] = []
        recognizer = FakeRecognizer(calls)
        regions = [
            region("r1", 10, 10, CellRole.QUANTITY),
            region("r2", 120, 10, CellRole.PRODUCT_NAME),
            region("r3", 230, 10, CellRole.INVOICE_DATE),
        ]

        LayoutDrivenFlow(
            FakeExtractor(calls),
            FakeClassifier(calls, regions),
            CellMapper(),
            cropper(),
            recognizer,
        ).run(page(), page())

        assert set(recognizer.seen) == {
            ContentKind.NUMBER,
            ContentKind.ARABIC_TEXT,
            ContentKind.DATE,
        }

    def test_printed_captions_are_not_read(self):
        """A column header says the same thing on every invoice ever printed on
        this form. Reading it costs a model call per cell."""
        calls: list[str] = []
        regions = [
            region("r1", 10, 10, CellRole.COLUMN_HEADER),
            region("r2", 120, 10, CellRole.LABEL),
            region("r3", 230, 10, CellRole.QUANTITY),
        ]

        outcome = LayoutDrivenFlow(
            FakeExtractor(calls),
            FakeClassifier(calls, regions),
            CellMapper(),
            cropper(),
            FakeRecognizer(calls),
        ).run(page(), page())

        assert len(outcome.ocr_result.fragments) == 1

    def test_fragments_carry_the_region_they_came_from(self):
        """What lets mapping assign by identity instead of guessing containment."""
        calls: list[str] = []
        regions = [region("cell-a", 10, 10, CellRole.QUANTITY)]

        outcome = LayoutDrivenFlow(
            FakeExtractor(calls),
            FakeClassifier(calls, regions),
            CellMapper(),
            cropper(),
            FakeRecognizer(calls),
        ).run(page(), page())

        assert outcome.ocr_result.fragments[0].source_id == "cell-a"

    def test_no_grid_warns_rather_than_silently_returning_nothing(self):
        """It must not fall back to the detector: a silent switch to a different
        reading strategy is what makes a bad extraction impossible to diagnose."""
        calls: list[str] = []

        outcome = LayoutDrivenFlow(
            FakeExtractor(calls),
            FakeClassifier(calls, []),
            CellMapper(),
            cropper(),
            FakeRecognizer(calls),
        ).run(page(), page())

        assert not outcome.ocr_result.fragments
        assert outcome.warnings
        assert "layout_driven" in outcome.warnings[0]


class TestDetectorDriven:
    def test_reads_what_the_detector_found(self):
        calls: list[str] = []
        detected = [
            TextRegion(bbox=BoundingBox(x=10, y=10, w=80, h=40)),
            TextRegion(bbox=BoundingBox(x=200, y=10, w=80, h=40)),
        ]

        outcome = DetectorDrivenFlow(
            FakeExtractor(calls),
            FakeClassifier(calls),
            CellMapper(),
            FakeDetector(calls, detected),
            NoopRegionRefiner(),
            cropper(),
            FakeRecognizer(calls),
        ).run(page(), page())

        assert calls == ["extract", "classify", "detect", "read"]
        assert len(outcome.ocr_result.fragments) == 2

    def test_reads_text_outside_any_ruled_box(self):
        """Its advantage over layout_driven, stated as a test."""
        calls: list[str] = []
        detected = [TextRegion(bbox=BoundingBox(x=300, y=250, w=60, h=30))]

        outcome = DetectorDrivenFlow(
            FakeExtractor(calls),
            FakeClassifier(calls, []),  # no grid at all
            CellMapper(),
            FakeDetector(calls, detected),
            NoopRegionRefiner(),
            cropper(),
            FakeRecognizer(calls),
        ).run(page(), page())

        assert len(outcome.ocr_result.fragments) == 1


class TestSingleEngine:
    def test_hands_the_whole_page_to_the_engine(self):
        calls: list[str] = []

        outcome = SingleEngineFlow(
            FakeExtractor(calls), FakeClassifier(calls), CellMapper(), FakeEngine(calls)
        ).run(page(), page())

        assert calls == ["extract", "classify", "recognize"]
        assert outcome.ocr_result.fragments[0].text == "whole page"


class TestRecognizerContract:
    def test_a_recognizer_that_drops_a_result_is_an_error(self):
        """Callers index texts against crops; a short list would silently pair
        every value after the gap with the wrong box."""

        class Dropping:
            name = "dropping"

            def read(self, crop):
                return "x"

            def read_all(self, crops):
                return ["only one"]

        calls: list[str] = []
        regions = [region("r1", 10, 10), region("r2", 120, 10)]

        flow = LayoutDrivenFlow(
            FakeExtractor(calls),
            FakeClassifier(calls, regions),
            CellMapper(),
            cropper(),
            Dropping(),
        )

        with pytest.raises(RuntimeError, match="read_all"):
            flow.run(page(), page())
