"""Detector finds the boxes, recognizer reads them.

This is what `HybridSuryaQwenEngine` did, unchanged in behaviour and split into
four collaborators the flow coordinates:

    detector.detect  ->  refiner.refine  ->  cropper.crop  ->  recognizer.read

Its strength is that it reads text wherever it appears, including outside any
printed box. Its two weaknesses are worth stating, because they are what the
layout-driven flow exists to avoid:

* The detector misses short isolated digits. A quantity of one or two characters
  is often never boxed, so it is never read -- and there is nothing in the
  output to say a value is missing rather than blank.
* Column boundaries have to be *declared*, as fractions of the page width, by
  someone who measured one printed form by hand.
"""

from __future__ import annotations

from core.domain.image_payload import ImagePayload
from core.domain.layout import InvoiceLayout
from core.domain.ocr import OCRResult
from orchestration.flows.base import ExtractionFlow
from orchestration.flows.registry import flow_registry


@flow_registry.register("detector_driven")
class DetectorDrivenFlow(ExtractionFlow):
    name = "detector_driven"
    REQUIRES = ("detector", "refiner", "cropper", "recognizer")

    @classmethod
    def from_config(cls, config, components, table_extractor, layout_classifier, cell_mapper):
        return cls(
            table_extractor,
            layout_classifier,
            cell_mapper,
            detector=components.detector,
            refiner=components.refiner,
            cropper=components.cropper,
            recognizer=components.recognizer,
            **config.flow.params,
        )

    def __init__(
        self,
        table_extractor,
        layout_classifier,
        cell_mapper,
        detector,
        refiner,
        cropper,
        recognizer,
    ):
        super().__init__(table_extractor, layout_classifier, cell_mapper)
        self.detector = detector
        self.refiner = refiner
        self.cropper = cropper
        self.recognizer = recognizer

    def _read_text(
        self, payload: ImagePayload, layout: InvoiceLayout, debug=None
    ) -> OCRResult:
        height, width = payload.image.shape[:2]

        detection = self.detector.detect(payload)
        regions = self.refiner.refine(detection.regions, width, height)

        return self._crop_and_read(
            payload,
            regions,
            engine_name=f"{self.detector.name}+{self.recognizer.name}",
            raw_output={
                "detector": detection.raw_detector_output,
                "detected": len(detection.regions),
                "after_refine": len(regions),
            },
            debug=debug
        )

    def _warnings(self, layout: InvoiceLayout, ocr_result: OCRResult) -> list[str]:
        if not ocr_result.fragments:
            return ["The text detector found nothing on this page."]
        return []

    def components(self) -> list[object]:
        return super().components() + [
            self.detector,
            self.refiner,
            self.cropper,
            self.recognizer,
        ]

    @property
    def engine_name(self) -> str:
        return f"{self.detector.name}+{self.recognizer.name}"