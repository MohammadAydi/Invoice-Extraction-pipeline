"""Surya's text detector, on its own.

Extracted from `HybridSuryaQwenEngine`, where it was fused with cropping and
Qwen recognition in one class. Surya's *recognizer* produced Chinese glyphs and
LaTeX on this project's handwritten Arabic; its *detector* placed boxes
correctly. Keeping only the detector is the whole point, and it can only be
"kept" separately if it is separate.

Known limit, stated plainly because it changes how results should be read: the
detector misses short isolated digits. On the sample invoice, quantities of one
or two characters were never boxed, so they never reached the recognizer at all.
That is the specific weakness the layout-driven flow does not have -- a printed
cell exists whether or not anything was detected inside it.
"""

from __future__ import annotations

import time

from surya.detection import DetectionPredictor

from core.domain.geometry import BoundingBox
from core.domain.image_payload import ImagePayload
from core.domain.ocr import DetectionResult, TextRegion
from core.domain.roles import ContentKind
from ocr.detectors.registry import detector_registry
from ocr.engines.qwen_vlm_engine import to_pil


@detector_registry.register("surya_detector")
class SuryaTextDetector:
    name = "surya_detector"

    def __init__(
        self,
        min_side: int = 12,
        ignore_beyond_x: float | None = None,
        **detector_params,
    ):
        # Ignore specks: a box a few pixels on a side is dust or a printing
        # artefact, and reading it costs a full model call.
        self.min_side = min_side

        # A fraction of the page width past which detections are dropped, for
        # forms with a margin the recognizer should never be pointed at.
        self.ignore_beyond_x = ignore_beyond_x

        self.detector_params = detector_params
        self._predictor: DetectionPredictor | None = None

    @property
    def predictor(self) -> DetectionPredictor:
        """Built on first use, not in __init__.

        Constructing it loads weights, and the pipeline pool builds an
        orchestrator to answer questions like "what config would you run?"
        without ever calling it.
        """
        if self._predictor is None:
            self._predictor = DetectionPredictor()
        return self._predictor

    def is_ready(self) -> bool:
        return self._predictor is not None

    def warmup(self) -> None:
        _ = self.predictor

    def detect(self, image: ImagePayload) -> DetectionResult:
        pil = to_pil(image.image)
        w, _h = pil.size

        started = time.perf_counter()
        det = self.predictor([pil])[0]
        elapsed = time.perf_counter() - started

        regions: list[TextRegion] = []
        for box in det.bboxes:
            x1, y1, x2, y2 = (int(v) for v in box.bbox)
            if x2 - x1 < self.min_side or y2 - y1 < self.min_side:
                continue
            if (
                self.ignore_beyond_x is not None
                and (x1 + x2) / 2 > self.ignore_beyond_x * w
            ):
                continue

            regions.append(
                TextRegion(
                    bbox=BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1),
                    # Detection confidence, not recognition confidence. The two
                    # are not interchangeable and conflating them would report a
                    # confidently-located box as a confidently-read value.
                    confidence=getattr(box, "confidence", None),
                    content_kind=ContentKind.ANY,
                )
            )

        return DetectionResult(
            regions=regions,
            detector_name=self.name,
            raw_detector_output={"det": det, "detect_s": round(elapsed, 3)},
        )
