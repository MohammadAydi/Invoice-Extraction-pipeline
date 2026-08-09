# ocr/engines/surya_engine.py
from __future__ import annotations

import numpy as np
from PIL import Image

from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.foundation import FoundationPredictor

from core.domain.geometry import BoundingBox
from core.domain.image_payload import ImagePayload
from core.domain.ocr import OCRFragment, OCRResult
from ocr.registry import engine_registry


@engine_registry.register("surya")
class SuryaEngine:
    def __init__(self, langs: list[str] = None, **engine_params):
        self.langs = langs or ["ar"]
        self.det_predictor = DetectionPredictor()
        foundation = FoundationPredictor()
        self.rec_predictor = RecognitionPredictor(foundation)

    def recognize(self, image: ImagePayload) -> OCRResult:
        arr = image.image
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        pil_img = Image.fromarray(arr).convert("RGB")

        results = self.rec_predictor(
            images=[pil_img],
            det_predictor=self.det_predictor,
        )

        fragments = []
        for line in results[0].text_lines:
            xs = [p[0] for p in line.polygon]
            ys = [p[1] for p in line.polygon]
            x1, y1 = min(xs), min(ys)
            fragments.append(OCRFragment(
                text=line.text,
                bbox=BoundingBox(x=x1, y=y1, w=max(xs) - x1, h=max(ys) - y1),
                confidence=line.confidence,
            ))

        return OCRResult(fragments=fragments, engine_name="surya", raw_engine_output=results[0])