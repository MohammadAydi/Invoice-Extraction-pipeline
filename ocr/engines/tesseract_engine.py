from __future__ import annotations

from core.domain.image_payload import ImagePayload
from core.domain.ocr import OCRResult
from ocr.registry import engine_registry


@engine_registry.register("tesseract")
class TesseractEngine:
    """Adapter over pytesseract. Isolates its output format (dict of
    parallel lists) behind our own OCRResult/OCRFragment types.
    """

    def __init__(self, **engine_params):
        self.engine_params = engine_params  # e.g. {"lang": "ara"}

    def recognize(self, image: ImagePayload) -> OCRResult:
        raise NotImplementedError("Wrap pytesseract.image_to_data() here; return an OCRResult.")
