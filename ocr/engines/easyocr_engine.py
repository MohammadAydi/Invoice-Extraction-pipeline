from __future__ import annotations

from core.domain.image_payload import ImagePayload
from core.domain.ocr import OCRResult
from ocr.registry import engine_registry


@engine_registry.register("easyocr")
class EasyOCREngine:
    """Adapter over easyocr.Reader -- demonstrates that switching OCR
    engines at runtime only requires changing `ocr.engine` in config.
    """

    def __init__(self, **engine_params):
        self.engine_params = engine_params

    def recognize(self, image: ImagePayload) -> OCRResult:
        raise NotImplementedError("Wrap easyocr.Reader.readtext() here; return an OCRResult.")
