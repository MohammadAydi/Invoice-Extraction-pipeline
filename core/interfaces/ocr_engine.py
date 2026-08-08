from __future__ import annotations

from typing import Protocol

from core.domain.image_payload import ImagePayload
from core.domain.ocr import OCRResult


class OCREngine(Protocol):
    def recognize(self, image: ImagePayload) -> OCRResult: ...
