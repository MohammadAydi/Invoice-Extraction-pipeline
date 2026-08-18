from __future__ import annotations

from typing import Protocol

from core.domain.image_payload import ImagePayload
from core.domain.ocr import DetectionResult


class TextDetector(Protocol):
    """Finds where text is, and says nothing about what it says.

    Split out of the OCR engine so the detector and the recognizer can be
    chosen independently -- which is the entire point of running two models.
    Before this split, changing the recognizer meant editing the class that
    owned the detector.
    """

    def detect(self, image: ImagePayload) -> DetectionResult: ...
