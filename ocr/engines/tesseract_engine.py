"""Adapter over pytesseract.

Isolates Tesseract's output format -- a dict of parallel lists, one entry per
detected token -- behind the project's own OCRResult/OCRFragment types, so
nothing downstream knows which engine produced a fragment.

This is the light default: it needs the Tesseract binary and its `ara`/`eng`
traineddata but no model weights, no GPU and no second virtual environment,
which is what makes it the engine the service can start with out of the box.
The Surya and Qwen engines in this package read the same invoices far better on
handwriting and are selected from the desktop app's settings when they are
installed.
"""

from __future__ import annotations

import numpy as np

from core.domain.geometry import BoundingBox
from core.domain.image_payload import ImagePayload
from core.domain.ocr import OCRFragment, OCRResult
from ocr.registry import engine_registry

# Tesseract reports -1 for entries that carry no recognized text (page, block
# and paragraph rows in the same table as the words).
_NO_CONFIDENCE = -1.0


@engine_registry.register("tesseract")
class TesseractEngine:
    """Adapter over pytesseract.

    `lang` is a Tesseract language spec, so several languages are joined with
    '+' (``ara+eng``). It is read once at construction: switching languages
    means a new engine, which is what the config-driven factory already does.
    """

    def __init__(self, lang: str = "ara", tesseract_cmd: str = "", **engine_params):
        self.lang = lang
        self.tesseract_cmd = tesseract_cmd
        self.engine_params = engine_params

        # Imported here rather than at module scope so that merely importing
        # this module -- which `ocr.engines` does at startup to populate the
        # registry -- does not require pytesseract to be installed.
        import pytesseract

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        self._pytesseract = pytesseract

    def is_ready(self) -> bool:
        """True when the Tesseract binary answers, which is what readiness means here.

        There are no weights to load, so the only thing that can be missing is
        the executable itself -- and that is worth reporting through /health as
        "not ready" rather than failing the first extraction.
        """
        try:
            self._pytesseract.get_tesseract_version()
            return True
        except Exception:  # noqa: BLE001 - any failure means "not usable"
            return False

    def warmup(self) -> None:
        """Verify the binary is reachable, and fail loudly at startup if not."""
        self._pytesseract.get_tesseract_version()

    def recognize(self, image: ImagePayload) -> OCRResult:
        raw = self._to_rgb(image.image)

        data = self._pytesseract.image_to_data(
            raw,
            lang=self.lang,
            output_type=self._pytesseract.Output.DICT,
        )

        fragments: list[OCRFragment] = []

        for index, text in enumerate(data["text"]):
            token = (text or "").strip()
            if not token:
                continue

            confidence = float(data["conf"][index])
            if confidence == _NO_CONFIDENCE:
                continue

            fragments.append(
                OCRFragment(
                    text=token,
                    bbox=BoundingBox(
                        x=int(data["left"][index]),
                        y=int(data["top"][index]),
                        w=int(data["width"][index]),
                        h=int(data["height"][index]),
                    ),
                    # Tesseract reports 0-100; the rest of the pipeline works
                    # in 0-1 like every other engine here.
                    confidence=confidence / 100.0,
                )
            )

        return OCRResult(
            fragments=fragments,
            engine_name="tesseract",
            raw_engine_output=data,
        )

    @staticmethod
    def _to_rgb(array: np.ndarray) -> np.ndarray:
        """Tesseract wants 3-channel input; the OCR branch usually ends binary."""
        if array is None or array.size == 0:
            raise ValueError("tesseract received an empty image.")
        if array.ndim == 2:
            return np.stack([array] * 3, axis=-1)
        return array
