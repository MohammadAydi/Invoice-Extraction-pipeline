"""Importing this package self-registers every OCR engine that can be imported.

`surya_qwen` used to live here. It did detection, box refinement, cropping,
recognition and debug-artifact writing in one class, which is why none of the
five could be swapped independently. It is now the `detector_driven` flow
composed from `ocr/detectors/`, `ocr/refiners/`, `ocr/cropping/` and
`ocr/recognizers/`.

The heavy engines pull in optional dependencies -- surya-ocr, torch,
transformers -- that a machine running only the light Tesseract path has no
reason to install. Importing them unconditionally would make the whole service
fail to start on such a machine, so each is imported defensively: a missing
dependency removes that engine from the registry and nothing else. Selecting an
engine that failed to import then raises a clear "Unknown OCR engine 'x'.
Available: ..." from the registry, which is a far more useful failure than an
ImportError at startup.
"""

from __future__ import annotations

import logging

from ocr.engines import stub_engine  # noqa: F401  (no dependencies at all)
from ocr.engines import tesseract_engine  # noqa: F401  (pytesseract imported lazily)

logger = logging.getLogger(__name__)

_OPTIONAL_ENGINES = (
    ("easyocr_engine", "easyocr"),
    ("surya_engine", "surya-ocr"),
    ("qwen_vlm_engine", "transformers / torch"),
)

for _module_name, _requirement in _OPTIONAL_ENGINES:
    try:
        __import__(f"ocr.engines.{_module_name}")
    except Exception as _exc:  # noqa: BLE001 - any import failure disables the engine
        logger.info(
            "OCR engine '%s' unavailable (needs %s): %s",
            _module_name,
            _requirement,
            _exc,
        )
