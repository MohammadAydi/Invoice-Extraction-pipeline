"""Importing this package self-registers every text detector that can be imported.

Surya pulls in torch. A machine running only the light path has no reason to
install it, and an unconditional import would stop the whole service from
starting there -- so the import is defensive, exactly as it is for the heavy OCR
engines. Selecting a detector that failed to import then raises a clear
"Unknown text detector 'x'. Available: ..." from the registry, which is a far
more useful failure than an ImportError at startup.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_OPTIONAL_DETECTORS = (("surya_detector", "surya-ocr"),)

for _module_name, _requirement in _OPTIONAL_DETECTORS:
    try:
        __import__(f"ocr.detectors.{_module_name}")
    except Exception as _exc:  # noqa: BLE001 - any import failure disables it
        logger.info(
            "Text detector '%s' unavailable (needs %s): %s",
            _module_name,
            _requirement,
            _exc,
        )
