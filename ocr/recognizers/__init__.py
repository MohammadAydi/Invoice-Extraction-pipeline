"""Importing this package self-registers every text recognizer that can be imported.

`echo` needs nothing and is always available -- it is what lets the flow
machinery be tested without a model. The rest are imported defensively, for the
same reason the heavy OCR engines are: a machine on the light path has no reason
to have torch installed, and an unconditional import would stop the service
starting there.
"""

from __future__ import annotations

import logging

from ocr.recognizers import echo_recognizer  # noqa: F401  (no dependencies)

logger = logging.getLogger(__name__)

_OPTIONAL_RECOGNIZERS = (("qwen_recognizer", "transformers / torch"),)

for _module_name, _requirement in _OPTIONAL_RECOGNIZERS:
    try:
        __import__(f"ocr.recognizers.{_module_name}")
    except Exception as _exc:  # noqa: BLE001 - any import failure disables it
        logger.info(
            "Text recognizer '%s' unavailable (needs %s): %s",
            _module_name,
            _requirement,
            _exc,
        )
