from __future__ import annotations

import ocr.engines  # noqa: F401  (triggers engine self-registration)
from config.schema import OCRConfig
from core.interfaces.ocr_engine import OCREngine
from ocr.registry import engine_registry


def build_ocr_engine(config: OCRConfig) -> OCREngine:
    return engine_registry.create(config.engine, **config.engine_params)
