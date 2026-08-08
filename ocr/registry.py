from __future__ import annotations

from core.interfaces.ocr_engine import OCREngine
from core.registry import Registry

engine_registry: Registry[OCREngine] = Registry(kind="OCR engine")
