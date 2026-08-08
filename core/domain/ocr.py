from __future__ import annotations

from dataclasses import dataclass

from core.domain.geometry import BoundingBox


@dataclass(frozen=True)
class OCRFragment:
    text: str
    bbox: BoundingBox
    confidence: float


@dataclass(frozen=True)
class OCRResult:
    fragments: list[OCRFragment]
    engine_name: str
    raw_engine_output: object = None  # escape hatch for adapter-specific debug data
