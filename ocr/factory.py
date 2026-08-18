from __future__ import annotations

import ocr.cropping  # noqa: F401  (triggers cropper self-registration)
import ocr.detectors  # noqa: F401  (triggers detector self-registration)
import ocr.engines  # noqa: F401  (triggers engine self-registration)
import ocr.recognizers  # noqa: F401  (triggers recognizer self-registration)
import ocr.refiners  # noqa: F401  (triggers refiner self-registration)
from config.schema import ComponentConfig, OCRConfig
from core.interfaces.ocr_engine import OCREngine
from core.interfaces.region_cropper import RegionCropper
from core.interfaces.region_refiner import RegionRefiner
from core.interfaces.text_detector import TextDetector
from core.interfaces.text_recognizer import TextRecognizer
from core.registry import Registry
from ocr.cropping.registry import cropper_registry
from ocr.detectors.registry import detector_registry
from ocr.recognizers.registry import recognizer_registry
from ocr.refiners.registry import refiner_registry
from ocr.registry import engine_registry


def _build(
    registry: Registry,
    component: ComponentConfig | None,
    *,
    required_by: str,
    field: str,
    default: ComponentConfig | None = None,
):
    component = component or default
    if component is None:
        raise ValueError(
            f"The '{required_by}' flow needs `ocr.{field}` in the configuration. "
            f"Available: {', '.join(registry.available()) or '(none registered)'}."
        )
    return registry.create(component.name, **component.params)


def build_ocr_engine(config: OCRConfig) -> OCREngine:
    if not config.engine:
        raise ValueError(
            "The 'single_engine' flow needs `ocr.engine` in the configuration. "
            f"Available: {', '.join(engine_registry.available())}."
        )
    return engine_registry.create(config.engine, **config.engine_params)


def build_text_detector(config: OCRConfig, flow: str = "detector_driven") -> TextDetector:
    return _build(detector_registry, config.detector, required_by=flow, field="detector")


def build_region_cropper(config: OCRConfig, flow: str) -> RegionCropper:
    # A cropper has no expensive state and one sensible default, so a
    # configuration that does not care need not say anything.
    return _build(
        cropper_registry,
        config.cropper,
        required_by=flow,
        field="cropper",
        default=ComponentConfig(name="padded_crop"),
    )


def build_text_recognizer(config: OCRConfig, flow: str) -> TextRecognizer:
    return _build(
        recognizer_registry, config.recognizer, required_by=flow, field="recognizer"
    )


def build_region_refiner(config: OCRConfig, flow: str = "detector_driven") -> RegionRefiner:
    return _build(
        refiner_registry,
        config.refiner,
        required_by=flow,
        field="refiner",
        default=ComponentConfig(name="noop"),
    )
