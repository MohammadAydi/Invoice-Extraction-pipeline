"""The expensive half of a flow, built separately so it can be cached.

A request may change a preprocessing parameter, which means a new orchestrator
and therefore a new flow -- but rebuilding the recognizer along with it would
reload model weights (about a minute for the VLM path) for a change that has
nothing to do with reading text. So the pieces that own weights or a subprocess
are built here, keyed on the `ocr` config section alone, and handed to whichever
flow is assembled around them.

**Only what the selected flow declares is built.** Each flow class names its
requirements in `REQUIRES`, and this module builds exactly those -- so adding a
fourth flow never edits this file. That is also the mechanism behind
"`layout_driven` does not use the detector": there is no detector object at all,
so it cannot be called by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from config.schema import AppConfig
from ocr.factory import (
    build_ocr_engine,
    build_region_cropper,
    build_region_refiner,
    build_text_detector,
    build_text_recognizer,
)
from orchestration.flows.registry import flow_registry


@dataclass(frozen=True)
class OCRComponents:
    """Whichever pieces the selected flow declared; the rest stay None."""

    engine: object | None = None
    detector: object | None = None
    refiner: object | None = None
    cropper: object | None = None
    recognizer: object | None = None


# One builder per field of OCRComponents. A flow names fields, never builders.
_BUILDERS = {
    "engine": lambda ocr, flow: build_ocr_engine(ocr),
    "detector": build_text_detector,
    "refiner": build_region_refiner,
    "cropper": build_region_cropper,
    "recognizer": build_text_recognizer,
}

_COMPONENT_FIELDS = frozenset(f.name for f in fields(OCRComponents))


def build_ocr_components(config: AppConfig) -> OCRComponents:
    flow_name = config.flow.name
    flow_cls = flow_registry.get(flow_name)

    required = getattr(flow_cls, "REQUIRES", ())
    unknown = set(required) - _COMPONENT_FIELDS
    if unknown:
        raise ValueError(
            f"Flow '{flow_name}' declares unknown component(s) {sorted(unknown)}. "
            f"Valid: {sorted(_COMPONENT_FIELDS)}."
        )

    return OCRComponents(
        **{name: _BUILDERS[name](config.ocr, flow_name) for name in required}
    )
