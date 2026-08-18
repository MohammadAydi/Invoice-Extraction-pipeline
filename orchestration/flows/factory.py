from __future__ import annotations

import orchestration.flows  # noqa: F401  (triggers flow self-registration)
from config.schema import AppConfig
from orchestration.flows.base import ExtractionFlow
from orchestration.flows.components import OCRComponents, build_ocr_components
from orchestration.flows.registry import flow_registry
from table_extraction.classifiers.factory import build_layout_classifier
from table_extraction.factory import build_table_extractor


def build_flow(
    config: AppConfig,
    cell_mapper,
    components: OCRComponents | None = None,
) -> ExtractionFlow:
    """Assemble the configured flow.

    `components` lets the HTTP layer pass in an already-built detector and
    recognizer, which is what keeps a preprocessing tweak from reloading model
    weights. Each flow assembles itself from them via its own `from_config`, so
    adding a flow never touches this function.
    """
    flow_cls = flow_registry.get(config.flow.name)

    return flow_cls.from_config(
        config,
        components if components is not None else build_ocr_components(config),
        build_table_extractor(config.table_extraction),
        build_layout_classifier(config.table_extraction),
        cell_mapper,
    )
