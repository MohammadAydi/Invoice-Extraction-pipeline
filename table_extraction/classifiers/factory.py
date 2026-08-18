from __future__ import annotations

import table_extraction.classifiers  # noqa: F401  (triggers self-registration)
from config.schema import TableExtractionConfig
from core.interfaces.layout_classifier import LayoutClassifier
from table_extraction.classifiers.registry import classifier_registry


def build_layout_classifier(config: TableExtractionConfig) -> LayoutClassifier:
    return classifier_registry.create(config.classifier, **config.classifier_params)
