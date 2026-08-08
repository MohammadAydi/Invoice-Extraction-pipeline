from __future__ import annotations

import table_extraction.extractors  # noqa: F401  (triggers extractor self-registration)
from config.schema import TableExtractionConfig
from core.interfaces.table_extractor import TableExtractor
from table_extraction.registry import extractor_registry


def build_table_extractor(config: TableExtractionConfig) -> TableExtractor:
    return extractor_registry.create(config.extractor, **config.extractor_params)
