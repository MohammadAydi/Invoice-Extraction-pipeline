from __future__ import annotations

from core.interfaces.table_extractor import TableExtractor
from core.registry import Registry

extractor_registry: Registry[TableExtractor] = Registry(kind="table extractor")
