from __future__ import annotations

from core.interfaces.output_formatter import OutputFormatter
from core.registry import Registry

formatter_registry: Registry[OutputFormatter] = Registry(kind="output formatter")
