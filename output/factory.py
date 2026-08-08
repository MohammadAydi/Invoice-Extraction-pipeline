from __future__ import annotations

import output.formatters  # noqa: F401  (triggers formatter self-registration)
from config.schema import OutputConfig
from core.interfaces.output_formatter import OutputFormatter
from output.registry import formatter_registry


def build_output_formatter(config: OutputConfig) -> OutputFormatter:
    return formatter_registry.create(config.formatter, **config.formatter_params)
