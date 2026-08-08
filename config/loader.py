from __future__ import annotations

from pathlib import Path

import yaml

from config.schema import AppConfig


def load_config(path: str | Path) -> AppConfig:
    """Load and validate an AppConfig from a YAML file.

    Validation failures raise pydantic.ValidationError with a precise,
    field-level message -- config problems fail fast at startup instead of
    deep inside the pipeline.
    """
    raw = yaml.safe_load(Path(path).read_text())
    return AppConfig(**raw)
