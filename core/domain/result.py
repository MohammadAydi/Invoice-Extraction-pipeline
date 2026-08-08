from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from core.domain.matching import MatchedElement


@dataclass
class PipelineResult:
    """The canonical, pre-formatting output of a full pipeline run.

    `display_image_path` points at the ONE geometrically corrected image
    shown on both sides of the UI. `config_snapshot` captures the exact
    AppConfig used for this run, for audit/reproducibility (per project
    decision to persist results).
    """

    invoice_id: str
    display_image_path: str
    elements: list[MatchedElement] = field(default_factory=list)
    config_snapshot: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    pipeline_version: str = "0.1.0"
