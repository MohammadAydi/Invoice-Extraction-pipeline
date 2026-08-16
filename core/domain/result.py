from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from core.domain.matching import MatchedElement

if TYPE_CHECKING:
    from invoice.models import InvoiceDraft


@dataclass
class PipelineResult:
    """The canonical, pre-formatting output of a full pipeline run.

    `display_image_path` points at the ONE geometrically corrected image
    shown on both sides of the UI. `config_snapshot` captures the exact
    AppConfig used for this run, for audit/reproducibility (per project
    decision to persist results).

    `invoice` is the same run read as an invoice -- header fields and line
    items rather than loose elements. It is what the desktop app actually
    binds to; `elements` remains the lossless view, kept because the overlay
    needs a box for every piece of text on the page, including the ones no
    invoice field claimed.
    """

    invoice_id: str
    display_image_path: str
    elements: list[MatchedElement] = field(default_factory=list)
    invoice: "InvoiceDraft | None" = None
    raw_text: str = ""
    ocr_engine: str = ""
    config_snapshot: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    pipeline_version: str = "1.0.0"
