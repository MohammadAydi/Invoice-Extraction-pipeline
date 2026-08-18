"""What one pipeline invocation produced.

Lives here rather than beside the orchestrator because it is a domain object
that crosses layers: the CLI prints it, the HTTP layer encodes its images, and
`run_batch` collects them. A caller should not have to import the composition
root to name the thing it hands back.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.domain.layout import InvoiceLayout
from core.domain.result import PipelineResult


@dataclass
class PipelineRun:
    """Everything one invocation produced.

    `output` is the formatter's view -- what goes on the wire. `result` is the
    canonical domain object behind it. The images are kept in memory rather
    than read back off disk so the HTTP layer can encode them without a round
    trip through the filesystem.
    """

    result: PipelineResult
    output: object
    display_image: np.ndarray
    enhanced_image: np.ndarray
    ocr_input_image: np.ndarray
    applied_steps: list[str] = field(default_factory=list)

    # The labelled page behind `result`. Carried so the HTTP layer and the CLI
    # can report which extractor/classifier pair actually ran -- a surprising
    # extraction is almost always a question about that pair.
    layout: InvoiceLayout = field(default_factory=InvoiceLayout)

    def __repr__(self) -> str:  # pragma: no cover - convenience for the CLI
        invoice = self.result.invoice
        items = len(invoice.line_items) if invoice else 0
        return (
            f"PipelineRun(invoice_id={self.result.invoice_id!r}, "
            f"elements={len(self.result.elements)}, line_items={items})"
        )
