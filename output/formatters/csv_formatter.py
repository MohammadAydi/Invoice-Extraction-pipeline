"""Stub. Alternative output representation (e.g. flat row-per-cell CSV for
export/reporting) -- demonstrates that the final format is swappable via
`output.formatter` in config without touching the orchestrator.
"""

from __future__ import annotations

from core.domain.result import PipelineResult
from output.registry import formatter_registry


@formatter_registry.register("csv")
class CSVFormatter:
    def __init__(self, **params):
        self.params = params

    def format(self, result: PipelineResult) -> str:
        raise NotImplementedError("Serialize PipelineResult into CSV rows.")
