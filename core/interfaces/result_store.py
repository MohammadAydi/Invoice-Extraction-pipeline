from __future__ import annotations

from typing import Protocol

from core.domain.result import PipelineResult


class ResultStore(Protocol):
    """Persists a PipelineResult (including its config snapshot) for
    audit/reproducibility, and allows re-loading a past run.
    """

    def save(self, result: PipelineResult) -> None: ...

    def load(self, invoice_id: str) -> PipelineResult: ...
