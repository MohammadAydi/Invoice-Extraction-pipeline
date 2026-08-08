"""Simplest possible ResultStore: one JSON file per invoice_id. Swappable
for a DB-backed implementation later without the orchestrator changing
(same ResultStore interface either way).
"""

from __future__ import annotations

from pathlib import Path

from core.domain.result import PipelineResult


class FileResultStore:
    def __init__(self, output_dir: str = "results/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, result: PipelineResult) -> None:
        raise NotImplementedError("Serialize PipelineResult (incl. config_snapshot) to JSON.")

    def load(self, invoice_id: str) -> PipelineResult:
        raise NotImplementedError("Deserialize a previously saved PipelineResult.")
