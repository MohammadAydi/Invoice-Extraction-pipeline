"""Simplest possible ResultStore: one JSON file per invoice_id. Swappable
for a DB-backed implementation later without the orchestrator changing
(same ResultStore interface either way).
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path

from core.domain.document import FreeFieldElement, TableCellElement
from core.domain.geometry import BoundingBox
from core.domain.matching import MatchedElement, MatchResult
from core.domain.ocr import OCRFragment
from core.domain.result import PipelineResult


class FileResultStore:
    def __init__(self, output_dir: str = "results/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _result_path(self, invoice_id: str) -> Path:
        invoice_dir = self.output_dir / invoice_id
        invoice_dir.mkdir(parents=True, exist_ok=True)
        return invoice_dir / "result.json"

    def save(self, result: PipelineResult) -> None:
        data = dataclasses.asdict(result)
        data["created_at"] = result.created_at.isoformat()
        path = self._result_path(result.invoice_id)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, invoice_id: str) -> PipelineResult:
        path = self._result_path(invoice_id)
        if not path.exists():
            raise FileNotFoundError(f"No saved result for invoice_id '{invoice_id}' at {path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        elements = [self._matched_element_from_dict(raw) for raw in data["elements"]]

        # The invoice view is deliberately not rebuilt: it is derived from the
        # elements and the request's catalogs, and a reload has neither the
        # catalogs nor a reason to re-match against a catalog that has moved on
        # since. Callers that need it re-run the parse.
        return PipelineResult(
            invoice_id=data["invoice_id"],
            display_image_path=data["display_image_path"],
            elements=elements,
            raw_text=data.get("raw_text", ""),
            ocr_engine=data.get("ocr_engine", ""),
            config_snapshot=data["config_snapshot"],
            created_at=datetime.fromisoformat(data["created_at"]),
            pipeline_version=data["pipeline_version"],
        )

    @staticmethod
    def _matched_element_from_dict(raw: dict) -> MatchedElement:
        el = raw["element"]
        bbox = BoundingBox(**el["bbox"])
        fragments = [
            OCRFragment(text=f["text"], bbox=BoundingBox(**f["bbox"]), confidence=f["confidence"])
            for f in el["fragments"]
        ]

        if el["kind"] == "table_cell":
            element = TableCellElement(
                id=el["id"],
                table_id=el["table_id"],
                row=el["row"],
                col=el["col"],
                bbox=bbox,
                fragments=fragments,
                merged_text=el["merged_text"],
            )
        else:
            element = FreeFieldElement(
                id=el["id"],
                bbox=bbox,
                fragments=fragments,
                merged_text=el["merged_text"],
            )

        match = MatchResult(**raw["match"])
        return MatchedElement(element=element, match=match)
