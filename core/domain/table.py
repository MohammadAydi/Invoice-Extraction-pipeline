from __future__ import annotations

from dataclasses import dataclass

from core.domain.geometry import BoundingBox


@dataclass(frozen=True)
class TableCell:
    bbox: BoundingBox
    row: int
    col: int
    table_id: str


@dataclass(frozen=True)
class TableStructure:
    table_id: str
    bbox: BoundingBox
    cells: list[TableCell]


@dataclass(frozen=True)
class TableExtractionResult:
    tables: list[TableStructure]
    extractor_name: str
    raw_extractor_output: object = None
