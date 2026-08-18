from __future__ import annotations

from dataclasses import dataclass

from core.domain.geometry import BoundingBox


@dataclass(frozen=True)
class TableCell:
    """One cell of a detected grid. Purely geometric -- what the cell *means*
    is decided later, by a LayoutClassifier, and lives on a LayoutRegion.

    `row_span`/`col_span` are real: a totals row with no vertical dividers
    drawn across it is one wide cell, not six narrow ones, and the extractors
    already work this out by checking whether a separator is actually inked.
    """

    bbox: BoundingBox
    row: int
    col: int
    table_id: str
    row_span: int = 1
    col_span: int = 1


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

    @property
    def cells(self) -> list[TableCell]:
        return [cell for table in self.tables for cell in table.cells]
