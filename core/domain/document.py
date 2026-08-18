"""Output of the mapping stage.

Both kinds of element share one bbox in one coordinate space (see
core.domain.geometry), so the UI can treat "detected field or table cell"
uniformly, per the project's UI requirements.

Both also carry a `role` and a `zone`. When a layout classifier ran, those say
which invoice field the element holds -- which is what lets the parser read a
labelled header field instead of inferring one from keywords. When none ran they
are UNKNOWN, and the parser falls back to its heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from core.domain.geometry import BoundingBox
from core.domain.ocr import OCRFragment
from core.domain.roles import CellRole, Zone


@dataclass
class TableCellElement:
    id: str
    table_id: str
    row: int
    col: int
    bbox: BoundingBox
    fragments: list[OCRFragment]
    merged_text: str
    role: CellRole = CellRole.UNKNOWN
    zone: Zone = Zone.TABLE
    row_span: int = 1
    col_span: int = 1
    kind: Literal["table_cell"] = "table_cell"


@dataclass
class FreeFieldElement:
    """A detected field (invoice number, date, vendor, ...) that doesn't
    belong to any table cell but still needs its own bbox + editable text.
    """

    id: str
    bbox: BoundingBox
    fragments: list[OCRFragment]
    merged_text: str
    role: CellRole = CellRole.UNKNOWN
    zone: Zone = Zone.UNKNOWN
    kind: Literal["free_field"] = "free_field"


DocumentElement = TableCellElement | FreeFieldElement


@dataclass
class StructuredDocument:
    elements: list[DocumentElement] = field(default_factory=list)

    def with_role(self, role: CellRole) -> list[DocumentElement]:
        return [el for el in self.elements if el.role is role]

    def first_with_role(self, role: CellRole) -> DocumentElement | None:
        return next((el for el in self.elements if el.role is role), None)

    @property
    def has_roles(self) -> bool:
        return any(el.role is not CellRole.UNKNOWN for el in self.elements)
