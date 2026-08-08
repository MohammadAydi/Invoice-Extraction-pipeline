"""Output of the mapping stage.

Both kinds of element share one bbox in one coordinate space (see
core.domain.geometry), so the UI can treat "detected field or table cell"
uniformly, per the project's UI requirements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

from core.domain.geometry import BoundingBox
from core.domain.ocr import OCRFragment


@dataclass
class TableCellElement:
    id: str
    table_id: str
    row: int
    col: int
    bbox: BoundingBox
    fragments: list[OCRFragment]
    merged_text: str
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
    kind: Literal["free_field"] = "free_field"


DocumentElement = TableCellElement | FreeFieldElement


@dataclass
class StructuredDocument:
    elements: list[DocumentElement] = field(default_factory=list)
