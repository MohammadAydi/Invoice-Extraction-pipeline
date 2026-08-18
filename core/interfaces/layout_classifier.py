from __future__ import annotations

from typing import Protocol

from core.domain.layout import InvoiceLayout
from core.domain.table import TableExtractionResult


class LayoutClassifier(Protocol):
    """Turns detected boxes into labelled invoice fields.

    Separate from the extractor because the two answer different questions and
    change for different reasons. "Where are the ruled boxes on this photo" is a
    computer-vision problem and is the same for every form. "The box above the
    table on the left is the invoice number" is knowledge of one printed form,
    and a second supplier's form changes it without changing a line of the
    detection code.

    `image_size` is (width, height) of the page the cells were measured in --
    needed because most classification rules are relative ("the topmost cell",
    "the leftmost third").
    """

    def classify(
        self, table: TableExtractionResult, image_size: tuple[int, int]
    ) -> InvoiceLayout: ...
