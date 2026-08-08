from __future__ import annotations

from typing import Protocol

from core.domain.image_payload import ImagePayload
from core.domain.table import TableExtractionResult


class TableExtractor(Protocol):
    def extract(self, image: ImagePayload) -> TableExtractionResult: ...
