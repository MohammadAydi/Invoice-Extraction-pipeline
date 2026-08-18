from __future__ import annotations

from typing import Protocol, Sequence

from core.domain.ocr import RegionCrop


class TextRecognizer(Protocol):
    """Reads one prepared crop. Knows nothing about invoices.

    The crop carries a :class:`~core.domain.roles.ContentKind`, not a role, so a
    recognizer asked for a NUMBER behaves the same whether that number is a
    quantity or a grand total. That is what keeps this layer reusable for a form
    that is not an invoice.
    """

    def read(self, crop: RegionCrop) -> str: ...

    def read_all(self, crops: Sequence[RegionCrop]) -> list[str]:
        """Read a batch. The default is a loop; implementations that can
        genuinely batch (a GPU recognizer with a real batch dimension) override
        it. Nothing may reorder or drop entries -- callers index the result
        against the crops they passed in.
        """
        ...
