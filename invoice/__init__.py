"""Reading a StructuredDocument as an invoice.

The models this produces -- `InvoiceDraft` and everything it holds -- live in
`core.domain.invoice` with the rest of the domain. What is here is the parsing:
the header heuristics, the column assignment, and the line reconstruction the
header path needs.
"""

from invoice.parser import InvoiceParser  # noqa: F401
from invoice.text_lines import group_lines  # noqa: F401

__all__ = ["InvoiceParser", "group_lines"]
