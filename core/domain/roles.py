"""What a region of the page *is*, as opposed to where it is.

Cell detection answers "there is a box here". This module is the vocabulary for
answering "and it holds the invoice number" -- which is what lets the mapping
and parsing stages read a labelled field instead of guessing at one from
keywords and x-positions.

The split into three enums is deliberate:

* :class:`Zone` -- coarse position relative to the item table. Cheap to compute
  from geometry alone and useful even when nothing else is known.
* :class:`CellRole` -- the invoice-specific meaning. Only a
  :class:`~core.interfaces.layout_classifier.LayoutClassifier` assigns these,
  because they encode knowledge of one printed form.
* :class:`ContentKind` -- what *kind of characters* the region holds. This is
  the only one the recognizer sees. Keeping it separate is what stops the OCR
  layer from having to know what an invoice is: a recognizer asked for a NUMBER
  behaves identically whether the number is a quantity or a grand total.
"""

from __future__ import annotations

from enum import Enum


class Zone(str, Enum):
    """Where a region sits relative to the line-item table."""

    HEADER = "header"
    TABLE = "table"
    FOOTER = "footer"
    UNKNOWN = "unknown"


class CellRole(str, Enum):
    """The invoice field a region holds."""

    UNKNOWN = "unknown"

    # Header
    MERCHANT_NAME = "merchant_name"
    INVOICE_NUMBER = "invoice_number"
    INVOICE_DATE = "invoice_date"
    CITY = "city"

    # Table
    COLUMN_HEADER = "column_header"
    LINE_NUMBER = "line_number"
    PRODUCT_NAME = "product_name"
    QUANTITY = "quantity"
    UNIT_PRICE = "unit_price"
    LINE_TOTAL = "line_total"
    NOTES = "notes"

    # Footer
    TOTAL_AMOUNT = "total_amount"
    TOTAL_IN_WORDS = "total_in_words"
    TOTAL_IN_FIGURES = "total_in_figures"

    # A printed caption ("الاسم:", "كتابة:") rather than a value. Recognized so
    # the overlay can show it, but never read as a field.
    LABEL = "label"


class ContentKind(str, Enum):
    """What kind of characters a region holds, and therefore how to read it."""

    NUMBER = "number"
    ARABIC_TEXT = "arabic_text"
    DATE = "date"
    ANY = "any"


# A date is NOT a NUMBER: the number prompt tells the model to write digits and
# never a letter, which mangles the separators in "٢٠٢٦/٨/١٥". It gets its own
# kind, and its own prompt.
CONTENT_KIND_BY_ROLE: dict[CellRole, ContentKind] = {
    CellRole.UNKNOWN: ContentKind.ANY,
    CellRole.MERCHANT_NAME: ContentKind.ARABIC_TEXT,
    CellRole.INVOICE_NUMBER: ContentKind.NUMBER,
    CellRole.INVOICE_DATE: ContentKind.DATE,
    CellRole.CITY: ContentKind.ARABIC_TEXT,
    CellRole.COLUMN_HEADER: ContentKind.ARABIC_TEXT,
    CellRole.LINE_NUMBER: ContentKind.NUMBER,
    CellRole.PRODUCT_NAME: ContentKind.ARABIC_TEXT,
    CellRole.QUANTITY: ContentKind.NUMBER,
    CellRole.UNIT_PRICE: ContentKind.NUMBER,
    CellRole.LINE_TOTAL: ContentKind.NUMBER,
    CellRole.NOTES: ContentKind.ARABIC_TEXT,
    CellRole.TOTAL_AMOUNT: ContentKind.NUMBER,
    CellRole.TOTAL_IN_WORDS: ContentKind.ARABIC_TEXT,
    CellRole.TOTAL_IN_FIGURES: ContentKind.NUMBER,
    CellRole.LABEL: ContentKind.ARABIC_TEXT,
}

ZONE_BY_ROLE: dict[CellRole, Zone] = {
    CellRole.MERCHANT_NAME: Zone.HEADER,
    CellRole.INVOICE_NUMBER: Zone.HEADER,
    CellRole.INVOICE_DATE: Zone.HEADER,
    CellRole.CITY: Zone.HEADER,
    CellRole.COLUMN_HEADER: Zone.TABLE,
    CellRole.LINE_NUMBER: Zone.TABLE,
    CellRole.PRODUCT_NAME: Zone.TABLE,
    CellRole.QUANTITY: Zone.TABLE,
    CellRole.UNIT_PRICE: Zone.TABLE,
    CellRole.LINE_TOTAL: Zone.TABLE,
    CellRole.NOTES: Zone.TABLE,
    CellRole.TOTAL_AMOUNT: Zone.FOOTER,
    CellRole.TOTAL_IN_WORDS: Zone.FOOTER,
    CellRole.TOTAL_IN_FIGURES: Zone.FOOTER,
}


def content_kind_for(role: CellRole) -> ContentKind:
    return CONTENT_KIND_BY_ROLE.get(role, ContentKind.ANY)


def zone_for(role: CellRole) -> Zone:
    return ZONE_BY_ROLE.get(role, Zone.UNKNOWN)
