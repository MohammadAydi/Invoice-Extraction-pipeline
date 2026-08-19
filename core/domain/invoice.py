"""The invoice-shaped view of a pipeline run.

Everything upstream of this package speaks in elements: a box, some text, a
confidence. The desktop app needs an invoice: a merchant, a date, a total, a
list of lines. These dataclasses are that translation's output, and
`output.formatters.ui_overlay_formatter` is what puts them on the wire in the
shape `docs/api-contract.md` specifies.

They are dataclasses rather than Pydantic models on purpose: this is the
pipeline's own domain, and it should not depend on the HTTP layer's schemas. The
API layer validates the serialized form on the way out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.domain.geometry import BoundingBox
from core.domain.catalog import FieldMatch

# Fields scoring below this are flagged for review rather than rejected. Shared
# with the API contract's confidence threshold so the verification screen has
# one number to highlight against.
REVIEW_CONFIDENCE_THRESHOLD = 0.75


def _bbox_to_wire(bbox: BoundingBox | None) -> list[int] | None:
    """Convert the pipeline's (x, y, w, h) to the contract's [x1, y1, x2, y2]."""
    if bbox is None:
        return None
    return [bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h]


@dataclass
class ExtractedValue:
    """One extracted field, in the envelope every field on the wire shares.

    `value` is what the app should show. `candidates` is the ranked alternative
    list behind it, so the field renders as an editable dropdown whose first
    entry is already selected -- confirming a correct read costs nothing and
    correcting a wrong one costs a click.
    """

    value: str | float | int | None = None
    confidence: float = 0.0

    # Pre-normalization text, present only when normalization changed the value.
    raw: str | None = None

    # Canonical catalog name matched, its primary key, and the specific name
    # that scored (the alias, when the invoice was printed with one).
    matched_to: str | None = None
    matched_id: int | None = None
    matched_name: str | None = None
    match_score: float | None = None

    candidates: list[dict] = field(default_factory=list)
    requires_manual_review: bool = False

    bbox: BoundingBox | None = None

    @classmethod
    def from_match(
        cls,
        match: FieldMatch,
        fallback_text: str | None,
        confidence: float,
        bbox: BoundingBox | None = None,
    ) -> "ExtractedValue":
        """Build a field from a catalog match, keeping the OCR text as the value.

        The raw reading **always** stands as `value`; a candidate never replaces
        it, however well it scored. `candidates` is the answer, and choosing from
        it is the desktop app's decision, made against its own threshold — see
        the `original_value` + `results` shape in `docs/api-contract.md`. Two
        sides applying the same threshold independently is one side too many: a
        wrong confident match silently corrupts an invoice, and there is no way
        to tell afterwards that a substitution happened.

        `matched_to` / `matched_id` are still recorded when the top candidate
        cleared the threshold, because the `results/` audit trail is worth being
        able to read back. They are not on the wire.
        """
        best = match.best
        accepted = best is not None and not match.requires_manual_review

        return cls(
            value=fallback_text or None,
            confidence=confidence,
            raw=None,
            matched_to=best.matched_value if accepted else None,
            matched_id=best.matched_id if accepted else None,
            matched_name=best.matched_name if accepted else None,
            match_score=best.score_fraction if best else None,
            candidates=[candidate.as_dict() for candidate in match.candidates],
            requires_manual_review=match.requires_manual_review,
            bbox=bbox,
        )

    def to_wire(self) -> dict:
        return {
            "value": self.value,
            "confidence": round(float(self.confidence or 0.0), 4),
            "raw": self.raw,
            "matched_to": self.matched_to,
            "matched_id": self.matched_id,
            "matched_name": self.matched_name,
            "match_score": (
                round(float(self.match_score), 4) if self.match_score is not None else None
            ),
            "candidates": self.candidates,
            "requires_manual_review": self.requires_manual_review,
            "bbox": _bbox_to_wire(self.bbox),
        }

    @property
    def is_low_confidence(self) -> bool:
        return self.value is not None and self.confidence < REVIEW_CONFIDENCE_THRESHOLD


@dataclass
class InvoiceHeader:
    merchant_name: ExtractedValue = field(default_factory=ExtractedValue)
    invoice_number: ExtractedValue = field(default_factory=ExtractedValue)
    invoice_date: ExtractedValue = field(default_factory=ExtractedValue)
    city: ExtractedValue = field(default_factory=ExtractedValue)
    total_amount: ExtractedValue = field(default_factory=ExtractedValue)

    def named_fields(self) -> list[tuple[str, ExtractedValue]]:
        return [
            ("merchant_name", self.merchant_name),
            ("invoice_number", self.invoice_number),
            ("invoice_date", self.invoice_date),
            ("city", self.city),
            ("total_amount", self.total_amount),
        ]

    def to_wire(self) -> dict:
        return {name: value.to_wire() for name, value in self.named_fields()}


@dataclass
class InvoiceLineItem:
    row_index: int = 0
    product_name: ExtractedValue = field(default_factory=ExtractedValue)
    quantity: ExtractedValue = field(default_factory=ExtractedValue)
    unit_price: ExtractedValue = field(default_factory=ExtractedValue)
    total_price: ExtractedValue = field(default_factory=ExtractedValue)

    # The service's own qty x price check. Advisory: the C# side re-validates
    # independently and does not trust this flag.
    arithmetic_ok: bool = True

    def to_wire(self) -> dict:
        return {
            "row_index": self.row_index,
            "product_name": self.product_name.to_wire(),
            "quantity": self.quantity.to_wire(),
            "unit_price": self.unit_price.to_wire(),
            "total_price": self.total_price.to_wire(),
            "arithmetic_ok": self.arithmetic_ok,
        }


@dataclass
class InvoiceWarning:
    code: str
    message: str
    field: str | None = None

    def to_wire(self) -> dict:
        return {"code": self.code, "field": self.field, "message": self.message}


@dataclass
class InvoiceDraft:
    """The parsed invoice, before the user has confirmed any of it."""

    header: InvoiceHeader = field(default_factory=InvoiceHeader)
    line_items: list[InvoiceLineItem] = field(default_factory=list)
    warnings: list[InvoiceWarning] = field(default_factory=list)

    def to_wire(self) -> dict:
        return {
            "header": self.header.to_wire(),
            "line_items": [item.to_wire() for item in self.line_items],
            "warnings": [warning.to_wire() for warning in self.warnings],
        }


class WarningCodes:
    LOW_CONFIDENCE_FIELD = "LOW_CONFIDENCE_FIELD"
    ARITHMETIC_MISMATCH = "ARITHMETIC_MISMATCH"
    TOTAL_MISMATCH = "TOTAL_MISMATCH"
    NO_LINE_ITEMS = "NO_LINE_ITEMS"
    SKEW_CORRECTED = "SKEW_CORRECTED"
    DEBUG_IMAGE_UNAVAILABLE = "DEBUG_IMAGE_UNAVAILABLE"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"
    NO_LAYOUT_DETECTED = "NO_LAYOUT_DETECTED"
