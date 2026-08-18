"""Pydantic models for the extraction API.

These are the Python half of `docs/api-contract.md`; the C# DTOs in
`InvoiceDigitizationApp/Services/AiServiceClient/Contracts.cs` are the other
half. Change the document first, then both sides. Unknown JSON fields are
ignored by both, so additive changes stay backward-compatible.
"""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from config.settings_contract import PipelineSettings

T = TypeVar("T")

# Fields scoring below this are flagged for human review rather than rejected.
REVIEW_CONFIDENCE_THRESHOLD = 0.75


class MatchCandidate(BaseModel):
    """One entry of a field's ranked candidate list.

    `similarity_score` is a **percentage**, 0-100, rounded to two places -- the
    shape the matching spec defines and the one the desktop app displays beside
    each dropdown entry. The field-level `match_score` is the same number as a
    0-1 fraction, kept for the confidence rendering that already works that way.
    """

    model_config = ConfigDict(extra="ignore")

    matched_value: str
    similarity_score: float = Field(ge=0.0, le=100.0)

    # Primary key of the catalog record, when the candidate came from one.
    matched_id: int | None = None

    # The specific name that scored, when it differs from the canonical value.
    matched_name: str | None = None


class ExtractedField(BaseModel, Generic[T]):
    """Envelope shared by every extracted value.

    A uniform shape lets the desktop UI render confidence the same way for
    every field instead of special-casing each one.
    """

    model_config = ConfigDict(populate_by_name=True)

    value: T | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # Pre-normalization text, present only when normalization changed the value.
    raw: str | None = None

    # Canonical catalog entry this value was matched to. For a merchant this is
    # always the customer's Name, even when the hit came through an alias.
    matched_to: str | None = None

    # Primary key of the matched catalog row (CustomerId / ProductId), so the
    # desktop app can bind straight to the record instead of resolving the name
    # a second time and possibly resolving it differently.
    matched_id: int | None = None

    # The specific name that scored the match.
    matched_name: str | None = None

    match_score: float | None = Field(default=None, ge=0.0, le=1.0)

    # Up to five ranked alternatives, best first. The app pre-selects
    # candidates[0] in an editable combo box and offers the rest, so a
    # near-miss costs one click rather than a re-typed line.
    candidates: list[MatchCandidate] = Field(default_factory=list)

    # True when the top candidate was too weak to accept unseen -- or when
    # there was none. The verification screen highlights these.
    requires_manual_review: bool = False

    # [x1, y1, x2, y2] in *preprocessed* image pixel space.
    bbox: list[int] | None = None


class ExtractionSource(BaseModel):
    filename: str | None = None
    page_count: int = 1
    page_used: int = 1
    width: int = 0
    height: int = 0


class ExtractedHeader(BaseModel):
    merchant_name: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    invoice_number: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    invoice_date: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    city: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    total_amount: ExtractedField[float] = Field(default_factory=ExtractedField[float])


class ExtractedLineItem(BaseModel):
    row_index: int = 0
    product_name: ExtractedField[str] = Field(default_factory=ExtractedField[str])

    # Quantity is an integer by the normalization rules: separators inside a
    # handwritten quantity are mis-read strokes, not decimal points.
    quantity: ExtractedField[int] = Field(default_factory=ExtractedField[int])
    unit_price: ExtractedField[float] = Field(default_factory=ExtractedField[float])
    total_price: ExtractedField[float] = Field(default_factory=ExtractedField[float])

    # Advisory. The C# side re-validates independently and does not trust this.
    arithmetic_ok: bool = True


class OverlayElement(BaseModel):
    """One box on the page, for the image overlay on the verification screen.

    Everything the OCR engine found appears here, including text no invoice
    field claimed, so the overlay can show the user what was read where.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    kind: str

    # Which invoice field the layout stage decided this box holds
    # ("invoice_number", "quantity", "total_amount", ...) and roughly where it
    # sits ("header" / "table" / "footer"). Both are "unknown" when no layout
    # classifier ran, so a client can tell "not classified" from "classified as
    # nothing".
    role: str = "unknown"
    zone: str = "unknown"

    bbox: list[int]
    raw_text: str = ""
    corrected_text: str = ""
    confidence: float = 0.0
    candidates: list[MatchCandidate] = Field(default_factory=list)
    editable: bool = True

    table_id: str | None = None
    row: int | None = None
    col: int | None = None

    # A totals row with no vertical dividers drawn across it is one wide cell,
    # not several narrow ones.
    row_span: int = 1
    col_span: int = 1


class ExtractionWarning(BaseModel):
    code: str
    field: str | None = None
    message: str


class ExtractionResult(BaseModel):
    request_id: str
    processing_ms: int = 0
    source: ExtractionSource = Field(default_factory=ExtractionSource)
    header: ExtractedHeader = Field(default_factory=ExtractedHeader)
    line_items: list[ExtractedLineItem] = Field(default_factory=list)
    warnings: list[ExtractionWarning] = Field(default_factory=list)
    raw_text: str = ""

    # The pipeline's own id for this run, which is also the folder name under
    # `results/`. Sending it back lets a support request name one run.
    invoice_id: str | None = None
    ocr_engine: str | None = None

    elements: list[OverlayElement] = Field(default_factory=list)

    # Diagnostic images, base64 PNG, present only when
    # options.return_debug_images is set. Both are downscaled to
    # settings.debug_image_max_width, so `bbox` coordinates -- which are in
    # full preprocessed-image space -- do not map onto them 1:1.
    enhanced_image_png: str | None = None
    ocr_input_image_png: str | None = None


# --------------------------------------------------------------------------
# Request catalogs
# --------------------------------------------------------------------------


def _unique_names(*values: str | None) -> list[str]:
    """Non-empty, trimmed names with case-insensitive duplicates dropped."""
    seen: set[str] = set()
    names: list[str] = []

    for value in values:
        text = (value or "").strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            names.append(text)

    return names


class KnownMerchant(BaseModel):
    """A contact from the Customers table, with every name it is known by.

    `name` and `aliases` are *equivalent* for matching: an invoice printed with
    the alias identifies exactly the same record as one printed with the
    canonical name. Which one matched is reported back, but either resolves to
    this entry.
    """

    model_config = ConfigDict(extra="ignore")

    customer_id: int | None = None
    name: str = ""
    aliases: list[str] = Field(default_factory=list)

    @property
    def entry_id(self) -> int | None:
        return self.customer_id

    @property
    def names(self) -> list[str]:
        return _unique_names(self.name, *self.aliases)

    @classmethod
    def coerce(cls, value: Any) -> "KnownMerchant":
        """Accept a bare name string as well as the full object.

        Keeps callers that only have a list of names -- and older clients --
        working without a second code path in the parser.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(name=value)
        return cls.model_validate(value)


class KnownProduct(BaseModel):
    """A row of the Products table as a match target.

    The catalog holds nothing but an id and a name, so the name is the entire
    matching surface; the id only travels back so the app can link the line to
    the record.
    """

    model_config = ConfigDict(extra="ignore")

    product_id: int | None = None
    name: str = ""

    @property
    def entry_id(self) -> int | None:
        return self.product_id

    @property
    def names(self) -> list[str]:
        return _unique_names(self.name)

    @classmethod
    def coerce(cls, value: Any) -> "KnownProduct":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(name=value)
        return cls.model_validate(value)


class KnownCity(BaseModel):
    """A city or governorate the app knows about.

    There is no Cities table: the desktop app sends the distinct values of
    `Customers.City`, so the matcher scores the OCR text against places this
    installation actually deals with rather than a generic gazetteer.
    """

    model_config = ConfigDict(extra="ignore")

    city_id: int | None = None
    name: str = ""
    aliases: list[str] = Field(default_factory=list)

    @property
    def entry_id(self) -> int | None:
        return self.city_id

    @property
    def names(self) -> list[str]:
        return _unique_names(self.name, *self.aliases)

    @classmethod
    def coerce(cls, value: Any) -> "KnownCity":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(name=value)
        return cls.model_validate(value)


class ExtractionOptions(BaseModel):
    """Options accompanying the upload. Every field has a usable default."""

    model_config = ConfigDict(extra="ignore")

    languages: list[str] = Field(default_factory=lambda: ["ar", "en"])
    invoice_type: Literal["Sale", "Purchase"] | None = None

    # Sent from the C# Customers/Products tables; empty disables matching for
    # that field and leaves the raw OCR text with an empty candidate list.
    known_merchants: list[KnownMerchant] = Field(default_factory=list)
    known_products: list[KnownProduct] = Field(default_factory=list)
    known_cities: list[KnownCity] = Field(default_factory=list)

    # How many ranked alternatives to return per matched field.
    max_candidates: int = Field(default=5, ge=1, le=25)

    return_debug_images: bool = False

    # The full pipeline configuration, per docs/settings-config-contract.md.
    # Omitted means "use the service's own default config file", which is what
    # the CLI and the tests rely on.
    configuration: PipelineSettings | None = None

    # Bare name strings are still accepted for every catalog, so a client that
    # has only names -- or one written against the previous contract -- works.
    @field_validator("known_merchants", mode="before")
    @classmethod
    def _accept_merchant_names(cls, value: Any) -> Any:
        return [KnownMerchant.coerce(v) for v in value] if isinstance(value, list) else value

    @field_validator("known_products", mode="before")
    @classmethod
    def _accept_product_names(cls, value: Any) -> Any:
        return [KnownProduct.coerce(v) for v in value] if isinstance(value, list) else value

    @field_validator("known_cities", mode="before")
    @classmethod
    def _accept_city_names(cls, value: Any) -> Any:
        return [KnownCity.coerce(v) for v in value] if isinstance(value, list) else value


class HealthStatus(BaseModel):
    status: str = "ok"
    version: str
    ocr_engine: str
    engine_ready: bool
    languages: list[str] = Field(default_factory=list)


class ErrorBody(BaseModel):
    code: str
    message: str
    detail: str | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody


class WarningCodes:
    LOW_CONFIDENCE_FIELD = "LOW_CONFIDENCE_FIELD"
    ARITHMETIC_MISMATCH = "ARITHMETIC_MISMATCH"
    TOTAL_MISMATCH = "TOTAL_MISMATCH"
    NO_LINE_ITEMS = "NO_LINE_ITEMS"
    SKEW_CORRECTED = "SKEW_CORRECTED"
    DEBUG_IMAGE_UNAVAILABLE = "DEBUG_IMAGE_UNAVAILABLE"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"
    NO_LAYOUT_DETECTED = "NO_LAYOUT_DETECTED"


class ErrorCodes:
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    CORRUPT_FILE = "CORRUPT_FILE"
    EMPTY_FILE = "EMPTY_FILE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_OPTIONS = "INVALID_OPTIONS"
    INVALID_CONFIGURATION = "INVALID_CONFIGURATION"
    ENGINE_NOT_READY = "ENGINE_NOT_READY"
    OCR_FAILED = "OCR_FAILED"
