"""Pydantic models for the extraction API.

These are the Python half of `docs/api-contract.md`; the C# DTOs in
`InvoiceDigitizationApp/Services/AiServiceClient/Contracts.cs` are the other
half. Change the document first, then both sides. Unknown JSON fields are
ignored by both, so additive changes stay backward-compatible.

Two field shapes, and which one a field gets is the contract's central decision:

* :class:`ValueField` -- one reading, its confidence, its box. For anything
  there is no catalog to match against.
* :class:`MatchedField` -- **no** ``value`` at all. The raw OCR text under
  ``original_value``, plus the ranked catalog entries under ``results``. The
  desktop app picks, against its own threshold. A wrong confident match
  silently corrupts an invoice, so the shape refuses to make that call rather
  than relying on both sides to agree not to.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

T = TypeVar("T")

# Fields scoring below this are flagged for human review rather than rejected.
# The desktop app applies it; the service reports the scores that feed it.
REVIEW_CONFIDENCE_THRESHOLD = 0.75


class BoundingBox(BaseModel):
    """Origin plus size, in the coordinate space of the geometrically corrected
    page -- the same space `source.width` x `source.height` describes, and the
    same one every box in the response lives in.
    """

    model_config = ConfigDict(extra="ignore")

    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0


class MatchResultEntry(BaseModel):
    """One ranked catalog entry for a matched field."""

    model_config = ConfigDict(extra="ignore")

    # The catalog row's primary key as a string (CustomerId / ProductId), or
    # null for a match that came from no record.
    id: str | None = None

    value: str = ""

    # A 0-1 fraction, not a percentage.
    string_matching_score: float = Field(default=0.0, ge=0.0, le=1.0)


class ValueField(BaseModel, Generic[T]):
    """A field read straight off the page, with nothing to match it against."""

    model_config = ConfigDict(extra="ignore")

    value: T | None = None
    ocr_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    bounding_box: BoundingBox | None = None


class MatchedField(BaseModel):
    """A field matched against one of the request's catalogs.

    Deliberately has no ``value``: ``original_value`` is what the paper said and
    ``results`` is what the catalog offered, best first. Nothing here decides
    between them.
    """

    model_config = ConfigDict(extra="ignore")

    bounding_box: BoundingBox | None = None
    ocr_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    original_value: str | None = None
    results: list[MatchResultEntry] = Field(default_factory=list)


class ProductRow(BaseModel):
    """One line of the item table."""

    model_config = ConfigDict(extra="ignore")

    product_name: MatchedField = Field(default_factory=MatchedField)

    # An integer by the normalization rules: a separator inside a handwritten
    # quantity is a mis-read stroke, not a decimal point.
    quantity: ValueField[int] = Field(default_factory=ValueField[int])
    unit_price: ValueField[float] = Field(default_factory=ValueField[float])
    total_price: ValueField[float] = Field(default_factory=ValueField[float])


class ExtractionSource(BaseModel):
    filename: str | None = None
    page_count: int = 1
    page_used: int = 1
    width: int = 0
    height: int = 0


class ExtractionResult(BaseModel):
    """The whole 200 body. These keys and no others."""

    processing_ms: int = 0
    source: ExtractionSource = Field(default_factory=ExtractionSource)

    # The invoice number printed on the paper. NOT the pipeline's own run id --
    # that stays in the service log and under `results/`, and is not sent.
    invoice_id: ValueField[str] = Field(default_factory=ValueField[str])

    customer_name: MatchedField = Field(default_factory=MatchedField)
    date: ValueField[str] = Field(default_factory=ValueField[str])
    city: MatchedField = Field(default_factory=MatchedField)
    products: list[ProductRow] = Field(default_factory=list)
    total_invoice_price: ValueField[float] = Field(default_factory=ValueField[float])

    # Diagnostic images, base64 PNG, present only when
    # options.return_debug_images is set. Both are downscaled to
    # settings.debug_image_max_width, so the boxes above -- which are in full
    # corrected-page space -- do not map onto them 1:1.
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
    """Options accompanying the upload. Every field has a usable default.

    `languages` and `invoice_type` are gone. The pipeline is Arabic-primary and
    every prompt, keyword list and normalization rule in it is written for
    Arabic, so a language list was a knob that changed nothing; and whether an
    invoice is a sale or a purchase is a property of the record the desktop app
    files, not of the paper being read.

    The pipeline configuration is no longer nested here either -- it is its own
    `config` part of the multipart request. The two are sent by different parts
    of the app for different reasons: options come from the current batch, the
    configuration from the settings page.
    """

    model_config = ConfigDict(extra="ignore")

    # Sent from the C# Customers/Products tables; empty disables matching for
    # that field and leaves the raw OCR text with an empty candidate list.
    known_merchants: list[KnownMerchant] = Field(default_factory=list)
    known_products: list[KnownProduct] = Field(default_factory=list)
    known_cities: list[KnownCity] = Field(default_factory=list)

    # How many ranked alternatives to return per matched field.
    max_candidates: int = Field(default=5, ge=1, le=25)

    return_debug_images: bool = False

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


class ErrorCodes:
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    CORRUPT_FILE = "CORRUPT_FILE"
    EMPTY_FILE = "EMPTY_FILE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_OPTIONS = "INVALID_OPTIONS"
    INVALID_CONFIGURATION = "INVALID_CONFIGURATION"
    ENGINE_NOT_READY = "ENGINE_NOT_READY"
    OCR_FAILED = "OCR_FAILED"
