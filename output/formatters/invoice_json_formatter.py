"""The formatter the desktop app consumes: one named key per invoice field.

This is the Python half of the `/api/v1/extract` 200 response documented in
`docs/api-contract.md`; the C# DTOs in
`InvoiceDigitizationApp/Services/AiServiceClient/Contracts.cs` are the other
half. Change the document first, then both sides.

Two shapes, and which one a field gets is the contract's whole point:

* **A value field** -- `invoice_id`, `date`, `quantity`, `unit_price`,
  `total_price`, `total_invoice_price`. One reading, its OCR confidence, and the
  box it came from. There is nothing to match it against, so there is nothing to
  choose.
* **A matched field** -- `customer_name`, `city`, `product_name`. These are
  matched against the catalogs the request carried, and they deliberately have
  **no** `value`: they carry `original_value`, which is always the raw OCR text,
  plus `results`, the ranked catalog entries. Picking one is the desktop app's
  decision, made against its own threshold. A wrong confident match silently
  corrupts an invoice, so the shape refuses to make that call here rather than
  relying on both sides to agree not to.

The formatter produces the *body*. The envelope around it -- timings, source
dimensions, debug images -- belongs to the HTTP layer, which is the only place
that knows about the request, and is added in `api/routes.py`.

`ui_overlay_json` is still registered and is what the CLI and the `results/`
audit trail use: it is lossless, carrying every detected box whether or not an
invoice field claimed it.
"""

from __future__ import annotations

from core.domain.geometry import BoundingBox
from core.domain.invoice import ExtractedValue
from core.domain.result import PipelineResult
from output.registry import formatter_registry
from string_matching.normalization import format_date, parse_date


def _bbox(bbox: BoundingBox | None) -> dict | None:
    """The contract's box: origin plus size, not two corners."""
    if bbox is None:
        return None
    return {"x": int(bbox.x), "y": int(bbox.y), "w": int(bbox.w), "h": int(bbox.h)}


def _confidence(value: ExtractedValue | None) -> float:
    return round(float(value.confidence), 4) if value is not None else 0.0


def _text_value(value: ExtractedValue | None) -> str | None:
    """A value field holding text."""
    if value is None or value.value is None:
        return None
    text = str(value.value).strip()
    return text or None


def _number_value(value: ExtractedValue | None, integral: bool = False):
    """A value field holding a number, as a real JSON number.

    Amounts travel as numbers rather than formatted strings: the desktop app
    parses them into `decimal` for exact arithmetic validation, and a string
    would put a locale-dependent parse between the two halves.
    """
    if value is None or value.value is None:
        return None

    try:
        number = float(value.value)
    except (TypeError, ValueError):
        return None

    return int(round(number)) if integral else round(number, 2)


def _date_value(value: ExtractedValue | None) -> str | None:
    """The date as ISO-8601, or null.

    The contract promises one or the other, so a reading the date rules could not
    parse is sent as null rather than as itself. The parser deliberately keeps
    that raw text -- it is in `results/<id>/result.json` and worth having when
    diagnosing a bad read -- but putting it on the wire would break the one
    guarantee a client can act on, in exchange for a string the user can read off
    the image beside it anyway.
    """
    text = _text_value(value)
    if text is None:
        return None

    parsed = parse_date(text)
    return format_date(parsed)


def _value_field(value: ExtractedValue | None, kind: str = "text") -> dict:
    if kind == "integer":
        payload = _number_value(value, integral=True)
    elif kind == "number":
        payload = _number_value(value)
    elif kind == "date":
        payload = _date_value(value)
    else:
        payload = _text_value(value)

    return {
        "value": payload,
        "ocr_confidence": _confidence(value),
        "bounding_box": _bbox(value.bbox if value else None),
    }


def _matched_field(value: ExtractedValue | None) -> dict:
    """A field the catalogs were searched for, with its ranked answer.

    `original_value` is the OCR text, always -- never a candidate substituted in
    for it. The candidates are what `results` is for, and the app applies the
    review threshold to them.
    """
    results = []

    for candidate in (value.candidates if value else []):
        matched_id = candidate.get("matched_id")
        results.append(
            {
                # A string, as the contract specifies; the app parses it back to
                # the Customers/Products primary key it came from.
                "id": None if matched_id is None else str(matched_id),
                "value": candidate.get("matched_value", ""),
                # A 0-1 fraction here, unlike the internal 0-100 percentage.
                "string_matching_score": round(
                    float(candidate.get("similarity_score", 0.0)) / 100.0, 4
                ),
            }
        )

    return {
        "bounding_box": _bbox(value.bbox if value else None),
        "ocr_confidence": _confidence(value),
        "original_value": _text_value(value),
        "results": results,
    }


@formatter_registry.register("invoice_json")
class InvoiceJsonFormatter:
    """Serializes a run as the invoice the desktop app binds to."""

    def __init__(self, **params):
        self.params = params

    def format(self, result: PipelineResult) -> dict:
        invoice = result.invoice
        header = invoice.header if invoice else None

        return {
            # The invoice number printed on the paper -- NOT the pipeline's own
            # run id, which stays in the service log and under results/.
            "invoice_id": _value_field(header.invoice_number if header else None),
            "customer_name": _matched_field(header.merchant_name if header else None),
            "date": _value_field(header.invoice_date if header else None, kind="date"),
            "city": _matched_field(header.city if header else None),
            "products": [
                {
                    "product_name": _matched_field(item.product_name),
                    "quantity": _value_field(item.quantity, kind="integer"),
                    "unit_price": _value_field(item.unit_price, kind="number"),
                    "total_price": _value_field(item.total_price, kind="number"),
                }
                for item in (invoice.line_items if invoice else [])
            ],
            "total_invoice_price": _value_field(
                header.total_amount if header else None, kind="number"
            ),
        }
