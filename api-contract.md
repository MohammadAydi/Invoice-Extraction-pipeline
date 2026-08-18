# API Contract — ai-service ⇄ desktop-app

Base URL: `http://127.0.0.1:8765` (configurable via `AppSettings['PythonServiceUrl']`).
All endpoints are versioned under `/api/v1`. All bodies are UTF-8 JSON unless stated.

---

## GET `/api/v1/health`

Liveness + readiness. The desktop app polls this after launching the service.

**200 Response**

```json
{
  "status": "ok",
  "version": "1.0.0",
  "ocr_engine": "stub",
  "engine_ready": true,
  "languages": ["ar", "en"]
}
```

`engine_ready` is `false` while a real OCR engine is still loading weights. The desktop
app treats `engine_ready: false` as "not ready yet" and keeps polling.

---

## POST `/api/v1/extract`

The single extraction endpoint. Accepts `multipart/form-data`.

| Part | Type | Required | Notes |
|---|---|---|---|
| `file` | binary | yes | `.png .jpg .jpeg` — no other format is accepted |
| `options` | JSON string | no | See below |

**`options`**

```json
{
  "languages": ["ar", "en"],
  "invoice_type": "Purchase",
  "known_merchants": [
    { "customer_id": 12, "name": "مؤسسة النور التجارية", "aliases": ["متجر النور"] },
    { "customer_id": 13, "name": "Ahmad Trading Est.", "aliases": ["Ahmad Trading"] }
  ],
  "known_products": [
    { "product_id": 31, "name": "Sugar 1kg" },
    { "product_id": 32, "name": "أرز بسمتي" }
  ],
  "return_debug_images": false
}
```

`return_debug_images` adds `enhanced_image_png` and `ocr_input_image_png` to the response;
see [Debug images](#debug-images) below.

`known_merchants` / `known_products` are sent from the C# side out of the Customers and
Products tables. The service fuzzy-matches OCR output against them, which is what lifts
merchant-name accuracy above raw OCR. Both are optional; omitting them disables matching.

Each catalog entry is a **record**, not a loose string:

| Field | Meaning |
|---|---|
| `customer_id` / `product_id` | Primary key, returned as `matched_id` on a hit |
| `name` | The canonical name — `Customers.Name` / `Products.Name` |
| `aliases` | Merchants only: every other name the contact answers to, currently `AliasName` |

`name` and each entry in `aliases` are **equivalent match targets**. An invoice printed
with an alias scores against that alias but resolves to the same record, and the response
carries the canonical `name` in `matched_to` with the alias in `matched_name`. Products
have no aliases: the catalog holds only an id and a name, so the name is the entire
matching surface.

Both catalogs also accept a plain array of name strings (`["Ahmad Trading"]`), which is
treated as one single-name entry per string with no id.

**200 Response**

```json
{
  "request_id": "3f2a9c1e-5b7d-4e0a-9c3f-1d2e3a4b5c6d",
  "processing_ms": 1840,
  "source": {
    "filename": "invoice_042.jpg",
    "page_count": 1,
    "page_used": 1,
    "width": 1700,
    "height": 2200
  },
  "header": {
    "merchant_name":  { "value": "مؤسسة النور التجارية", "confidence": 0.94, "matched_to": "مؤسسة النور التجارية", "matched_id": 12, "matched_name": "متجر النور", "match_score": 0.97, "bbox": [120, 80, 640, 150] },
    "invoice_number": { "value": "INV-2291",  "confidence": 0.88, "bbox": [1200, 90, 1580, 140] },
    "invoice_date":   { "value": "2026-03-14", "confidence": 0.81, "raw": "14/03/2026", "bbox": [1200, 160, 1580, 210] },
    "city":           { "value": "Amman", "confidence": 0.72, "bbox": null },
    "total_amount":   { "value": 128.75, "confidence": 0.91, "bbox": [1300, 1980, 1600, 2040] }
  },
  "line_items": [
    {
      "row_index": 0,
      "product_name": { "value": "Sugar 1kg", "confidence": 0.90, "matched_to": "Sugar 1kg", "matched_id": 31, "match_score": 0.88 },
      "quantity":     { "value": 2.0,   "confidence": 0.95 },
      "unit_price":   { "value": 1.25,  "confidence": 0.93 },
      "total_price":  { "value": 2.50,  "confidence": 0.93 },
      "arithmetic_ok": true
    }
  ],
  "warnings": [
    { "code": "LOW_CONFIDENCE_FIELD", "field": "city", "message": "City confidence 0.72 is below 0.75" }
  ],
  "elements": [
    {
      "id": "9f1c:r3c4",
      "kind": "table_cell",
      "role": "product_name",
      "zone": "table",
      "bbox": [1069, 1180, 1449, 1240],
      "raw_text": "بورش أزرق ولادي",
      "corrected_text": "بورش أزرق ولادي",
      "confidence": 0.0,
      "candidates": [],
      "editable": true,
      "table_id": "9f1c",
      "row": 3,
      "col": 4,
      "row_span": 1,
      "col_span": 1
    }
  ],
  "raw_text": "...full OCR dump, newline separated...",
  "enhanced_image_png": null,
  "ocr_input_image_png": null
}
```

### Overlay elements

One entry per box on the page — every cell of the detected grid plus every piece of text
that landed outside one. This is what the verification screen draws its clickable regions
from, so a cell the recognizer could not read still appears, with `raw_text` empty: a
missing element would silently shift the row it belongs to.

| Key | Meaning |
|---|---|
| `id` | Stable within one response. `"<table_id>:r<row>c<col>"` for a grid cell, `"free:<n>"` otherwise |
| `kind` | `table_cell` or `free_field` |
| `role` | Which invoice field the layout stage decided this box holds — see below |
| `zone` | `header`, `table`, `footer`, or `unknown` |
| `bbox` | `[x1, y1, x2, y2]` in the coordinate space of the displayed image |
| `row_span` / `col_span` | >1 where a printed divider is absent, e.g. a totals row spanning the table |

`role` is one of `unknown`, `merchant_name`, `invoice_number`, `invoice_date`, `city`,
`column_header`, `line_number`, `product_name`, `quantity`, `unit_price`, `line_total`,
`notes`, `total_amount`, `total_in_words`, `total_in_figures`, `label`.

Both `role` and `zone` are **`"unknown"` when no layout classifier ran**, which is the case
for any configuration that does not select one. `"unknown"` therefore means "not
classified", not "classified as nothing" — treat it as absent rather than as a value.

When a role *is* present, the corresponding header field's `bbox` points at that same box,
so clicking the merchant field on the verification screen highlights the box the value was
actually read from rather than a line of reconstructed text.

### Debug images

Two optional diagnostic images, both base64-encoded PNG, both `null` unless
`options.return_debug_images` was `true`:

| Key | Content |
|---|---|
| `enhanced_image_png` | The enhanced grayscale page — preprocessing output immediately before binarization. Easier for a human to read than the binary image. |
| `ocr_input_image_png` | The exact image the OCR engine read (binarized). |

**Both are downscaled** to `DEBUG_IMAGE_MAX_WIDTH` (default 1200 px) before encoding —
they travel inside the JSON body, and a full-resolution scan would dominate the payload.
Downscaling is never upscaling: an image already narrower than the limit is sent as-is.

Consequently **`bbox` coordinates do not map 1:1 onto these images.** Boxes are in full
preprocessed-image space, whose dimensions are reported in `source.width` / `source.height`;
overlaying one on a returned image requires scaling by the ratio between them. (Nothing
consumes `bbox` today — this is a caveat for whatever does.)

If encoding fails, both keys stay `null`, a `DEBUG_IMAGE_UNAVAILABLE` warning is added, and
the extraction itself still returns `200` — a diagnostic image is never worth failing a
good extraction over.

### Field envelope

Every extracted value uses the same envelope so the UI can render confidence uniformly:

| Key | Type | Meaning |
|---|---|---|
| `value` | string \| number \| null | Normalized value. `null` when not found. |
| `confidence` | float 0–1 | OCR confidence. `0.0` when `value` is null. |
| `raw` | string? | Pre-normalization text, when normalization changed it. |
| `matched_to` | string? | **Canonical** name of the catalog record matched — always `Customers.Name` / `Products.Name`, even when an alias is what scored. |
| `matched_id` | int? | Primary key of that record (`CustomerId` / `ProductId`), so the app binds to the row rather than re-resolving the name. |
| `matched_name` | string? | The specific name that scored — the alias, when the invoice was printed with one. Diagnostic; the app files the invoice under `matched_to`. |
| `match_score` | float? | 0–1 similarity for that match. |
| `bbox` | `[x1,y1,x2,y2]`? | Pixel box in the **preprocessed** image space (`source.width` × `source.height`) — *not* the space of the downscaled debug images. |

**Normalization guarantees.** `invoice_date` is always ISO `YYYY-MM-DD` or `null`.
Monetary and quantity values are JSON numbers, never strings, with Arabic-Indic digits
(`٠١٢٣٤٥٦٧٨٩`) folded to ASCII and thousands separators stripped. `arithmetic_ok` is the
service's own check of `quantity × unit_price ≈ total_price` within 0.01 — the C# side
re-validates independently and does not trust this flag.

### Confidence thresholds

The service does not reject low-confidence results; it surfaces them. The desktop app
highlights any field below **0.75** in amber on the verification screen, prompting review.

### Warning codes

| Code | Meaning |
|---|---|
| `LOW_CONFIDENCE_FIELD` | A field scored below the review threshold |
| `ARITHMETIC_MISMATCH` | A line item's qty × price ≠ total |
| `TOTAL_MISMATCH` | Sum of line items ≠ header total |
| `NO_LINE_ITEMS` | Table detection found no rows |
| `SKEW_CORRECTED` | Deskew rotated the image by >1° |
| `DEBUG_IMAGE_UNAVAILABLE` | A requested debug image could not be encoded; extraction unaffected |
| `MANUAL_REVIEW_REQUIRED` | A field did not match the catalog confidently; candidates offered |
| `NO_LAYOUT_DETECTED` | The `layout_driven` flow found no printed grid, so nothing was read. Distinguishes "no grid on this page" from "this page is blank" |

Warnings are advisory. A response with warnings is still `200`.

---

## Errors

All failures return this shape with a 4xx/5xx status:

```json
{
  "error": {
    "code": "UNSUPPORTED_FORMAT",
    "message": "File extension '.docx' is not a supported invoice format.",
    "detail": null
  }
}
```

| HTTP | Code | Cause |
|---|---|---|
| 400 | `UNSUPPORTED_FORMAT` | Extension not in the allowed set |
| 400 | `CORRUPT_FILE` | File could not be decoded as an image |
| 400 | `EMPTY_FILE` | Zero-byte upload |
| 413 | `FILE_TOO_LARGE` | Exceeds 25 MB |
| 422 | `INVALID_OPTIONS` | `options` was not valid JSON / failed schema validation |
| 503 | `ENGINE_NOT_READY` | OCR weights still loading |
| 500 | `OCR_FAILED` | Engine raised; `detail` carries the exception summary |

The C# client maps every one of these to `AiServiceException` carrying `Code`, so
ViewModels can branch on the code rather than parse messages.

---

## Contract stability

The C# DTOs in `Services/AiServiceClient/Contracts/` and the Pydantic models in
`ai-service/app/schemas/` are the two halves of this document. **Changing one without the
other breaks the system** — update this file first, then both sides. Unknown JSON fields
are ignored by both sides, so additive changes are backward-compatible.
