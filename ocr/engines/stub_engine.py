"""Deterministic fake OCR engine.

Exists so the desktop app can be built, demoed and tested end to end before a
real engine is installed -- no Tesseract binary, no model weights, no second
virtualenv. It fabricates a plausible Arabic/English invoice whose content is
*derived from a hash of the image bytes*, so the same file always yields the
same result, which is what makes it usable in automated tests.

It is never a fallback for a failed real engine: a silent switch to fabricated
data would be far worse than an error. Selecting it is always explicit, by
setting `ocr.engine: stub` in the configuration.
"""

from __future__ import annotations

import hashlib

import numpy as np

from core.domain.geometry import BoundingBox
from core.domain.image_payload import ImagePayload
from core.domain.ocr import OCRFragment, OCRResult
from ocr.registry import engine_registry

# Sample catalogs used to synthesize invoices. Mixed Arabic and English on
# purpose: real invoices here carry Arabic merchant and product names beside
# Latin-digit amounts.
_MERCHANTS = [
    "متجر النور",
    "مؤسسة الرياض التجارية",
    "Ahmad Trading Est.",
    "سوبرماركت السلام",
    "Gulf Office Supplies",
]

_PRODUCTS = [
    ("سكر 1كغ", 1.25),
    ("أرز بسمتي 5كغ", 8.50),
    ("زيت زيتون 1ل", 12.00),
    ("A4 Paper Ream", 4.75),
    ("Blue Pen box", 3.20),
    ("شاي أحمر 500غ", 6.40),
    ("حليب طويل الأمد", 2.10),
]

_CITIES = ["عمان", "دمشق", "Irbid", "الزرقاء", "حلب"]


@engine_registry.register("stub")
class StubEngine:
    def __init__(self, **engine_params):
        self.engine_params = engine_params

    def is_ready(self) -> bool:
        return True

    def warmup(self) -> None:
        return None

    def recognize(self, image: ImagePayload) -> OCRResult:
        array = image.image
        height, width = array.shape[:2]

        # Seeded from image content, so identical input gives identical output
        # across runs and across machines.
        digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)

        merchant = _MERCHANTS[seed % len(_MERCHANTS)]
        city = _CITIES[(seed >> 8) % len(_CITIES)]
        invoice_number = f"INV-{(seed >> 16) % 9000 + 1000}"
        day = (seed >> 24) % 28 + 1
        month = (seed >> 32) % 12 + 1
        date_text = f"{day:02d}/{month:02d}/2026"

        fragments: list[OCRFragment] = []

        def add(text: str, x: int, y: int, w: int, h: int = 40) -> None:
            fragments.append(
                OCRFragment(
                    text=text,
                    bbox=BoundingBox(x=x, y=y, w=w, h=h),
                    # 0.82-0.98: high enough to look real, variable enough that
                    # the low-confidence UI path actually gets exercised.
                    confidence=float(rng.uniform(0.82, 0.98)),
                )
            )

        margin = int(width * 0.06)
        right = width - margin

        # Labels and values are emitted as separate boxes, the way a real engine
        # returns them -- a single "المدينة: دمشق" token would not exercise the
        # label/value association the parser actually has to perform.
        add(merchant, margin, int(height * 0.04), int(width * 0.42), 60)

        add("المدينة:", margin, int(height * 0.11), int(width * 0.08))
        add(city, margin + int(width * 0.10), int(height * 0.11), int(width * 0.15))

        add("رقم الفاتورة", right - int(width * 0.30), int(height * 0.04), int(width * 0.14))
        add(invoice_number, right - int(width * 0.14), int(height * 0.04), int(width * 0.14))

        add("التاريخ", right - int(width * 0.30), int(height * 0.10), int(width * 0.08))
        add(date_text, right - int(width * 0.20), int(height * 0.10), int(width * 0.20))

        # Column header row, then the item rows beneath it.
        table_top = int(height * 0.22)
        add("الصنف", margin, table_top, int(width * 0.30))
        add("الكمية", margin + int(width * 0.34), table_top, int(width * 0.08))
        add("السعر", margin + int(width * 0.46), table_top, int(width * 0.12))
        add("المبلغ", margin + int(width * 0.62), table_top, int(width * 0.12))

        item_count = int(rng.integers(2, 6))
        row_height = max(1, int(height * 0.055))
        running_total = 0.0

        chosen = rng.choice(len(_PRODUCTS), size=item_count, replace=False)
        for i, product_index in enumerate(chosen):
            name, unit_price = _PRODUCTS[int(product_index)]
            quantity = int(rng.integers(1, 8))
            line_total = round(quantity * unit_price, 2)
            running_total += line_total

            y = table_top + row_height * (i + 1)
            add(name, margin, y, int(width * 0.30))
            add(str(quantity), margin + int(width * 0.34), y, int(width * 0.08))
            add(f"{unit_price:.2f}", margin + int(width * 0.46), y, int(width * 0.12))
            add(f"{line_total:.2f}", margin + int(width * 0.62), y, int(width * 0.12))

        total_y = table_top + row_height * (item_count + 2)
        add("الإجمالي", margin + int(width * 0.46), total_y, int(width * 0.12))
        add(f"{running_total:.2f}", margin + int(width * 0.62), total_y, int(width * 0.12))

        return OCRResult(fragments=fragments, engine_name="stub")
