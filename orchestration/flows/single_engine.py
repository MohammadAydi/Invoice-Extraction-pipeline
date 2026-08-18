"""One OCR engine reads the whole page.

The default flow, and what every engine that takes a page and returns text uses:
`stub`, `tesseract`, `surya`, `easyocr`, `qwen_vlm`. It has no detector, no
cropper and no recognizer -- the engine is all three at once, internally.

It still analyzes the layout, because mapping the fragments it produced onto
cells is what turns loose text into an invoice.
"""

from __future__ import annotations

from core.domain.image_payload import ImagePayload
from core.domain.layout import InvoiceLayout
from core.domain.ocr import OCRResult
from orchestration.flows.base import ExtractionFlow
from orchestration.flows.registry import flow_registry


@flow_registry.register("single_engine")
class SingleEngineFlow(ExtractionFlow):
    name = "single_engine"
    REQUIRES = ("engine",)

    def __init__(self, table_extractor, layout_classifier, cell_mapper, engine):
        super().__init__(table_extractor, layout_classifier, cell_mapper)
        self.engine = engine

    @classmethod
    def from_config(cls, config, components, table_extractor, layout_classifier, cell_mapper):
        return cls(
            table_extractor,
            layout_classifier,
            cell_mapper,
            engine=components.engine,
            **config.flow.params,
        )

    def _read_text(
        self, payload: ImagePayload, layout: InvoiceLayout, debug=None
    ) -> OCRResult:
        return self.engine.recognize(payload)

    def components(self) -> list[object]:
        return super().components() + [self.engine]

    @property
    def engine_name(self) -> str:
        return getattr(self.engine, "name", type(self.engine).__name__)