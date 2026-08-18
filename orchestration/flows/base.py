"""How a page gets read, as a Template Method.

Three strategies exist and they differ in exactly one respect -- where the
regions to read come from -- yet each still has to analyze the layout, read the
text, and assemble a document. Writing that sequence out three times would let
the three drift; branching on a mode flag inside one method would put all three
in one place and make adding a fourth an edit to working code.

So :meth:`ExtractionFlow.run` fixes the sequence once and is never overridden,
and the subclasses fill in the three hooks:

    run()
      +-- _analyze_layout()   hook: detect the grid, label the cells
      +-- _read_text()        hook: this is the one that actually differs
      +-- _assemble()         hook: fragments + layout -> StructuredDocument

`_crop_and_read` is a concrete helper the two region-based flows share, so
padding, upscaling, ordering and fragment construction are written once.

The flows live under `orchestration/` on purpose. Only the orchestration layer
knows the pipeline end to end; the detector, cropper and recognizer know nothing
about each other, and a flow is the object that coordinates them.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

from core.domain.document import StructuredDocument
from core.domain.image_payload import ImagePayload
from core.domain.layout import InvoiceLayout
from core.domain.ocr import OCRFragment, OCRResult, TextRegion
from core.domain.table import TableExtractionResult

logger = logging.getLogger(__name__)


@dataclass
class FlowOutcome:
    """Everything one reading strategy produced.

    `layout` is carried alongside the document because the parser needs the
    roles and the API layer wants to report which extractor/classifier pair ran.
    """

    layout: InvoiceLayout
    ocr_result: OCRResult
    document: StructuredDocument
    table_result: TableExtractionResult | None = None
    warnings: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)


class ExtractionFlow(ABC):
    """Template Method. `run()` fixes the sequence; subclasses fill the hooks."""

    name: str = "extraction_flow"

    def __init__(self, table_extractor, layout_classifier, cell_mapper,
                 debug_writer=None):
        self.table_extractor = table_extractor
        self.layout_classifier = layout_classifier
        self.cell_mapper = cell_mapper
        self.debug_writer = debug_writer

    # -- the template ---------------------------------------------------

    def run(
        self,
        ocr_payload: ImagePayload,
        table_payload: ImagePayload,
        debug=None,
    ) -> FlowOutcome:
        timings: dict = {}

        started = time.perf_counter()
        layout, table_result = self._analyze_layout(table_payload)
        timings["layout_s"] = round(time.perf_counter() - started, 3)

        started = time.perf_counter()
        ocr_result = self._read_text(ocr_payload, layout, debug)
        timings["read_s"] = round(time.perf_counter() - started, 3)

        started = time.perf_counter()
        document = self._assemble(ocr_result, layout)
        timings["assemble_s"] = round(time.perf_counter() - started, 3)

        return FlowOutcome(
            layout=layout,
            ocr_result=ocr_result,
            document=document,
            table_result=table_result,
            warnings=self._warnings(layout, ocr_result),
            timings=timings,
        )

    # -- hooks ----------------------------------------------------------

    @abstractmethod
    def _read_text(
        self, payload: ImagePayload, layout: InvoiceLayout, debug=None
    ) -> OCRResult:
        """Produce the page's text fragments. The one step the flows disagree on.

        Implementations pass `debug` straight through to `_crop_and_read`; they
        never write anything themselves.
        """

    def _analyze_layout(
        self, table_payload: ImagePayload
    ) -> tuple[InvoiceLayout, TableExtractionResult | None]:

        height, width = table_payload.image.shape[:2]

        try:
            table_result = self.table_extractor.extract(table_payload)
        except NotImplementedError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("Table extraction failed (%s); continuing without a grid.", exc)
            return InvoiceLayout(source="extraction_failed"), None

        layout = self.layout_classifier.classify(table_result, (width, height))
        return layout, table_result

    def _assemble(self, ocr_result: OCRResult, layout: InvoiceLayout) -> StructuredDocument:
        return self.cell_mapper.map(ocr_result, layout)

    def _warnings(self, layout: InvoiceLayout, ocr_result: OCRResult) -> list[str]:
        return []

    # -- shared concrete steps ------------------------------------------

    def _crop_and_read(
        self,
        payload: ImagePayload,
        regions: Sequence[TextRegion],
        engine_name: str,
        raw_output: object = None,
        debug=None,
    ) -> OCRResult:
        """Crop every region and read it, one fragment per region.

        The fragment keeps the *region's* bbox, never the crop's: the crop was
        padded and upscaled, and its size is an implementation detail of
        preparing pixels for a model. The coordinates that matter are the ones
        the region already had, in the one canonical space.
        """
        if not regions:
            return OCRResult(fragments=[], engine_name=engine_name, raw_engine_output=raw_output)

        started = time.perf_counter()
        crops = self.cropper.crop(payload, regions)
        crop_s = time.perf_counter() - started

        started = time.perf_counter()
        if self.debug_writer is not None and hasattr(self.recognizer, "read"):
            texts, read_times = [], []
            for crop in crops:
                t0 = time.perf_counter()
                texts.append(self.recognizer.read(crop))
                read_times.append(time.perf_counter() - t0)
        else:
            texts = self.recognizer.read_all(crops)
            read_times = None
        read_s = time.perf_counter() - started

        if len(texts) != len(crops):
            raise RuntimeError(
                f"{type(self.recognizer).__name__}.read_all returned {len(texts)} "
                f"results for {len(crops)} crops; callers index one against the other."
            )

        if self.debug_writer is not None:
            try:
                self.debug_writer.write(
                    payload.image, crops, texts,
                    engine_name=engine_name, read_times=read_times,
                )
            except Exception as exc:  # noqa: BLE001
                # Debug output is never worth failing a run for.
                logger.warning("Debug output failed (%s); continuing.", exc)

        fragments = [
            OCRFragment(
                text=text,
                bbox=crop.region.bbox,
                # Detection confidence, where there was one. A VLM recognizer
                # exposes no score, and inventing one would report a guess as a
                # certainty.
                confidence=crop.region.confidence,
                source_id=crop.region.source_id,
            )
            for crop, text in zip(crops, texts)
        ]

        if debug is not None:
            debug.write(payload.image, crops, texts)
        logger.info(
            "%s: %d region(s) -> %d crop(s) in %.2fs, read in %.2fs (%.3fs/crop)",
            self.name,
            len(regions),
            len(crops),
            crop_s,
            read_s,
            read_s / len(crops) if crops else 0.0,
        )

        return OCRResult(
            fragments=fragments,
            engine_name=engine_name,
            raw_engine_output=raw_output,
            timings={
                "crop_s": round(crop_s, 3),
                "read_s": round(read_s, 3),
                "crop_count": len(crops),
            },
        )

    # -- readiness, for /health -----------------------------------------

    def _probe(self, attr: str, default: bool) -> bool:
        results = []
        for component in self.components():
            probe = getattr(component, attr, None)
            if callable(probe):
                results.append(bool(probe()))
        return all(results) if results else default

    def components(self) -> list[object]:
        """Every collaborator whose readiness matters. Subclasses extend this."""
        return [self.table_extractor, self.layout_classifier]

    def is_ready(self) -> bool:
        """Whether this flow can serve a request right now.

        A component that does not implement `is_ready` has nothing to load -- it
        was constructed successfully, so it is ready.
        """
        return self._probe("is_ready", default=True)

    def warmup(self) -> None:
        """Load weights / verify binaries up front, so readiness is not a lie."""
        for component in self.components():
            warm = getattr(component, "warmup", None)
            if callable(warm):
                warm()

    @property
    def engine_name(self) -> str:
        return self.name