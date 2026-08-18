"""Composition root: wires every stage together from a single AppConfig
and exposes one `run()` entry point. This is the only module in the
project that knows the full pipeline shape end to end -- every other
module only knows its own interface.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from config.schema import AppConfig
from core.domain.image_payload import ImagePayload
from core.domain.layout import InvoiceLayout
from core.domain.matching import KeywordDictionary, MatchedElement, MatchResult
from core.domain.result import PipelineResult
from core.domain.run import PipelineRun
from core.exceptions import NotImplementedStrategyError
from core.domain.invoice import InvoiceDraft, InvoiceWarning, WarningCodes
from core.domain.catalog import Catalogs
from invoice.parser import InvoiceParser
from mapping.cell_mapper import CellMapper
from orchestration.flows.factory import build_flow
from output.factory import build_output_formatter
from persistence.file_result_store import FileResultStore
from preprocessing.pipeline_builder import PreprocessingPipelineBuilder
from string_matching.factory import build_string_matcher

logger = logging.getLogger(__name__)

# Steps after which the image is binary. The "enhanced" diagnostic image is the
# page as it looked immediately before the first of these ran -- grayscale, so a
# human can still read it, which is the whole reason it is returned separately
# from the binarized image the engine actually read.
_BINARIZING_STEPS = frozenset({"fixed_threshold", "otsu_threshold", "adaptive_threshold"})


class PipelineOrchestrator:
    def __init__(self, config: AppConfig, ocr_components=None):
        """`ocr_components` lets a caller inject already-constructed models.

        The HTTP layer needs this: a request may change a preprocessing
        parameter, which means a new orchestrator, but rebuilding the detector
        and recognizer along with it would reload the model weights -- about a
        minute for the VLM path -- for a change that has nothing to do with
        reading text.
        """
        self.config = config

        # Geometric correction runs ONCE and is shared by both branches --
        # this is what guarantees OCR bboxes and table bboxes end up in the
        # same coordinate space as the image the UI displays on both sides.
        self.geometric_pipeline = PreprocessingPipelineBuilder.build(
            config.preprocessing.geometric_steps
        )
        self.ocr_photometric_pipeline = PreprocessingPipelineBuilder.build(
            config.preprocessing.ocr_photometric_steps
        )
        self.table_photometric_pipeline = PreprocessingPipelineBuilder.build(
            config.preprocessing.table_photometric_steps
        )

        self.cell_mapper = CellMapper()

        # The flow owns the table extractor, the layout classifier and whichever
        # of detector/cropper/recognizer/engine it declared. Which three stages
        # run, and in what order, is its decision -- see orchestration/flows/.
        self.flow = build_flow(config, self.cell_mapper, components=ocr_components)

        self.string_matcher = build_string_matcher(config.string_matching)
        self.output_formatter = build_output_formatter(config.output)
        self.invoice_parser = InvoiceParser()
        self.result_store = FileResultStore(**config.persistence.store_params)

        self.keyword_dictionary = self._load_keyword_dictionary(
            config.string_matching.dictionary_path
        )

    # ------------------------------------------------------------------
    # Readiness, for the HTTP layer's /health probe
    # ------------------------------------------------------------------

    @property
    def engine_name(self) -> str:
        return self.flow.engine_name

    def is_ready(self) -> bool:
        """Whether this pipeline can serve a request right now."""
        return self.flow.is_ready()

    def warmup(self) -> None:
        """Load weights / verify binaries up front, so readiness is not a lie."""
        self.flow.warmup()

    @staticmethod
    def _load_keyword_dictionary(path: str) -> KeywordDictionary:
        file = Path(path)
        if not file.exists():
            # A missing dictionary disables keyword correction; it must not stop
            # the service from starting, since catalog matching is independent
            # of it and is what the desktop app actually relies on.
            logger.warning("Keyword dictionary '%s' not found; keyword matching disabled.", path)
            return KeywordDictionary(keywords=[], source_path=path)

        data = json.loads(file.read_text(encoding="utf-8"))
        return KeywordDictionary(keywords=data.get("keywords", []), source_path=path)

    @staticmethod
    def _run_optional_stage(stage_name, fn, fallback_factory):
        """Runs a downstream stage that may be a deliberately-unbuilt strategy.

        Catches `NotImplementedStrategyError` only -- the exception a registered
        but unimplemented strategy raises on purpose. A plain
        `NotImplementedError` escaping from inside working code is a bug, and
        catching it here used to convert that bug into silently empty output:
        the run completed, the response validated, and the page came back blank
        with nothing to say why.
        """
        try:
            return fn()
        except NotImplementedStrategyError as exc:
            logger.warning("Skipping stage '%s' (not implemented yet): %s", stage_name, exc)
            return fallback_factory()

    def run(
        self,
        raw_image: ImagePayload,
        catalogs: Catalogs | None = None,
        write_debug_images: bool = True,
        persist: bool = True,
    ) -> PipelineRun:
        """Run one invoice end to end.

        `catalogs` carries the per-request match targets the desktop app sends
        (its Customers, Products and the cities it knows). They are a property
        of the request, not of the configuration, which is why they arrive here
        rather than through AppConfig.

        `write_debug_images` and `persist` are off for HTTP requests: writing
        one PNG per preprocessing step per request would turn the results
        directory into the slowest part of the service.
        """
        invoice_id = str(uuid.uuid4())
        invoice_dir = self.result_store.output_dir / invoice_id
        debug_root = invoice_dir / "preprocessing" if write_debug_images else None

        # 1. Geometric correction -- runs once. Its output is BOTH the
        #    left-hand (plain) and right-hand (overlaid) image in the UI.
        display_payload = self.geometric_pipeline.run(
            raw_image, debug_dir=debug_root / "geometric" if debug_root else None
        )

        # 2. Two photometric branches, run sequentially (per current
        #    requirements). Each PreprocessingPipeline.run() call is
        #    independent and stateless, so switching to concurrent
        #    execution later is a one-line change here, not a redesign.
        ocr_snapshots: dict = {}
        ocr_payload = self.ocr_photometric_pipeline.run(
            display_payload,
            debug_dir=debug_root / "ocr_photometric" if debug_root else None,
            snapshots=ocr_snapshots,
        )
        table_payload = self.table_photometric_pipeline.run(
            display_payload,
            debug_dir=debug_root / "table_photometric" if debug_root else None,
        )

        # 3-4. Layout analysis, recognition and mapping, sequenced by the
        #      configured flow. Which of them run in which order is the flow's
        #      decision and the only thing the three strategies disagree on;
        #      everything either side of this call is identical for all three.
        outcome = self.flow.run(ocr_payload, table_payload)
        ocr_result = outcome.ocr_result
        structured_doc = outcome.document

        # 5. String matching -- same keyword dictionary for every element,
        #    table cell or free field alike (per project decision). This is the
        #    generic keyword correction; the invoice stage below does the
        #    catalog matching that produces the candidate lists the UI binds to.
        matched_elements = []
        for el in structured_doc.elements:
            match = self._run_optional_stage(
                "string_matcher.match",
                lambda el=el: self.string_matcher.match(el.merged_text, self.keyword_dictionary),
                lambda el=el: MatchResult(
                    corrected_text=el.merged_text, confidence=0.0, alternatives=[]
                ),
            )
            matched_elements.append(MatchedElement(element=el, match=match))

        # 6. Read the elements as an invoice: header fields, line items,
        #    warnings, each matched against the request's catalogs.
        invoice = self._run_optional_stage(
            "invoice_parser.parse",
            lambda: self.invoice_parser.parse(structured_doc, ocr_result, catalogs),
            lambda: InvoiceDraft(),
        )

        # A flow that could not do its job says so on the wire, not only in the
        # log. "layout_driven found no grid" and "this invoice is blank" produce
        # the same empty result otherwise, and the user can only act on the
        # first one.
        for message in outcome.warnings:
            logger.warning("[%s] %s", self.flow.name, message)
            invoice.warnings.append(
                InvoiceWarning(code=WarningCodes.NO_LAYOUT_DETECTED, message=message)
            )

        # 7. Assemble the canonical result, with a full config snapshot for
        #    audit/reproducibility, and persist it.
        result = PipelineResult(
            invoice_id=invoice_id,
            display_image_path=(
                self._save_display_image(display_payload, invoice_id) if persist else ""
            ),
            elements=matched_elements,
            invoice=invoice,
            raw_text="\n".join(
                f.text for f in ocr_result.fragments if (f.text or "").strip()
            ),
            ocr_engine=ocr_result.engine_name,
            config_snapshot=self.config.model_dump(),
        )

        if persist:
            self.result_store.save(result)

        # 8. Format for the UI (or whatever OutputFormatter is configured).
        output = self._run_optional_stage(
            "output_formatter.format",
            lambda: self.output_formatter.format(result),
            lambda: {"invoice_id": result.invoice_id},
        )

        return PipelineRun(
            result=result,
            output=output,
            layout=outcome.layout,
            display_image=display_payload.image,
            enhanced_image=self._enhanced_image(ocr_snapshots, ocr_payload.image),
            ocr_input_image=ocr_payload.image,
            applied_steps=list(ocr_payload.applied_steps),
        )

    @staticmethod
    def _enhanced_image(snapshots: dict, fallback: np.ndarray) -> np.ndarray:
        """The page as it looked just before binarization.

        Falls back to the final OCR image when no thresholding step ran, which
        is the correct answer for a configuration that reads a grayscale page
        directly -- there is no separate "before binarization" state to show.
        """
        # Insertion order is execution order, so this finds the *first*
        # binarizing step even if a configuration runs more than one.
        for step_name, image in snapshots.items():
            if step_name in _BINARIZING_STEPS:
                return image
        return snapshots.get("__final__", fallback)

    def _save_display_image(self, payload: ImagePayload, invoice_id: str) -> str:
        invoice_dir = self.result_store.output_dir / invoice_id
        invoice_dir.mkdir(parents=True, exist_ok=True)
        path = invoice_dir / "display.png"
        cv2.imwrite(str(path), payload.image)
        return str(path)
