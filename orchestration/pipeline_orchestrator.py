"""Composition root: wires every stage together from a single AppConfig
and exposes one `run()` entry point. This is the only module in the
project that knows the full pipeline shape end to end -- every other
module only knows its own interface.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import uuid
from pathlib import Path

import cv2

from config.schema import AppConfig
from core.domain.document import StructuredDocument
from core.domain.image_payload import ImagePayload
from core.domain.matching import KeywordDictionary, MatchedElement, MatchResult
from core.domain.ocr import OCRResult
from core.domain.result import PipelineResult
from core.domain.table import TableExtractionResult
from mapping.cell_mapper import CellMapper
from ocr.factory import build_ocr_engine
from output.factory import build_output_formatter
from persistence.file_result_store import FileResultStore
from preprocessing.pipeline_builder import PreprocessingPipelineBuilder
from string_matching.factory import build_string_matcher
from table_extraction.factory import build_table_extractor

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    def __init__(self, config: AppConfig):
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

        self.ocr_engine = build_ocr_engine(config.ocr)
        self.table_extractor = build_table_extractor(config.table_extraction)
        self.string_matcher = build_string_matcher(config.string_matching)
        self.output_formatter = build_output_formatter(config.output)
        self.cell_mapper = CellMapper()
        self.result_store = FileResultStore(**config.persistence.store_params)

        self.keyword_dictionary = self._load_keyword_dictionary(
            config.string_matching.dictionary_path
        )

    @staticmethod
    def _load_keyword_dictionary(path: str) -> KeywordDictionary:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return KeywordDictionary(keywords=data.get("keywords", []), source_path=path)

    @staticmethod
    def _run_optional_stage(stage_name, fn, fallback_factory):
        """Runs a downstream stage that may still be an unimplemented stub.

        If `fn` raises NotImplementedError (the stage hasn't been built by
        its owner yet), log a warning and substitute a placeholder value of
        the correct type instead of crashing the whole run -- preprocessing
        (and everything up to it) should still complete and persist.
        """
        try:
            return fn()
        except NotImplementedError as exc:
            logger.warning("Skipping stage '%s' (not implemented yet): %s", stage_name, exc)
            return fallback_factory()

    def run(self, raw_image: ImagePayload):
        invoice_id = str(uuid.uuid4())
        invoice_dir = self.result_store.output_dir / invoice_id
        debug_root = invoice_dir / "preprocessing"

        # 1. Geometric correction -- runs once. Its output is BOTH the
        #    left-hand (plain) and right-hand (overlaid) image in the UI.
        display_payload = self.geometric_pipeline.run(
            raw_image, debug_dir=debug_root / "geometric"
        )

        # 2. Two photometric branches, run sequentially (per current
        #    requirements). Each PreprocessingPipeline.run() call is
        #    independent and stateless, so switching to concurrent
        #    execution later is a one-line change here, not a redesign.
        ocr_payload = self.ocr_photometric_pipeline.run(
            display_payload, debug_dir=debug_root / "ocr_photometric"
        )
        table_payload = self.table_photometric_pipeline.run(
            display_payload, debug_dir=debug_root / "table_photometric"
        )

        # 3. Recognition, sequentially. Both stages may still be
        #    unimplemented for other engines/extractors -- skip gracefully.
        ocr_result = self._run_optional_stage(
            "ocr_engine.recognize",
            lambda: self.ocr_engine.recognize(ocr_payload),
            lambda: OCRResult(fragments=[], engine_name=f"{self.config.ocr.engine}(unimplemented)"),
        )
        table_result = self._run_optional_stage(
            "table_extractor.extract",
            lambda: self.table_extractor.extract(table_payload),
            lambda: TableExtractionResult(
                tables=[], extractor_name=f"{self.config.table_extraction.extractor}(unimplemented)"
            ),
        )

        # 4. Mapping: OCR fragments -> table cells / free fields.
        structured_doc = self._run_optional_stage(
            "cell_mapper.map",
            lambda: self.cell_mapper.map(ocr_result, table_result),
            lambda: StructuredDocument(elements=[]),
        )

        # 5. String matching -- same keyword dictionary for every element,
        #    table cell or free field alike (per project decision).
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

        # 6. Assemble the canonical result, with a full config snapshot for
        #    audit/reproducibility, and persist it.
        result = PipelineResult(
            invoice_id=invoice_id,
            display_image_path=self._save_display_image(display_payload, invoice_id),
            elements=matched_elements,
            config_snapshot=self.config.model_dump(),
        )
        self.result_store.save(result)

        # 7. Format for the UI (or whatever OutputFormatter is configured).
        return self._run_optional_stage(
            "output_formatter.format",
            lambda: self.output_formatter.format(result),
            lambda: dataclasses.asdict(result),
        )

    def _save_display_image(self, payload: ImagePayload, invoice_id: str) -> str:
        invoice_dir = self.result_store.output_dir / invoice_id
        invoice_dir.mkdir(parents=True, exist_ok=True)
        path = invoice_dir / "display.png"
        cv2.imwrite(str(path), payload.image)
        return str(path)
