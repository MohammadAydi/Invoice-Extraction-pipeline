"""Composition root: wires every stage together from a single AppConfig
and exposes one `run()` entry point. This is the only module in the
project that knows the full pipeline shape end to end -- every other
module only knows its own interface.
"""

from __future__ import annotations

import uuid

from config.schema import AppConfig
from core.domain.image_payload import ImagePayload
from core.domain.matching import KeywordDictionary, MatchedElement
from core.domain.result import PipelineResult
from mapping.cell_mapper import CellMapper
from ocr.factory import build_ocr_engine
from output.factory import build_output_formatter
from persistence.file_result_store import FileResultStore
from preprocessing.pipeline_builder import PreprocessingPipelineBuilder
from string_matching.factory import build_string_matcher
from table_extraction.factory import build_table_extractor


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
        raise NotImplementedError("Load the keyword list from disk into a KeywordDictionary.")

    def run(self, raw_image: ImagePayload):
        invoice_id = str(uuid.uuid4())

        # 1. Geometric correction -- runs once. Its output is BOTH the
        #    left-hand (plain) and right-hand (overlaid) image in the UI.
        display_payload = self.geometric_pipeline.run(raw_image)

        # 2. Two photometric branches, run sequentially (per current
        #    requirements). Each PreprocessingPipeline.run() call is
        #    independent and stateless, so switching to concurrent
        #    execution later is a one-line change here, not a redesign.
        ocr_payload = self.ocr_photometric_pipeline.run(display_payload)
        table_payload = self.table_photometric_pipeline.run(display_payload)

        # 3. Recognition, sequentially.
        ocr_result = self.ocr_engine.recognize(ocr_payload)
        table_result = self.table_extractor.extract(table_payload)

        # 4. Mapping: OCR fragments -> table cells / free fields.
        structured_doc = self.cell_mapper.map(ocr_result, table_result)

        # 5. String matching -- same keyword dictionary for every element,
        #    table cell or free field alike (per project decision).
        matched_elements = [
            MatchedElement(
                element=el,
                match=self.string_matcher.match(el.merged_text, self.keyword_dictionary),
            )
            for el in structured_doc.elements
        ]

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
        return self.output_formatter.format(result)

    @staticmethod
    def _save_display_image(payload: ImagePayload, invoice_id: str) -> str:
        raise NotImplementedError(
            "Persist the geometrically corrected image to disk/storage; return its path/URL."
        )
