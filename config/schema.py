"""Single source of truth for all pipeline configuration.

Every runtime-swappable choice mentioned in the project brief -- which
preprocessing steps run and in what order, which OCR engine, which table
extractor, which string-matching algorithm, which output format -- is a
field somewhere in AppConfig. Nothing downstream reads an environment
variable or a hardcoded default; everything is threaded through this
object.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class StepConfig(BaseModel):
    """One entry in a preprocessing step list.

    `name` must match a name registered in preprocessing.steps.registry.
    `enabled` and list position are what give you "which steps run" and
    "in what order" from one place.
    """

    name: str
    enabled: bool = True
    params: dict = Field(default_factory=dict)


class PreprocessingConfig(BaseModel):
    # Runs ONCE, upstream of both branches below. See core/domain/geometry.py
    # for why geometric correction is factored out on its own.
    geometric_steps: list[StepConfig] = Field(default_factory=list)

    # Photometric-only steps (no coordinate changes), one ordered list per
    # branch, both operating on the same geometrically corrected image.
    ocr_photometric_steps: list[StepConfig] = Field(default_factory=list)
    table_photometric_steps: list[StepConfig] = Field(default_factory=list)


class OCRConfig(BaseModel):
    engine: str
    engine_params: dict = Field(default_factory=dict)


class TableExtractionConfig(BaseModel):
    extractor: str
    extractor_params: dict = Field(default_factory=dict)


class StringMatchingConfig(BaseModel):
    algorithm: str
    algorithm_params: dict = Field(default_factory=dict)
    dictionary_path: str


class OutputConfig(BaseModel):
    formatter: str
    formatter_params: dict = Field(default_factory=dict)


class PersistenceConfig(BaseModel):
    store: str = "file_result_store"
    store_params: dict = Field(default_factory=dict)


class AppConfig(BaseModel):
    preprocessing: PreprocessingConfig
    ocr: OCRConfig
    table_extraction: TableExtractionConfig
    string_matching: StringMatchingConfig
    output: OutputConfig
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
