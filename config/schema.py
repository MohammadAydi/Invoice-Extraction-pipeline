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


class ComponentConfig(BaseModel):
    """A registered implementation plus its constructor arguments.

    The same two-field shape as `StepConfig` minus the ordering, used wherever a
    stage names one pluggable collaborator.
    """

    name: str
    params: dict = Field(default_factory=dict)


class OCRConfig(BaseModel):
    """How text gets read -- one shape covering all three flows.

    `engine` is the whole story for the `single_engine` flow: one object that
    takes a page and returns text. The four component fields are what the
    detector-driven and layout-driven flows compose instead, because reading
    handwritten Arabic well needs a detector and a recognizer that can be
    chosen independently. Only the fields the selected flow actually uses are
    required; see `flow.name` and `orchestration/flows/`.
    """

    engine: str | None = None
    engine_params: dict = Field(default_factory=dict)

    # Detector-driven only: find the ink, then straighten the boxes up against
    # the known column layout.
    detector: ComponentConfig | None = None
    refiner: ComponentConfig | None = None

    # Shared by both region-based flows.
    cropper: ComponentConfig | None = None
    recognizer: ComponentConfig | None = None


class FlowConfig(BaseModel):
    """Which end-to-end reading strategy runs.

    `single_engine` -- one OCREngine reads the whole page. The default, and what
    every light engine (stub, tesseract, surya, easyocr) uses.

    `detector_driven` -- a text detector finds boxes anywhere on the page, a
    refiner squares them up against the column layout, each is cropped and read.
    Finds text outside any ruled box, at the cost of needing the column
    boundaries declared in config.

    `layout_driven` -- the printed grid is detected and classified first, and
    each labelled cell is cropped and read with the prompt its role implies. The
    grid supplies the real column boundaries, so nothing has to be declared; but
    text outside a ruled box is not read at all.
    """

    name: str = "single_engine"
    params: dict = Field(default_factory=dict)


class TableExtractionConfig(BaseModel):
    extractor: str
    extractor_params: dict = Field(default_factory=dict)

    # What the detected boxes MEAN. Separate from the extractor because "find
    # the ruled regions" is the same problem for every form while "the box above
    # the table is the invoice number" is knowledge of one supplier's form.
    # `passthrough` labels nothing, which is the pre-classification behaviour.
    classifier: str = "passthrough"
    classifier_params: dict = Field(default_factory=dict)


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

    # Defaulted so every configuration written before flows existed still loads
    # and still behaves exactly as it did.
    flow: FlowConfig = Field(default_factory=FlowConfig)
