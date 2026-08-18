"""The wire shape of pipeline configuration, per `docs/settings-config-contract.md`.

Two shapes describe the same pipeline and this module is the bridge between
them:

* :class:`~config.schema.AppConfig` -- the *internal* shape. Ordered lists of
  steps, where list position is execution order. It is what the builder and the
  orchestrator consume, and what the YAML files on disk are written in.
* :class:`PipelineSettings` -- the *wire* shape. Each preprocessing step is a
  fixed, named key; the UI controls only whether a step is enabled, which
  algorithm implements it, and that algorithm's params. Execution order is
  fixed by the backend and is deliberately **not** derived from key order,
  because a JSON object has no meaningful order to rely on.

The settings page in the desktop app is built against the wire shape, so this
is where a request's configuration becomes something the pipeline can run.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from config.schema import (
    AppConfig,
    FlowConfig,
    OCRConfig,
    OutputConfig,
    PersistenceConfig,
    PreprocessingConfig,
    StepConfig,
    StringMatchingConfig,
    TableExtractionConfig,
)

# Execution order, fixed by the backend. Phase 1 produces the single corrected
# image; phases 2 and 3 together are the OCR-photometric branch, in this order.
GEOMETRIC_STEP_KEYS: tuple[str, ...] = ("perspective_correction", "deskew")

PHOTOMETRIC_STEP_KEYS: tuple[str, ...] = (
    "channel_selection",
    "illumination_normalization",
    "contrast_enhancement",
    "denoising",
    "thresholding",
    "morphological_cleanup",
)

STEP_KEYS: tuple[str, ...] = GEOMETRIC_STEP_KEYS + PHOTOMETRIC_STEP_KEYS

# Which registered algorithms are legal under each step key. Validated here so a
# UI bug that sends `clahe` as the thresholding algorithm is rejected with a
# precise message at the API boundary, rather than building a pipeline whose
# steps are in nonsensical positions.
ALGORITHMS_BY_STEP: dict[str, tuple[str, ...]] = {
    "perspective_correction": ("perspective_correction",),
    "deskew": ("deskew_hough", "deskew_min_area_rect"),
    "channel_selection": ("channel_selection",),
    "illumination_normalization": (
        "illumination_normalization_blur_divide",
        "illumination_normalization_blackhat",
    ),
    "contrast_enhancement": ("clahe", "plain_equalization"),
    "denoising": ("median_blur", "gaussian_blur", "nlm_denoise", "bilateral_filter"),
    "thresholding": ("fixed_threshold", "otsu_threshold", "adaptive_threshold"),
    "morphological_cleanup": ("morphological_cleanup",),
}

# Registered but deliberately unbuilt. The service refuses them at validation
# time rather than letting the run reach NotImplementedStrategyError halfway
# through: the desktop app should not offer these, and if it does, the user gets
# a clear message instead of a failed extraction.
UNIMPLEMENTED_ALGORITHMS: frozenset[str] = frozenset(
    {"deskew_min_area_rect", "illumination_normalization_blackhat"}
)


class StepSettings(BaseModel):
    """One named preprocessing step: on/off, which algorithm, and its params."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    algorithm: str
    params: dict = Field(default_factory=dict)


class PreprocessingSettings(BaseModel):
    """All eight steps. Every key is required -- disabling a step means
    `enabled: false`, never omitting it, so the configuration always describes
    the whole pipeline rather than a subset the reader has to guess at.
    """

    model_config = ConfigDict(extra="ignore")

    perspective_correction: StepSettings
    deskew: StepSettings
    channel_selection: StepSettings
    illumination_normalization: StepSettings
    contrast_enhancement: StepSettings
    denoising: StepSettings
    thresholding: StepSettings
    morphological_cleanup: StepSettings

    @model_validator(mode="after")
    def _check_algorithms(self) -> "PreprocessingSettings":
        for key in STEP_KEYS:
            step: StepSettings = getattr(self, key)
            allowed = ALGORITHMS_BY_STEP[key]

            if step.algorithm not in allowed:
                raise ValueError(
                    f"'{step.algorithm}' is not a valid algorithm for step '{key}'. "
                    f"Expected one of: {', '.join(allowed)}."
                )

            # An unimplemented strategy that is switched off is harmless; it is
            # only selecting one to actually run that has to be refused.
            if step.enabled and step.algorithm in UNIMPLEMENTED_ALGORITHMS:
                raise ValueError(
                    f"Step '{key}' algorithm '{step.algorithm}' is not implemented yet. "
                    "Disable the step or pick another algorithm."
                )

        return self

    def step(self, key: str) -> StepSettings:
        return getattr(self, key)


class PipelineSettings(BaseModel):
    """The whole `configuration` object sent with an extract request."""

    model_config = ConfigDict(extra="ignore")

    preprocessing: PreprocessingSettings
    ocr: OCRConfig
    table_extraction: TableExtractionConfig
    string_matching: StringMatchingConfig
    output: OutputConfig = Field(default_factory=lambda: OutputConfig(formatter="ui_overlay_json"))
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)

    # Which reading strategy runs. Defaulted, so a client written against the
    # previous contract still sends a valid configuration and still gets the
    # single-engine behaviour it was written for.
    flow: FlowConfig = Field(default_factory=FlowConfig)

    def to_app_config(self, table_photometric_steps: list[StepConfig] | None = None) -> AppConfig:
        """Translate to the internal ordered-list shape the pipeline runs.

        `table_photometric_steps` is intentionally not part of this contract --
        the table branch's preprocessing is not exposed in the settings UI -- so
        the caller passes the server-side default through unchanged.
        """
        return AppConfig(
            preprocessing=PreprocessingConfig(
                geometric_steps=[self._to_step(key) for key in GEOMETRIC_STEP_KEYS],
                ocr_photometric_steps=[self._to_step(key) for key in PHOTOMETRIC_STEP_KEYS],
                table_photometric_steps=list(table_photometric_steps or []),
            ),
            ocr=self.ocr,
            table_extraction=self.table_extraction,
            string_matching=self.string_matching,
            output=self.output,
            persistence=self.persistence,
            flow=self.flow,
        )

    def _to_step(self, key: str) -> StepConfig:
        step = self.preprocessing.step(key)
        # The algorithm name IS the registry name for every step in this
        # contract, which is why the registry needs no per-step lookup table.
        return StepConfig(name=step.algorithm, enabled=step.enabled, params=step.params)

    @classmethod
    def from_app_config(cls, config: AppConfig) -> "PipelineSettings":
        """The inverse, so a YAML file on disk can answer "what would the UI show?".

        Used to serve the service's own defaults to the settings page, which is
        what stops the desktop app from having to hardcode a second copy of
        them and drift.
        """
        by_name: dict[str, StepConfig] = {}
        for step in (
            *config.preprocessing.geometric_steps,
            *config.preprocessing.ocr_photometric_steps,
        ):
            by_name[step.name] = step

        preprocessing: dict[str, StepSettings] = {}
        for key in STEP_KEYS:
            allowed = ALGORITHMS_BY_STEP[key]
            found = next((by_name[name] for name in allowed if name in by_name), None)

            preprocessing[key] = (
                StepSettings(
                    enabled=found.enabled, algorithm=found.name, params=found.params
                )
                if found is not None
                # A step the YAML never mentions is off, shown with the first
                # implemented algorithm pre-selected so the UI has something to
                # render its param form from.
                else StepSettings(
                    enabled=False,
                    algorithm=next(
                        a for a in allowed if a not in UNIMPLEMENTED_ALGORITHMS
                    ),
                    params={},
                )
            )

        return cls(
            preprocessing=PreprocessingSettings(**preprocessing),
            ocr=config.ocr,
            table_extraction=config.table_extraction,
            string_matching=config.string_matching,
            output=config.output,
            persistence=config.persistence,
            flow=config.flow,
        )
