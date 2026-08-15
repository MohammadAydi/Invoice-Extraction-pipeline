from __future__ import annotations


class PipelineError(Exception):
    """Base class for all pipeline-related errors."""


class ConfigurationError(PipelineError):
    """Raised when config is invalid or references an unknown component name."""


class StepExecutionError(PipelineError):
    """Raised when a preprocessing step fails during `apply`."""

    def __init__(self, step_name: str, original: Exception):
        super().__init__(f"Step '{step_name}' failed: {original}")
        self.step_name = step_name
        self.original = original


class NotImplementedStrategyError(PipelineError, NotImplementedError):
    """Raised by a strategy variant that was deliberately left unbuilt
    (e.g. an alternative algorithm not yet implemented), as opposed to a
    step that simply hasn't been started yet.

    Inherits from NotImplementedError too so PreprocessingPipeline.run()
    still lets it propagate unwrapped (its existing `except
    NotImplementedError: raise` special case applies here for free),
    while also being identifiable/catchable as "intentionally unbuilt"
    via the PipelineError hierarchy.
    """

    def __init__(self, step_name: str, strategy_name: str, message: str = ""):
        text = f"Step '{step_name}' strategy '{strategy_name}' is not implemented."
        if message:
            text += f" {message}"
        super().__init__(text)
        self.step_name = step_name
        self.strategy_name = strategy_name
