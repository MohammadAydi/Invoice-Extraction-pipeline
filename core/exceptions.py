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
