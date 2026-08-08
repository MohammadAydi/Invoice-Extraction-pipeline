"""Turns a list of StepConfig into a ready-to-run PreprocessingPipeline.

This is where "which steps are enabled", "execution order", and "per-step
configuration" -- the three things the project brief asked to be
configurable in one place -- actually take effect. Steps are instantiated
from `step_registry` purely by name; this module never changes when a new
step type is added elsewhere in the codebase.
"""

from __future__ import annotations

import preprocessing.steps  # noqa: F401  (import triggers step self-registration)
from config.schema import StepConfig
from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.steps.registry import step_registry


class PreprocessingPipelineBuilder:
    @staticmethod
    def build(step_configs: list[StepConfig]) -> PreprocessingPipeline:
        steps = [
            step_registry.create(cfg.name, **cfg.params)
            for cfg in step_configs
            if cfg.enabled
        ]
        return PreprocessingPipeline(steps)
