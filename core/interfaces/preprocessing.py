from __future__ import annotations

from typing import Protocol

from core.domain.image_payload import PipelineContext


class PreprocessingStep(Protocol):
    """A single, self-contained image operation.

    Concrete steps live under preprocessing/steps/{geometric,photometric}/,
    self-register by name in `preprocessing.steps.registry.step_registry`,
    and are instantiated by `PreprocessingPipelineBuilder` purely from
    config. Nothing here hardcodes which steps exist or what order they
    run in.
    """

    name: str

    def apply(self, ctx: PipelineContext) -> PipelineContext: ...
