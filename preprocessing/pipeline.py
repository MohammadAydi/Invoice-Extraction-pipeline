from __future__ import annotations

from core.domain.image_payload import ImagePayload, PipelineContext
from core.exceptions import StepExecutionError
from core.interfaces.preprocessing import PreprocessingStep


class PreprocessingPipeline:
    """Executes an ordered list of already-instantiated steps against one
    image. Three instances exist per run -- geometric, ocr_photometric,
    table_photometric -- sequenced by PipelineOrchestrator.
    """

    def __init__(self, steps: list[PreprocessingStep]):
        self._steps = steps

    def run(self, initial_payload: ImagePayload) -> ImagePayload:
        ctx = PipelineContext(payload=initial_payload)
        for step in self._steps:
            try:
                ctx = step.apply(ctx)
            except NotImplementedError:
                raise
            except Exception as exc:  # noqa: BLE001 - wrap for a uniform pipeline-level error
                raise StepExecutionError(step.name, exc) from exc
            ctx.payload.applied_steps.append(step.name)
        return ctx.payload

    @property
    def step_names(self) -> list[str]:
        return [s.name for s in self._steps]
