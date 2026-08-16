from __future__ import annotations

from pathlib import Path

import cv2

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

    def run(
        self,
        initial_payload: ImagePayload,
        debug_dir: Path | None = None,
        snapshots: dict | None = None,
    ) -> ImagePayload:
        """Run every step in order against one payload.

        `snapshots`, when given, is filled with a copy of the image *before*
        each step runs, keyed by that step's name, plus `"__final__"` at the
        end. The HTTP layer uses it to answer "what did the page look like
        before binarization?" without re-running the pipeline or writing
        anything to disk -- which is what the `enhanced_image_png` debug image
        in the API contract is.
        """
        ctx = PipelineContext(payload=initial_payload)

        # Steps that can produce an extra diagnostic image read this before
        # building one. Without it, a step like thresholding would render its
        # histogram on every invoice and the pipeline would immediately discard
        # it -- which under the HTTP service is the whole cost with none of the
        # benefit.
        ctx.payload.metadata["collect_debug_images"] = debug_dir is not None

        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            self._save_debug_image(debug_dir, "00_input", ctx.payload.image)
        for idx, step in enumerate(self._steps, start=1):
            if snapshots is not None:
                snapshots[step.name] = ctx.payload.image.copy()
            try:
                ctx = step.apply(ctx)
            except NotImplementedError:
                raise
            except Exception as exc:  # noqa: BLE001 - wrap for a uniform pipeline-level error
                raise StepExecutionError(step.name, exc) from exc
            ctx.payload.applied_steps.append(step.name)
            if debug_dir is not None:
                self._save_debug_image(debug_dir, f"{idx:02d}_{step.name}", ctx.payload.image)
                for extra_name, extra_image in ctx.payload.debug_images:
                    self._save_debug_image(
                        debug_dir, f"{idx:02d}_{step.name}__{extra_name}", extra_image
                    )
                ctx.payload.debug_images = []
        if snapshots is not None:
            snapshots["__final__"] = ctx.payload.image
        return ctx.payload

    @staticmethod
    def _save_debug_image(debug_dir: Path, file_stem: str, image) -> None:
        cv2.imwrite(str(debug_dir / f"{file_stem}.png"), image)

    @property
    def step_names(self) -> list[str]:
        return [s.name for s in self._steps]
