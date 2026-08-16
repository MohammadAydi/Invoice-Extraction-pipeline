"""BGR -> single channel. `gray` uses cv2.COLOR_BGR2GRAY; `b`/`g`/`r`
select the raw color channel. Not one of the four preprocessing scripts,
but a required prerequisite: every downstream photometric step assumes a
single-channel image.
"""

from __future__ import annotations

import cv2

from core.domain.image_payload import PipelineContext
from preprocessing.steps.registry import step_registry

_CHANNEL_INDEX = {"b": 0, "g": 1, "r": 2}


@step_registry.register("channel_selection")
class ChannelSelectionStep:
    name = "channel_selection"

    def __init__(self, channel: str = "gray", **params):
        self.channel = channel
        self.params = params

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        image = ctx.payload.image
        if image is None or image.size == 0:
            raise ValueError("channel_selection received an empty image.")

        if self.channel == "gray":
            out = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        elif self.channel in _CHANNEL_INDEX:
            if image.ndim != 3:
                raise ValueError(
                    f"channel_selection channel='{self.channel}' requires a "
                    "3-channel color image."
                )
            out = image[:, :, _CHANNEL_INDEX[self.channel]]
        else:
            raise ValueError(
                f"Unknown channel '{self.channel}'; expected one of "
                "gray, b, g, r."
            )

        ctx.payload.image = out
        return ctx
