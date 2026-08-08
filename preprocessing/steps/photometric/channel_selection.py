"""Stub. Reference material: rgb_split.py (split BGR channels, compare
against cv2.COLOR_BGR2GRAY -- config picks which channel/mode to keep).
"""

from __future__ import annotations

from core.domain.image_payload import PipelineContext
from preprocessing.steps.registry import step_registry


@step_registry.register("channel_selection")
class ChannelSelectionStep:
    name = "channel_selection"

    def __init__(self, channel: str = "gray", **params):
        self.channel = channel
        self.params = params

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        raise NotImplementedError("Implement channel selection (see docstring).")
