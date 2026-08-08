"""Stub. Produces the JSON payload the UI consumes directly: a reference
to the single display image plus one entry per DocumentElement with its
bbox, raw + corrected text, confidence, and an `editable` flag. See the
approved architecture doc, section "Final output model", for the exact
shape.
"""

from __future__ import annotations

from core.domain.result import PipelineResult
from output.registry import formatter_registry


@formatter_registry.register("ui_overlay_json")
class UIOverlayFormatter:
    def __init__(self, **params):
        self.params = params

    def format(self, result: PipelineResult) -> dict:
        raise NotImplementedError("Serialize PipelineResult into the UI overlay JSON shape.")
