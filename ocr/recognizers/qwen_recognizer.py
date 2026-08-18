"""Qwen as a TextRecognizer.

A thin adapter over :class:`~ocr.engines.qwen_vlm_engine.QwenVLMEngine`, not a
reimplementation. That class owns three real backends -- a subprocess against a
second virtualenv, a local HTTP server, and in-process transformers -- and all
the awkward parts (waiting for the worker's READY sentinel, skipping stray
warnings on stdout, UTF-8 on Windows). None of that is worth having twice.

What this adds is the seam: it takes a
:class:`~core.domain.ocr.RegionCrop` and turns its
:class:`~core.domain.roles.ContentKind` into the right prompt. The recognizer
therefore knows about *kinds of characters* and nothing about invoices, which is
what lets the same object serve both flows -- and a non-invoice form later.
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

from PIL import Image

from core.domain.ocr import RegionCrop
from core.domain.roles import ContentKind
from ocr.engines.qwen_vlm_engine import PROMPTS, QwenVLMEngine, to_pil
from ocr.recognizers.registry import recognizer_registry

logger = logging.getLogger(__name__)

# The model is told to answer with this when a cell is blank or illegible, so it
# does not invent a plausible value for an empty box.
_EMPTY_SENTINEL = "EMPTY"


@recognizer_registry.register("qwen")
class QwenRecognizer:
    name = "qwen"

    def __init__(self, default_kind: str = ContentKind.ANY.value, **engine_params):
        self.default_kind = ContentKind(default_kind)
        self._engine = QwenVLMEngine(**engine_params)

        missing = [k.value for k in ContentKind if k.value not in PROMPTS]
        if missing:
            raise ValueError(
                f"qwen recognizer: no prompt defined for content kind(s) {missing}. "
                "Add them to PROMPTS in ocr/engines/qwen_vlm_engine.py."
            )

    @property
    def engine(self) -> QwenVLMEngine:
        return self._engine

    def is_ready(self) -> bool:
        probe = getattr(self._engine, "is_ready", None)
        return bool(probe()) if callable(probe) else True

    def warmup(self) -> None:
        warm = getattr(self._engine, "warmup", None)
        if callable(warm):
            warm()

    def read(self, crop: RegionCrop) -> str:
        kind = crop.content_kind or self.default_kind
        image = crop.image
        pil = image if isinstance(image, Image.Image) else to_pil(image)

        started = time.perf_counter()
        try:
            text = self._engine.read(pil, kind=kind.value)
        except Exception as exc:  # noqa: BLE001
            # One unreadable field must not abort a whole invoice: the other
            # thirty cells on the page are still worth having, and the empty one
            # comes back flagged for review rather than silently guessed.
            logger.warning(
                "qwen failed to read region at %s: %s", crop.region.bbox.as_tuple(), exc
            )
            return ""

        elapsed = time.perf_counter() - started
        logger.debug(
            "qwen read %s (%s) in %.3fs -> %r",
            crop.region.bbox.as_tuple(),
            kind.value,
            elapsed,
            text,
        )

        return "" if text.strip().upper() == _EMPTY_SENTINEL else text

    def read_all(self, crops: Sequence[RegionCrop]) -> list[str]:
        # The subprocess backend is strictly one crop per round trip, so there
        # is nothing to batch. Kept explicit so a future batching backend has an
        # obvious place to land.
        return [self.read(crop) for crop in crops]
