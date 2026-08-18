from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.domain.geometry import BoundingBox
from core.domain.roles import ContentKind


@dataclass(frozen=True)
class TextRegion:
    """Somewhere on the page that text is expected, before anything read it.

    Produced two different ways, which is the whole reason it exists as a type
    of its own: a text *detector* finds these by looking at ink, and the
    *layout* stage derives them from the cells of a printed grid. Downstream --
    cropping, recognition -- cannot tell the difference and does not need to.

    `content_kind` travels with the region because it is what selects the
    recognition prompt, and only the producer knows it: a detector that found a
    blob of ink knows nothing, while a layout region that is the quantity
    column knows it holds digits.

    `source_id` is the id of the LayoutRegion this came from, when it came from
    one. It is what lets mapping assign the result back by identity instead of
    re-deriving the assignment from geometry it already knew.
    """

    bbox: BoundingBox
    confidence: float | None = None
    content_kind: ContentKind = ContentKind.ANY
    source_id: str | None = None


@dataclass(frozen=True)
class DetectionResult:
    regions: list[TextRegion]
    detector_name: str
    raw_detector_output: object = None


@dataclass(frozen=True)
class RegionCrop:
    """One region's pixels, cut out and prepared for a recognizer.

    `image` is padded and possibly upscaled, so its size does not match
    `region.bbox`. That is intentional and harmless: nothing downstream measures
    anything off the crop, because the coordinates that matter are the ones
    already recorded on `region`.
    """

    region: TextRegion
    image: np.ndarray

    @property
    def content_kind(self) -> ContentKind:
        return self.region.content_kind


@dataclass(frozen=True)
class OCRFragment:
    """One piece of text that was read, and where it was read from.

    `confidence` is optional because it is genuinely absent for some engines: a
    detector-only pipeline paired with a VLM recognizer has a *detection* score
    and no recognition score at all, and inventing one would be worse than
    admitting it. Consumers treat None as "unknown", not as zero.
    """

    text: str
    bbox: BoundingBox
    confidence: float | None = None
    source_id: str | None = None


@dataclass(frozen=True)
class OCRResult:
    fragments: list[OCRFragment]
    engine_name: str
    raw_engine_output: object = None  # escape hatch for adapter-specific debug data
    timings: dict = field(default_factory=dict)
