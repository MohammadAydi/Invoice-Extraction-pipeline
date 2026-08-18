from __future__ import annotations

from typing import Protocol, Sequence

from core.domain.image_payload import ImagePayload
from core.domain.ocr import RegionCrop, TextRegion


class RegionCropper(Protocol):
    """Cuts each region out of the page and prepares it for a recognizer.

    Its own stage because the preparation is a real, tunable decision -- how
    much padding, how much upscaling, what minimum size is worth reading -- that
    belongs to neither the detector that found the box nor the model that reads
    it, and that both flows share unchanged.
    """

    def crop(
        self, image: ImagePayload, regions: Sequence[TextRegion]
    ) -> list[RegionCrop]: ...
