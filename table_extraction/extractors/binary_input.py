"""The input contract every ruled-table extractor shares.

Extractors used to accept whatever arrived and quietly fix it up -- converting
to grayscale, then running `adaptiveThreshold` on an image the table branch had
*already* thresholded. Thresholding a binary image is not a no-op: it produces a
plausible-looking mask with the polarity and the stroke widths subtly wrong, and
nothing downstream can tell.

Worse, the same methods also re-ran deskew, which rotated the page a second time
and put every cell bbox in a different coordinate space from every OCR bbox --
silently breaking the invariant the whole pipeline is built on.

So extractors no longer preprocess. They state what they need and say so
precisely when they do not get it.
"""

from __future__ import annotations

import numpy as np

from core.domain.image_payload import ImagePayload

_REQUIRED_STEPS = (
    "    table_photometric_steps:\n"
    "      - {name: channel_selection,  enabled: true, params: {channel: gray}}\n"
    "      - {name: adaptive_threshold, enabled: true, params: {block_size: 15, c: 10, invert: true}}"
)


def require_binary_ink_white(payload: ImagePayload, extractor_name: str) -> np.ndarray:
    """Return the payload's image, having checked it is a binary mask with ink
    at 255 -- the polarity morphological line reconstruction needs.

    Raises ValueError naming the config that produces it, because the fix is
    always a config change and never a code change.
    """
    image = payload.image
    if image is None or image.size == 0:
        raise ValueError(f"{extractor_name} received an empty image.")

    if image.ndim != 2:
        raise ValueError(
            f"{extractor_name} expects a single-channel binary image, got "
            f"{image.ndim} channels. Configure the table branch:\n{_REQUIRED_STEPS}"
        )

    values = np.unique(image)
    if values.size > 2 or not set(values.tolist()) <= {0, 255}:
        raise ValueError(
            f"{extractor_name} expects a binary (0/255) image, got {values.size} "
            f"distinct values. Configure the table branch:\n{_REQUIRED_STEPS}"
        )

    # `binary_ink_is_white` is set by whichever thresholding step ran. Absent
    # means no thresholding step ran at all, which the ndim/values checks above
    # would normally have caught -- unless the source image happens to be binary
    # already, in which case the polarity is genuinely unknown and guessing is
    # exactly what got us here.
    if not payload.metadata.get("binary_ink_is_white"):
        raise ValueError(
            f"{extractor_name} needs ink at 255 (white ink on black), but the "
            "table branch did not produce it. Morphological line reconstruction "
            "erodes the foreground; on an ink-at-0 image it reconstructs the "
            f"paper instead of the rules. Set `invert: true`:\n{_REQUIRED_STEPS}"
        )

    return image
