"""Decoding uploads and encoding the diagnostic images that go back out."""

from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import numpy as np


class CorruptFileError(Exception):
    """The bytes could not be decoded as an image."""


@dataclass
class LoadedDocument:
    image: np.ndarray
    filename: str
    page_count: int = 1
    page_used: int = 1


def load_document(data: bytes, filename: str) -> LoadedDocument:
    """Decode an uploaded image.

    Single-page only: the API contract accepts `.png`, `.jpg` and `.jpeg` and
    nothing else, so `page_count` is always 1. The field exists so adding PDF
    support later is an additive change rather than a contract break.
    """
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    if image is None or image.size == 0:
        raise CorruptFileError(
            f"'{filename}' could not be decoded as an image. It may be truncated "
            "or not actually an image file."
        )

    return LoadedDocument(image=image, filename=filename)


def encode_png_base64(image: np.ndarray, max_width: int = 1200) -> str | None:
    """Base64 PNG of `image`, downscaled to `max_width`.

    Returns None rather than raising: a diagnostic image that fails to encode
    is never worth failing a good extraction over, and the caller turns a None
    into a `DEBUG_IMAGE_UNAVAILABLE` warning on an otherwise-200 response.

    Downscaling is never upscaling -- an image already narrower than the limit
    is encoded as it is.
    """
    try:
        if image is None or image.size == 0:
            return None

        prepared = _downscale(image, max_width)

        ok, buffer = cv2.imencode(".png", prepared)
        if not ok:
            return None

        return base64.b64encode(buffer.tobytes()).decode("ascii")
    except Exception:  # noqa: BLE001 - any encoding failure is "unavailable"
        return None


def _downscale(image: np.ndarray, max_width: int) -> np.ndarray:
    height, width = image.shape[:2]
    if max_width <= 0 or width <= max_width:
        return image

    scale = max_width / width
    return cv2.resize(
        image,
        (max_width, max(1, int(round(height * scale)))),
        # INTER_AREA is the right filter for shrinking: it averages the pixels
        # that collapse together instead of point-sampling one of them, which
        # is what keeps thin Arabic strokes visible in the downscaled preview.
        interpolation=cv2.INTER_AREA,
    )
