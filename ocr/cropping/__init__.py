"""Importing this package self-registers every region cropper.

Nothing here needs a model, so unlike the detectors and recognizers these can be
imported unconditionally.
"""

from ocr.cropping import padded_cropper  # noqa: F401
