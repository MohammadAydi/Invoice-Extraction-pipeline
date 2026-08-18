"""Morphological line reconstruction, shared by every ruled-table extractor.

This used to exist twice -- once in `grid_utils.py`, once in `temp/table_det.py`
-- with the copies slowly diverging (only one of them could turn dot-bridging
off, only one could hand back the intermediate). Both extractors now import
this.

Everything here expects **ink at 255** on a black background, which is what the
table branch's `adaptive_threshold` with `invert: true` produces. Erosion and
dilation act on the foreground; fed an ink-at-0 image they would reconstruct the
paper instead of the rules.
"""

from __future__ import annotations

import cv2
import numpy as np


def build_line_mask(
    binary: np.ndarray,
    is_horizontal: bool,
    w: int,
    h: int,
    dot_bridge_scale: int = 150,
    main_kernel_scale: int = 30,
    use_dot_bridging: bool = True,
    return_intermediate: bool = False,
):
    """Isolate the rules running in one direction.

    Two stages, so this works for solid rules AND for the dotted/dashed ones
    these invoice forms are printed with:

    Stage 1 (dot bridging): a SHORT dilation along the line's direction merges
    nearby dots into one continuous stroke. Without it, stage 2's erosion simply
    erases isolated dots -- each is smaller than the main kernel.

    Stage 2 (reconstruction): erode then dilate with a long kernel, which now
    works because the input is continuous. Already-solid rules survive both.

    `dot_bridge_scale` and `main_kernel_scale` are divisors of the image
    dimension, not pixel counts, so the same config works at any scan
    resolution: smaller means a longer kernel, i.e. bridges bigger gaps /
    demands longer runs.
    """
    if use_dot_bridging:
        if is_horizontal:
            bridge_len = max(3, w // dot_bridge_scale)
            bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_len, 1))
        else:
            bridge_len = max(3, h // dot_bridge_scale)
            bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, bridge_len))
        bridged = cv2.dilate(binary, bridge_kernel, iterations=1)
    else:
        bridged = binary

    if is_horizontal:
        main_len = max(10, w // main_kernel_scale)
        main_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (main_len, 1))
    else:
        main_len = max(10, h // main_kernel_scale)
        main_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, main_len))

    eroded = cv2.erode(bridged, main_kernel, iterations=1)
    result = cv2.dilate(eroded, main_kernel, iterations=1)

    if return_intermediate:
        return result, bridged
    return result
