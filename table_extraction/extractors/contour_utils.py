"""Cell detection by contour, ported from `temp/table_det.py`.

The other extractor (`grid_line`) reconstructs the grid from *lines*: find every
rule, merge, extend, then intersect them into cells. That works when the rules
are clean. This one goes the other way -- it takes the closed regions the rules
enclose directly, via `findContours` -- which turns out to be more robust on
these forms for two reasons:

1. It does not need a rule to be complete. A cell whose border is broken in one
   place is still found as long as the contour closes.
2. It can reject a *region* on shape grounds (too small, too flat, too wide, too
   large a share of the page), which is a much sharper filter than rejecting a
   *line* on length grounds.

The stack filter (`remove_stacked_text_blobs`) is the piece with no equivalent
on the line-based side, and it is the one that matters most on handwritten
invoices: several lines of handwriting stacked above each other, of similar
width and overlapping in x, look exactly like a set of horizontal rules after
morphological reconstruction. Real rules are distinguishable because they
*intersect a vertical rule*; a stack of writing does not.
"""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np


def _blob_intersects_vertical_mask(vertical_mask, x1, x2, y_center, band: int = 4) -> bool:
    """Whether any vertical rule crosses this blob's horizontal extent."""
    h, w = vertical_mask.shape[:2]
    y0, y1 = max(0, int(y_center - band)), min(h, int(y_center + band + 1))
    x1, x2 = max(0, int(x1)), min(w, int(x2))
    if x2 <= x1 or y1 <= y0:
        return False
    return bool(np.any(vertical_mask[y0:y1, x1:x2] > 0))


def remove_stacked_text_blobs(
    mask,
    is_horizontal: bool,
    y_proximity: int = 25,
    length_similarity: float = 0.35,
    min_group_size: int = 2,
    overlap_ratio: float = 0.5,
    vertical_mask=None,
    intersection_band: int = 4,
):
    """Drop horizontal blobs that are stacked handwriting, not table rules.

    Returns `(clean_mask, removed_mask)`. The removed half is returned rather
    than discarded because it is the only way to see, on a debug render, why a
    rule went missing.

    A cluster is judged to be text when its members are of similar length AND
    overlap each other in x -- the signature of consecutive lines of writing. A
    member that a *vertical* rule crosses is kept regardless: that is positive
    evidence of a real grid, and it is what stops a genuinely short table rule
    inside a dense block from being thrown away.

    Vertical masks are passed through untouched. Vertical strokes of handwriting
    are far shorter than a column rule and are already removed by the length
    filter in reconstruction.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    removed_mask = np.zeros_like(mask)

    if not contours:
        return clean_mask, removed_mask

    if not is_horizontal:
        for c in contours:
            cv2.drawContours(clean_mask, [c], -1, 255, thickness=cv2.FILLED)
        return clean_mask, removed_mask

    blobs = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        blobs.append(
            {"contour": c, "x1": x, "x2": x + w, "y_center": y + h / 2, "length": w}
        )

    blobs.sort(key=lambda b: b["y_center"])

    clusters: list[list[dict]] = []
    current = [blobs[0]]
    for b in blobs[1:]:
        if abs(b["y_center"] - current[-1]["y_center"]) <= y_proximity:
            current.append(b)
        else:
            clusters.append(current)
            current = [b]
    clusters.append(current)

    def x_overlaps(a, b) -> bool:
        overlap = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
        shorter = min(a["x2"] - a["x1"], b["x2"] - b["x1"]) or 1
        return (overlap / shorter) >= overlap_ratio

    for cluster in clusters:
        if len(cluster) < min_group_size:
            for b in cluster:
                cv2.drawContours(clean_mask, [b["contour"]], -1, 255, thickness=cv2.FILLED)
            continue

        median_len = np.median([b["length"] for b in cluster])

        similar_count = 0
        for i, b in enumerate(cluster):
            len_ratio = abs(b["length"] - median_len) / (median_len or 1)
            has_overlap = any(
                x_overlaps(b, other) for j, other in enumerate(cluster) if j != i
            )
            if len_ratio <= length_similarity and has_overlap:
                similar_count += 1

        cluster_is_text_pattern = similar_count >= min_group_size

        for b in cluster:
            crosses_vertical = vertical_mask is not None and _blob_intersects_vertical_mask(
                vertical_mask, b["x1"], b["x2"], b["y_center"], intersection_band
            )
            target = (
                removed_mask
                if (cluster_is_text_pattern and not crosses_vertical)
                else clean_mask
            )
            cv2.drawContours(target, [b["contour"]], -1, 255, thickness=cv2.FILLED)

    return clean_mask, removed_mask


def get_table_cell_bounds(
    grid_mask,
    min_size: int = 15,
    min_height: int = 20,
    max_area_ratio: float = 0.20,
    max_aspect_ratio: float = 0.3,
    max_width_ratio: float = 0.6,
) -> list[dict]:
    """Closed regions of the grid mask, filtered to plausible cells.

    Each filter rejects a specific, observed false positive:

    * `min_size` / `min_height` -- specks, and the sliver between two rules
      drawn a few pixels apart.
    * `max_aspect_ratio` -- a region taller than it is wide by this much is a
      column rule's own outline, not a cell.
    * `max_area_ratio` / `max_width_ratio` -- the whole table, and the page
      border, both close as contours too. Keeping them would swallow every real
      cell in the containment pass below.

    Finally, any region that *contains* another is dropped: `RETR_TREE` returns
    the table outline alongside its own cells, and a fragment inside both
    belongs to the cell.
    """
    _, grid_mask = cv2.threshold(grid_mask, 127, 255, cv2.THRESH_BINARY)

    img_h, img_w = grid_mask.shape[:2]
    max_cell_area = img_h * img_w * max_area_ratio
    max_cell_width = img_w * max_width_ratio

    contours, _ = cv2.findContours(grid_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cell_bounds: list[dict] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_size or h < min_size:
            continue
        if h < min_height:
            continue
        if h > 0 and (w / h) < max_aspect_ratio:
            continue
        if w * h > max_cell_area:
            continue
        if w > max_cell_width:
            continue
        cell_bounds.append(
            {"x": x, "y": y, "width": w, "height": h, "x_max": x + w, "y_max": y + h}
        )

    def contains(outer, inner) -> bool:
        return (
            outer["x"] <= inner["x"]
            and outer["y"] <= inner["y"]
            and outer["x_max"] >= inner["x_max"]
            and outer["y_max"] >= inner["y_max"]
            and outer is not inner
        )

    is_outer = [False] * len(cell_bounds)
    for i, a in enumerate(cell_bounds):
        for j, b in enumerate(cell_bounds):
            if i != j and contains(a, b):
                is_outer[i] = True
                break

    return [c for i, c in enumerate(cell_bounds) if not is_outer[i]]


def find_table_bounds(cells: list[dict], proximity: int = 1) -> dict | None:
    """The bounding rect of the largest group of touching cells.

    Union-find over "these two boxes touch or overlap". An invoice page holds
    several disconnected groups of ruled boxes -- the header captions, the item
    table, the totals row -- and the item table is by a wide margin the biggest.
    Finding it is what lets everything else be classified *relative* to it,
    which is far more robust than absolute positions on a photographed page.

    Returns `{"x1", "y1", "x2", "y2", "cells"}`, or None when there are no cells.
    """
    if not cells:
        return None

    n = len(cells)
    parent = list(range(n))
    rank = [0] * n

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path halving
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri == rj:
            return
        if rank[ri] < rank[rj]:
            ri, rj = rj, ri
        parent[rj] = ri
        if rank[ri] == rank[rj]:
            rank[ri] += 1

    def boxes_overlap(a, b) -> bool:
        return (
            a["x"] - proximity < b["x_max"] + proximity
            and a["x_max"] + proximity > b["x"] - proximity
            and a["y"] - proximity < b["y_max"] + proximity
            and a["y_max"] + proximity > b["y"] - proximity
        )

    for i in range(n):
        for j in range(i + 1, n):
            if boxes_overlap(cells[i], cells[j]):
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    group_cells = [cells[i] for i in max(groups.values(), key=len)]

    return {
        "x1": min(c["x"] for c in group_cells),
        "y1": min(c["y"] for c in group_cells),
        "x2": max(c["x_max"] for c in group_cells),
        "y2": max(c["y_max"] for c in group_cells),
        "cells": group_cells,
    }


def dedup_cells(cells: list[dict], tol: int = 5) -> list[dict]:
    """Drop boxes that are the same box within `tol` pixels.

    A rule drawn thick enough closes as two nested contours whose bounding rects
    differ by a pixel or two; both survive the containment filter because
    neither strictly contains the other.
    """
    seen: set[tuple[int, int, int, int]] = set()
    unique: list[dict] = []
    for c in cells:
        key = (
            round(c["x"] / tol),
            round(c["y"] / tol),
            round(c["width"] / tol),
            round(c["height"] / tol),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    return unique


def cluster_cells_into_rows(cells: list[dict], y_proximity: float | None = None):
    """Group cells into rows by vertical centre, top to bottom, each row sorted
    left to right.

    `y_proximity` defaults to a fraction of the median cell height rather than a
    pixel constant, so the same code clusters correctly on a phone photo and on
    a 600-dpi scan.

    Each cell is compared against the running *mean* of the row so far, not
    against the previous cell: a tall cell followed by a short one would
    otherwise drag the comparison point far enough to split one printed row in
    two.
    """
    if not cells:
        return []

    for c in cells:
        c["y_center"] = c["y"] + c["height"] / 2
        c["x_center"] = c["x"] + c["width"] / 2

    if y_proximity is None:
        median_h = np.median([c["height"] for c in cells])
        y_proximity = max(10.0, float(median_h) * 0.6)

    sorted_cells = sorted(cells, key=lambda c: c["y_center"])

    rows: list[list[dict]] = []
    current = [sorted_cells[0]]
    for c in sorted_cells[1:]:
        row_y = float(np.mean([r["y_center"] for r in current]))
        if abs(c["y_center"] - row_y) <= y_proximity:
            current.append(c)
        else:
            rows.append(current)
            current = [c]
    rows.append(current)

    for row in rows:
        row.sort(key=lambda c: c["x_center"])

    return rows
