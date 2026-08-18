"""Line-based grid reconstruction helpers for `grid_line_extractor`.

Three things used to live here and no longer do:

* `estimate_skew_angle` / `deskew_image` -- a second copy of deskew, which the
  extractor ran on an image the pipeline had ALREADY deskewed. That second
  rotation put every table bbox in a different coordinate space from every OCR
  bbox. Deskew now happens once, upstream, in
  `preprocessing/steps/geometric/deskew.py`.
* `build_line_mask` -- also duplicated in `temp/table_det.py`, now shared from
  `morphology.py`.
* `detect_table_grid` and its `__main__` block -- a standalone CLI that
  re-implemented the extractor and wrote debug PNGs into the *parent* directory
  as a side effect of being called. It is `tools/render_layout.py` now.
"""

import cv2
import numpy as np


def extract_lines_from_mask(mask, is_horizontal, min_length=40, merge_gap=10):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if is_horizontal:
            if w >= min_length:
                raw_boxes.append((x, y + h // 2, x + w, y + h // 2))
        else:
            if h >= min_length:
                raw_boxes.append((x + w // 2, y, x + w // 2, y + h))

    if not raw_boxes:
        return []

    key_idx = 1 if is_horizontal else 0
    raw_boxes.sort(key=lambda b: b[key_idx])

    merged = []
    group = [raw_boxes[0]]
    for box in raw_boxes[1:]:
        prev_pos = group[-1][key_idx]
        curr_pos = box[key_idx]
        if abs(curr_pos - prev_pos) <= merge_gap:
            group.append(box)
        else:
            merged.append(_flatten_group(group, is_horizontal))
            group = [box]
    merged.append(_flatten_group(group, is_horizontal))
    return merged


def _flatten_group(group, is_horizontal):
    arr = np.array(group)
    if is_horizontal:
        y = int(np.mean(arr[:, 1]))
        x1, x2 = int(np.min(arr[:, 0])), int(np.max(arr[:, 2]))
        return [x1, y, x2, y]
    else:
        x = int(np.mean(arr[:, 0]))
        y1, y2 = int(np.min(arr[:, 1])), int(np.max(arr[:, 3]))
        return [x, y1, x, y2]


def merge_close_lines(lines, is_horizontal, proximity_thresh=15):
    """
    Final cleanup step: collapses multiple nearby parallel lines
    (duplicates of the same physical table line, or slightly split
    detections) into a single line. Groups lines whose position
    (average y for horizontal, average x for vertical) is within
    `proximity_thresh` px of each other, then replaces each group with
    one line spanning the group's full extent.
    """
    if not lines:
        return []

    key_idx = 1 if is_horizontal else 0  # y for horizontal, x for vertical
    lines_sorted = sorted(lines, key=lambda l: (l[key_idx] + l[key_idx + 2]) / 2
                           if is_horizontal else (l[0] + l[2]) / 2)

    def pos(line):
        return (line[1] + line[3]) / 2 if is_horizontal else (line[0] + line[2]) / 2

    merged = []
    group = [lines_sorted[0]]
    for line in lines_sorted[1:]:
        if abs(pos(line) - pos(group[-1])) <= proximity_thresh:
            group.append(line)
        else:
            merged.append(_flatten_group(
                [(l[0], l[1], l[2], l[3]) for l in group], is_horizontal
            ))
            group = [line]
    merged.append(_flatten_group(
        [(l[0], l[1], l[2], l[3]) for l in group], is_horizontal
    ))
    return merged


def _segment_has_line(mask, is_vertical_separator, pos, range_start, range_end,
                       coverage_ratio=0.5):
    """
    Checks whether a real separator line actually exists at `pos`
    (an x-coordinate for a vertical separator, or a y-coordinate for a
    horizontal separator) across the span [range_start, range_end].

    This is what lets us tell the difference between "grid position
    where a line should be" and "grid position where a line ACTUALLY
    is" — the gap is exactly where spanning/merged cells happen (e.g. a
    totals row with no vertical divider drawn across it).

    coverage_ratio: fraction of the span that must contain ink for us
    to trust that a real line is there (0.5 = at least half the gap is
    covered — tolerant of small gaps/anti-aliasing, but rejects a
    mostly-empty gap).
    """
    h, w = mask.shape[:2]
    span = max(1, range_end - range_start)
    band = 3  # pixels of tolerance around the expected line position

    if is_vertical_separator:
        x1, x2 = max(0, pos - band), min(w, pos + band + 1)
        strip = mask[range_start:range_end, x1:x2]
    else:
        y1, y2 = max(0, pos - band), min(h, pos + band + 1)
        strip = mask[y1:y2, range_start:range_end]

    if strip.size == 0:
        return False

    # For each position along the span, does at least one pixel in the
    # perpendicular band contain ink?
    covered = np.any(strip > 0, axis=(1 if is_vertical_separator else 0))
    coverage = np.sum(covered) / span
    return coverage >= coverage_ratio


def extend_lines_to_table_bounds(h_lines, v_lines, min_extend_length_ratio=0.5):
    """
    Snaps line endpoints to the table's outer bounding box so partial
    lines (e.g. a line that only reaches 3/4 of the way across) become
    full-width/full-height — BUT only for lines that are already long
    enough to be trusted as real grid lines in the first place.

    Without the length check, short noise segments that slipped past
    the intersection filter (e.g. two short stray marks that happened
    to cross each other) would get stretched into full fake lines,
    corrupting the grid. min_extend_length_ratio=0.5 means a horizontal
    line must already cover at least 50% of the table width (and a
    vertical line 50% of the table height) to qualify; anything shorter
    is dropped instead of stretched.
    """
    if not h_lines or not v_lines:
        return h_lines, v_lines

    x_min = min(min(l[0], l[2]) for l in h_lines + v_lines)
    x_max = max(max(l[0], l[2]) for l in h_lines + v_lines)
    y_min = min(min(l[1], l[3]) for l in h_lines + v_lines)
    y_max = max(max(l[1], l[3]) for l in h_lines + v_lines)

    table_width = max(1, x_max - x_min)
    table_height = max(1, y_max - y_min)

    extended_h = []
    for l in h_lines:
        length = abs(l[2] - l[0])
        if length >= table_width * min_extend_length_ratio:
            extended_h.append([x_min, l[1], x_max, l[3]])
        # else: too short to trust — dropped, not stretched

    extended_v = []
    for l in v_lines:
        length = abs(l[3] - l[1])
        if length >= table_height * min_extend_length_ratio:
            extended_v.append([l[0], y_min, l[2], y_max])
        # else: too short to trust — dropped, not stretched

    return extended_h, extended_v


def extract_cells_with_spans(h_lines, v_lines, vertical_mask, horizontal_mask,
                              min_cell_size=10, coverage_ratio=0.5):
    """
    Same grid-based approach as extract_cells, but before treating two
    neighboring grid positions as separate cells, it verifies a real
    line actually exists between them in the mask. If not, the cells
    are merged into one spanning cell (colspan/rowspan tracked).

    Returns a flat list of dicts instead of a strict 2D grid, since
    spanning cells break the simple row/col rectangle shape:
        {"x1","y1","x2","y2","row","col","rowspan","colspan"}
    """
    if len(h_lines) < 2 or len(v_lines) < 2:
        print("Not enough lines to form cells (need >=2 horizontal and >=2 vertical).")
        return []

    y_positions = sorted(set(int((l[1] + l[3]) / 2) for l in h_lines))
    x_positions = sorted(set(int((l[0] + l[2]) / 2) for l in v_lines))

    n_rows = len(y_positions) - 1
    n_cols = len(x_positions) - 1

    # Step 1: for each row band, merge horizontally across missing
    # vertical separators -> variable-width cells per row
    row_cells = []  # row_cells[r] = list of (col_start, col_end, x1, y1, x2, y2)
    for r in range(n_rows):
        y1, y2 = y_positions[r], y_positions[r + 1]
        if (y2 - y1) < min_cell_size:
            row_cells.append([])
            continue

        cells_in_row = []
        col_start = 0
        for c in range(1, n_cols + 1):
            is_last = (c == n_cols)
            has_sep = False
            if not is_last:
                has_sep = _segment_has_line(
                    vertical_mask, True, x_positions[c], y1, y2, coverage_ratio
                )
            if has_sep or is_last:
                x1, x2 = x_positions[col_start], x_positions[c]
                if (x2 - x1) >= min_cell_size:
                    cells_in_row.append([col_start, c - 1, x1, y1, x2, y2])
                col_start = c
        row_cells.append(cells_in_row)

    # Step 2: merge vertically across missing horizontal separators.
    # Two cells in consecutive rows merge if they cover the same column
    # range AND there's no real horizontal line between them.
    merged = []
    consumed = [[False] * len(row_cells[r]) for r in range(n_rows)]

    for r in range(n_rows):
        for i, (c_start, c_end, x1, y1, x2, y2) in enumerate(row_cells[r]):
            if consumed[r][i]:
                continue
            final_y2 = y2
            final_row_end = r
            rr = r
            while rr + 1 < n_rows:
                next_row = row_cells[rr + 1]
                match_idx = next((j for j, cell in enumerate(next_row)
                                   if cell[0] == c_start and cell[1] == c_end),
                                  None)
                if match_idx is None:
                    break
                sep_y = y_positions[rr + 1]
                has_sep = _segment_has_line(
                    horizontal_mask, False, sep_y, x1, x2, coverage_ratio
                )
                if has_sep:
                    break
                # No real separator -> merge this row into the span
                consumed[rr + 1][match_idx] = True
                final_y2 = next_row[match_idx][5]
                final_row_end = rr + 1
                rr += 1

            merged.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": final_y2,
                "row": r, "col": c_start,
                "rowspan": final_row_end - r + 1,
                "colspan": c_end - c_start + 1,
            })

    return merged


def filter_by_intersection(h_lines, v_lines, tolerance=15):
    """
    Keeps only lines that actually intersect at least one line of the
    opposite orientation (within `tolerance` px). Real table grid lines
    always meet other grid lines; stray detections (page edges, shadows,
    leftover text blobs) usually don't line up with anything.
    """
    def intersects(h, v, tol):
        hx1, hy, hx2, _ = h
        vx, vy1, _, vy2 = v
        # v's x must fall within h's x-range (with tolerance)
        x_ok = (min(hx1, hx2) - tol) <= vx <= (max(hx1, hx2) + tol)
        # h's y must fall within v's y-range (with tolerance)
        y_ok = (min(vy1, vy2) - tol) <= hy <= (max(vy1, vy2) + tol)
        return x_ok and y_ok

    kept_h = [h for h in h_lines if any(intersects(h, v, tolerance) for v in v_lines)]
    kept_v = [v for v in v_lines if any(intersects(h, v, tolerance) for h in h_lines)]
    return kept_h, kept_v
