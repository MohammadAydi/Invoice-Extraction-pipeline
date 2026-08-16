import cv2
import numpy as np


def estimate_skew_angle(gray):
    """Estimates the dominant tilt of the image using Hough line voting."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    raw_lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=15
    )
    if raw_lines is None:
        return 0.0

    angles = []
    for line in raw_lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        folded = angle % 90
        if folded > 45:
            folded -= 90
        angles.append(folded)

    return float(np.median(angles)) if angles else 0.0


def deskew_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(
        img, rot_mat, (new_w, new_h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


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


def extract_cells(h_lines, v_lines, min_cell_size=10):
    """
    Builds cell rectangles directly from the grid lines. A cell is
    simply the rectangle between two consecutive horizontal lines
    (top/bottom) and two consecutive vertical lines (left/right) — no
    span-merging logic, every grid position is its own cell.

    Returns a flat list of dicts (matches the shape the rest of the
    pipeline / your text-mapping step expects):
        {"x1","y1","x2","y2","row","col","rowspan":1,"colspan":1}
    """
    if len(h_lines) < 2 or len(v_lines) < 2:
        print("Not enough lines to form cells (need >=2 horizontal and >=2 vertical).")
        return []

    y_positions = sorted(set(int((l[1] + l[3]) / 2) for l in h_lines))
    x_positions = sorted(set(int((l[0] + l[2]) / 2) for l in v_lines))

    cells = []
    for r in range(len(y_positions) - 1):
        y1, y2 = y_positions[r], y_positions[r + 1]
        if (y2 - y1) < min_cell_size:
            continue
        for c in range(len(x_positions) - 1):
            x1, x2 = x_positions[c], x_positions[c + 1]
            if (x2 - x1) < min_cell_size:
                continue
            cells.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "row": r, "col": c, "rowspan": 1, "colspan": 1,
            })

    return cells


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


def draw_cells(img, cells, result_path="cells_detected.png"):
    """Draws every (possibly spanning) cell rectangle for visual review,
    and labels colspan/rowspan when >1. Does NOT crop or save individual
    cell images — cells are returned as coordinates only, meant to be
    matched against text-detection boxes from your separate OCR model."""
    output = img.copy()

    for cell in cells:
        x1, y1, x2, y2 = cell["x1"], cell["y1"], cell["x2"], cell["y2"]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if cell["rowspan"] > 1 or cell["colspan"] > 1:
            label = f'{cell["rowspan"]}x{cell["colspan"]}'
            cv2.putText(output, label, (x1 + 4, y1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

    cv2.imwrite(result_path, output)
    print(f"Extracted {len(cells)} cells (spanning-aware). Saved preview to: {result_path}")


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


def build_line_mask(binary, is_horizontal, w, h,
                     dot_bridge_scale=150, main_kernel_scale=30):
    """
    Two-stage reconstruction so it works for BOTH solid lines and
    dotted/dashed lines made of small separated marks:

    Stage 1 (dot bridging): a SHORT dilation along the line's direction
    merges nearby dots/dashes into one continuous stroke. Without this
    stage, erosion in stage 2 would just erase isolated dots since they
    are smaller than the main kernel.

    Stage 2 (main reconstruction): the same erode->dilate with a long
    kernel as before, which now works because the input is continuous.
    """
    if is_horizontal:
        bridge_len = max(3, w // dot_bridge_scale)
        bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_len, 1))
    else:
        bridge_len = max(3, h // dot_bridge_scale)
        bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, bridge_len))

    # Stage 1: bridge dots/dashes into continuous strokes
    bridged = cv2.dilate(binary, bridge_kernel, iterations=1)

    # Stage 2: standard long-kernel reconstruction (also keeps already-solid lines)
    if is_horizontal:
        main_len = max(10, w // main_kernel_scale)
        main_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (main_len, 1))
    else:
        main_len = max(10, h // main_kernel_scale)
        main_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, main_len))

    eroded = cv2.erode(bridged, main_kernel, iterations=1)
    result = cv2.dilate(eroded, main_kernel, iterations=1)
    return result


def detect_table_grid(
    image_path,
    result_path="grid_detected.png",
    dot_bridge_scale=150,   # smaller = bridges bigger gaps between dots (150 -> ~image_width/150 px bridge)
    main_kernel_scale=30,   # smaller = requires shorter runs to count as a "line"
    min_line_length_ratio=0.05,
    intersection_tolerance=15,  # px tolerance when checking if two lines actually meet
    merge_proximity=15,  # px: lines closer than this (same orientation) get merged into one
    min_extend_length_ratio=0.5,  # a line must already cover this fraction of the table's width/height to be trusted and extended; shorter ones are dropped as noise
):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    skew_angle = estimate_skew_angle(gray)
    print(f"Estimated skew angle: {skew_angle:.2f} degrees")
    if abs(skew_angle) > 0.1:
        img = deskew_image(img, skew_angle)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )

    # NEW: dot-bridging stage happens inside build_line_mask now, so
    # this single call handles solid lines AND dotted/dashed lines.
    horizontal_mask = build_line_mask(
        binary, is_horizontal=True, w=w, h=h,
        dot_bridge_scale=dot_bridge_scale, main_kernel_scale=main_kernel_scale
    )
    vertical_mask = build_line_mask(
        binary, is_horizontal=False, w=w, h=h,
        dot_bridge_scale=dot_bridge_scale, main_kernel_scale=main_kernel_scale
    )

    min_len_h = int(w * min_line_length_ratio)
    min_len_v = int(h * min_line_length_ratio)

    clean_horizontals = extract_lines_from_mask(horizontal_mask, True, min_length=min_len_h)
    clean_verticals = extract_lines_from_mask(vertical_mask, False, min_length=min_len_v)

    # Step A: merge nearby parallel lines that are really the same
    # physical table line (duplicate/split detections)
    clean_horizontals = merge_close_lines(clean_horizontals, is_horizontal=True,
                                           proximity_thresh=merge_proximity)
    clean_verticals = merge_close_lines(clean_verticals, is_horizontal=False,
                                         proximity_thresh=merge_proximity)

    # Step B: keep only lines that intersect at least one line of the
    # opposite orientation — real grid lines always meet each other;
    # stray detections (page edges, isolated marks, leftover noise) don't.
    clean_horizontals, clean_verticals = filter_by_intersection(
        clean_horizontals, clean_verticals, tolerance=intersection_tolerance
    )

    # Step C: extend every line to the table's outer bounds so a
    # partially-detected line (e.g. only reaching 3/4 of the way across)
    # doesn't get mistaken for "no separator here" and cause a false
    # spanning-cell merge.
    clean_horizontals, clean_verticals = extend_lines_to_table_bounds(
        clean_horizontals, clean_verticals, min_extend_length_ratio=min_extend_length_ratio
    )

    grid_mask = cv2.bitwise_or(horizontal_mask, vertical_mask)

    output_img = img.copy()
    for x1, y1, x2, y2 in clean_horizontals:
        cv2.line(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    for x1, y1, x2, y2 in clean_verticals:
        cv2.line(output_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imwrite(result_path, output_img)
    cv2.imwrite("../grid_mask_debug.png", grid_mask)
    cv2.imwrite("../binary_debug.png", binary)

    print("Successfully processed image!")
    print(f"Saved visualization map to: {result_path}")
    print(f"Detected {len(clean_horizontals)} horizontal lines, {len(clean_verticals)} vertical lines")

    # Build cells directly from the cleaned + extended grid — every grid
    # position is its own cell, no span-merging. No cropping — cells are
    # returned as coordinates only, for mapping against your separate
    # text-detection model's output boxes.
    cells = extract_cells(clean_horizontals, clean_verticals)
    draw_cells(img, cells, result_path="../cells_detected.png")

    return clean_horizontals, clean_verticals, cells


if __name__ == "__main__":
    try:
        horiz, vert, cells = detect_table_grid("./temp/ty1.jpg")
    except Exception as e:
        print(f"Error: {e}")