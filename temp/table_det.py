import cv2
import numpy as np
import os
import shutil
from collections import defaultdict


def estimate_skew_angle(gray):
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


def _blob_intersects_vertical_mask(vertical_mask, x1, x2, y_center, band=4):
    h, w = vertical_mask.shape[:2]
    y0, y1 = max(0, int(y_center - band)), min(h, int(y_center + band + 1))
    x1, x2 = max(0, int(x1)), min(w, int(x2))
    if x2 <= x1 or y1 <= y0:
        return False
    strip = vertical_mask[y0:y1, x1:x2]
    return bool(np.any(strip > 0))


def remove_stacked_text_blobs(mask, is_horizontal, y_proximity=25,
                               length_similarity=0.35, min_group_size=2,
                               overlap_ratio=0.5, vertical_mask=None,
                               intersection_band=4):
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
        blobs.append({"contour": c, "x1": x, "x2": x + w,
                       "y_center": y + h / 2, "length": w})

    blobs.sort(key=lambda b: b["y_center"])

    clusters = []
    current = [blobs[0]]
    for b in blobs[1:]:
        if abs(b["y_center"] - current[-1]["y_center"]) <= y_proximity:
            current.append(b)
        else:
            clusters.append(current)
            current = [b]
    clusters.append(current)

    def x_overlaps(a, b):
        overlap = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
        shorter = min(a["x2"] - a["x1"], b["x2"] - b["x1"]) or 1
        return (overlap / shorter) >= overlap_ratio

    for cluster in clusters:
        if len(cluster) < min_group_size:
            for b in cluster:
                cv2.drawContours(clean_mask, [b["contour"]], -1, 255,
                                 thickness=cv2.FILLED)
            continue

        lengths = [b["length"] for b in cluster]
        median_len = np.median(lengths)

        similar_count = 0
        for i, b in enumerate(cluster):
            len_ratio = abs(b["length"] - median_len) / (median_len or 1)
            has_overlap = any(
                x_overlaps(b, other)
                for j, other in enumerate(cluster) if j != i
            )
            if len_ratio <= length_similarity and has_overlap:
                similar_count += 1

        cluster_is_text_pattern = similar_count >= min_group_size

        for b in cluster:
            crosses_vertical = (
                vertical_mask is not None and
                _blob_intersects_vertical_mask(
                    vertical_mask, b["x1"], b["x2"], b["y_center"],
                    intersection_band
                )
            )
            target = (removed_mask
                      if (cluster_is_text_pattern and not crosses_vertical)
                      else clean_mask)
            cv2.drawContours(target, [b["contour"]], -1, 255,
                             thickness=cv2.FILLED)

    return clean_mask, removed_mask


def build_line_mask(binary, is_horizontal, w, h,
                    dot_bridge_scale=150, main_kernel_scale=30,
                    use_dot_bridging=True, return_intermediate=False):
    if use_dot_bridging:
        if is_horizontal:
            bridge_len = max(3, w // dot_bridge_scale)
            bridge_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (bridge_len, 1))
        else:
            bridge_len = max(3, h // dot_bridge_scale)
            bridge_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (1, bridge_len))
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


def get_table_cell_bounds(binary_mask, min_size=15, max_area_ratio=0.20,
                           max_aspect_ratio=0.3):
    """
    Returns filtered cell bounding boxes from a binary mask.

    Drops:
      - blobs smaller than min_size in either dimension (noise)
      - blobs covering more than max_area_ratio of the image (outer border)
      - blobs where width/height < max_aspect_ratio (thin vertical slivers)
      - blobs that fully contain another blob (keep the inner one)
    """
    _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)

    img_h, img_w  = binary_mask.shape
    max_cell_area = img_h * img_w * max_area_ratio

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cell_bounds = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_size or h < min_size:
            continue
        if h > 0 and (w / h) < max_aspect_ratio:
            continue
        if w * h > max_cell_area:
            continue
        cell_bounds.append({
            "x": x, "y": y,
            "width": w, "height": h,
            "x_max": x + w,
            "y_max": y + h,
        })

    # --- Drop any cell that fully contains at least one other cell ---
    # "A contains B" means B's box fits entirely inside A's box.
    # We keep B (the inner) and discard A (the outer).
    def contains(outer, inner):
        return (outer["x"]     <= inner["x"]     and
                outer["y"]     <= inner["y"]     and
                outer["x_max"] >= inner["x_max"] and
                outer["y_max"] >= inner["y_max"] and
                outer != inner)

    is_outer = [False] * len(cell_bounds)
    for i, a in enumerate(cell_bounds):
        for j, b in enumerate(cell_bounds):
            if i != j and contains(a, b):
                is_outer[i] = True
                break   # one inner cell is enough to condemn the outer

    cell_bounds = [c for i, c in enumerate(cell_bounds) if not is_outer[i]]

    return cell_bounds
def find_table_bounds(cells, proximity=1):
    """
    Groups nearby cells with union-find and returns the bounding rectangle
    of the largest group (the main table), plus its member cells.

    Returns {"x1", "y1", "x2", "y2", "cells"} or None if cells is empty.
    """
    if not cells:
        return None

    n = len(cells)
    parent = list(range(n))
    rank   = [0] * n

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri == rj:
            return
        if rank[ri] < rank[rj]:
            ri, rj = rj, ri
        parent[rj] = ri
        if rank[ri] == rank[rj]:
            rank[ri] += 1

    def boxes_overlap(a, b):
        return (
            a["x"]     - proximity < b["x_max"] + proximity and
            a["x_max"] + proximity > b["x"]     - proximity and
            a["y"]     - proximity < b["y_max"] + proximity and
            a["y_max"] + proximity > b["y"]     - proximity
        )

    for i in range(n):
        for j in range(i + 1, n):
            if boxes_overlap(cells[i], cells[j]):
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    largest_group = max(groups.values(), key=len)
    group_cells   = [cells[i] for i in largest_group]

    return {
        "x1":   min(c["x"]     for c in group_cells),
        "y1":   min(c["y"]     for c in group_cells),
        "x2":   max(c["x_max"] for c in group_cells),
        "y2":   max(c["y_max"] for c in group_cells),
        "cells": group_cells,
    }


def detect_table_grid(
    image_path,
    result_path="grid_detected.png",
    dot_bridge_scale=150,
    main_kernel_scale=30,
    use_dot_bridging=True,
    use_stack_filter=True,
    stack_y_proximity=25,
    stack_length_similarity=0.35,
    stack_min_group_size=2,
    require_no_vertical_intersection=True,
    stack_intersection_band=4,
    cell_min_size=15,
    cell_max_area_ratio=0.20,
    cell_max_aspect_ratio=0.3,
    table_proximity=3,
    save_steps=True,
    steps_dir="pipeline_steps",
):
    if save_steps:
        os.makedirs(steps_dir, exist_ok=True)

    def save_step(name, image):
        if save_steps:
            cv2.imwrite(os.path.join(steps_dir, name), image)

    # --- Load ---
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    save_step("00_original.png", img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Step 1: deskew ---
    skew_angle = estimate_skew_angle(gray)
    print(f"Estimated skew angle: {skew_angle:.2f} degrees")
    if abs(skew_angle) > 0.1:
        img = deskew_image(img, skew_angle)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_step("01_deskewed.png", img)

    h, w = gray.shape[:2]

    # --- Step 2: binarize ---
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 10
    )
    save_step("02_binary.png", binary)

    # --- Step 3: morphological reconstruction ---
    horizontal_mask, h_bridged = build_line_mask(
        binary, is_horizontal=True, w=w, h=h,
        dot_bridge_scale=dot_bridge_scale,
        main_kernel_scale=main_kernel_scale,
        use_dot_bridging=use_dot_bridging,
        return_intermediate=True,
    )
    vertical_mask, v_bridged = build_line_mask(
        binary, is_horizontal=False, w=w, h=h,
        dot_bridge_scale=dot_bridge_scale,
        main_kernel_scale=main_kernel_scale,
        use_dot_bridging=use_dot_bridging,
        return_intermediate=True,
    )
    save_step("03_dot_bridging.png",
              cv2.bitwise_or(h_bridged, v_bridged))
    save_step("04_morphological_reconstruction.png",
              cv2.bitwise_or(horizontal_mask, vertical_mask))

    # --- Step 4b/4c: stack-filter text blobs from horizontal mask ---
    if use_stack_filter:
        horizontal_mask_clean, h_removed_mask = remove_stacked_text_blobs(
            horizontal_mask, is_horizontal=True,
            y_proximity=stack_y_proximity,
            length_similarity=stack_length_similarity,
            min_group_size=stack_min_group_size,
            vertical_mask=vertical_mask if require_no_vertical_intersection else None,
            intersection_band=stack_intersection_band,
        )
        save_step("04b_mask_after_stack_filter.png",
                  cv2.bitwise_or(horizontal_mask_clean, vertical_mask))
        save_step("04c_mask_removed_as_text.png", h_removed_mask)
        horizontal_mask = horizontal_mask_clean

    # --- Step 5: combine cleaned masks → filtered cells → table bounds ---
    grid_mask = cv2.bitwise_or(horizontal_mask, vertical_mask)
    cv2.imwrite("grid_mask_debug.png", grid_mask)

    cells = get_table_cell_bounds(
        grid_mask.copy(),
        min_size=cell_min_size,
        max_area_ratio=cell_max_area_ratio,
        max_aspect_ratio=cell_max_aspect_ratio,
    )
    print(f"Detected {len(cells)} cells after filtering.")

    table_bounds = find_table_bounds(cells, proximity=table_proximity)
    if table_bounds:
        print(f"Table bounding rect: "
              f"({table_bounds['x1']}, {table_bounds['y1']}) -> "
              f"({table_bounds['x2']}, {table_bounds['y2']}) "
              f"— {len(table_bounds['cells'])} cells in group")

    # --- Single preview: filtered cells (blue) + table rect (red) ---
    preview = img.copy()
    for cell in cells:
        cv2.rectangle(preview,
                      (cell["x"],     cell["y"]),
                      (cell["x_max"], cell["y_max"]),
                      (255, 50, 50), 2)   # blue
    if table_bounds:
        cv2.rectangle(preview,
                      (table_bounds["x1"], table_bounds["y1"]),
                      (table_bounds["x2"], table_bounds["y2"]),
                      (0, 0, 255), 3)     # red

    # result_path and the step image are the same canvas — no divergence
    cv2.imwrite(result_path, preview)
    cells_path = (os.path.join(steps_dir, "05_cells_detected.png")
                  if save_steps else "cells_detected.png")
    cv2.imwrite(cells_path, preview)
    print(f"Preview saved to: {result_path}")

    if save_steps:
        print(f"Step images saved to: {steps_dir}/")

    return cells, grid_mask, table_bounds


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def process_path(input_path, output_dir="results", **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    if os.path.isdir(input_path):
        image_files = sorted([
            f for f in os.listdir(input_path)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ])
        if not image_files:
            print(f"No image files found in folder: {input_path}")
            return results

        print(f"Found {len(image_files)} image(s) in '{input_path}'. Processing...")
        for filename in image_files:
            full_path = os.path.join(input_path, filename)
            name_no_ext = os.path.splitext(filename)[0]
            image_output_dir = os.path.join(output_dir, name_no_ext)
            os.makedirs(image_output_dir, exist_ok=True)
            print(f"\n--- Processing: {filename} ---")
            try:
                cells, grid_mask, table_bounds = _run_single_image(
                    full_path, image_output_dir, **kwargs)
                results[filename] = (cells, grid_mask, table_bounds)
            except Exception as e:
                print(f"Skipped '{filename}' due to error: {e}")

    elif os.path.isfile(input_path):
        filename = os.path.basename(input_path)
        name_no_ext = os.path.splitext(filename)[0]
        image_output_dir = os.path.join(output_dir, name_no_ext)
        os.makedirs(image_output_dir, exist_ok=True)
        print(f"Processing single image: {filename}")
        cells, grid_mask, table_bounds = _run_single_image(
            input_path, image_output_dir, **kwargs)
        results[filename] = (cells, grid_mask, table_bounds)

    else:
        raise FileNotFoundError(f"Path does not exist: {input_path}")

    print(f"\nDone. Processed {len(results)} image(s). Results saved under: {output_dir}/")
    return results


def _run_single_image(image_path, image_output_dir, **kwargs):
    result_path = os.path.join(image_output_dir, "grid_detected.png")
    steps_dir   = os.path.join(image_output_dir,
                                kwargs.pop("steps_dir", "pipeline_steps"))

    cells, grid_mask, table_bounds = detect_table_grid(
        image_path, result_path=result_path, steps_dir=steps_dir, **kwargs
    )

    for fname in ("grid_mask_debug.png", "binary_debug.png"):
        if os.path.exists(fname):
            shutil.move(fname, os.path.join(image_output_dir, fname))

    return cells, grid_mask, table_bounds


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "ty1.jpg"

    try:
        process_path(target)
    except Exception as e:
        print(f"Error: {e}")