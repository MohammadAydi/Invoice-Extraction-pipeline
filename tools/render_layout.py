"""Render what the table extractor found and what the classifier made of it.

This is where `draw_classified_cells` / `draw_cells` from `temp/table_det.py`
and `detect_table_grid` from `grid_utils.py` ended up. They were library
functions that wrote PNGs -- one of them into its *parent* directory -- as a
side effect of being called. Rendering is a tool, so it lives in tools/.

    python tools/render_layout.py data/raw/test3.jpg
    python tools/render_layout.py data/raw --config config/qwen_config.yaml

It runs the real geometric and table-photometric preprocessing from the config,
then the configured extractor and classifier, so what it draws is exactly what
the pipeline would work from -- not an approximation with its own thresholds.
No OCR runs and no model is loaded.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.loader import load_config  # noqa: E402
from core.domain.image_payload import ImagePayload  # noqa: E402
from core.domain.layout import InvoiceLayout  # noqa: E402
from core.domain.roles import CellRole, Zone  # noqa: E402
from preprocessing.pipeline_builder import PreprocessingPipelineBuilder  # noqa: E402
from table_extraction.classifiers.factory import build_layout_classifier  # noqa: E402
from table_extraction.factory import build_table_extractor  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# BGR. Header, table and footer are separated by hue so the zone split is
# readable at a glance; unclassified cells are gray so they do not compete.
_ZONE_COLOUR = {
    Zone.HEADER: (60, 180, 250),
    Zone.TABLE: (60, 200, 60),
    Zone.FOOTER: (250, 140, 60),
    Zone.UNKNOWN: (150, 150, 150),
}
_UNKNOWN_COLOUR = (150, 150, 150)


def analyze(image_path: Path, config):
    raw = cv2.imread(str(image_path))
    if raw is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    geometric = PreprocessingPipelineBuilder.build(config.preprocessing.geometric_steps)
    table_branch = PreprocessingPipelineBuilder.build(
        config.preprocessing.table_photometric_steps
    )

    display = geometric.run(ImagePayload(image=raw))
    table_payload = table_branch.run(display)

    extractor = build_table_extractor(config.table_extraction)
    classifier = build_layout_classifier(config.table_extraction)

    table = extractor.extract(table_payload)
    height, width = table_payload.image.shape[:2]
    layout = classifier.classify(table, (width, height))

    return display.image, table_payload.image, layout


def draw(page: np.ndarray, layout: InvoiceLayout) -> np.ndarray:
    vis = page.copy() if page.ndim == 3 else cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)

    if layout.table_bbox is not None:
        b = layout.table_bbox
        cv2.rectangle(vis, (b.x, b.y), (b.x + b.w, b.y + b.h), (0, 0, 255), 4)

    scale = max(0.4, min(1.2, vis.shape[1] / 2000.0))

    for region in layout.regions:
        b = region.bbox
        known = region.role is not CellRole.UNKNOWN
        colour = _ZONE_COLOUR.get(region.zone, _UNKNOWN_COLOUR) if known else _UNKNOWN_COLOUR
        cv2.rectangle(vis, (b.x, b.y), (b.x + b.w, b.y + b.h), colour, 2)

        label = region.role.value if known else "?"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(vis, (b.x + 1, b.y), (b.x + tw + 5, b.y + th + 6), (255, 255, 255), -1)
        cv2.putText(
            vis,
            label,
            (b.x + 3, b.y + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            colour,
            1,
            cv2.LINE_AA,
        )

    return vis


def report(path: Path, layout: InvoiceLayout) -> None:
    counts = Counter(r.role.value for r in layout.regions)
    zones = Counter(r.zone.value for r in layout.regions)

    print(f"\n{path.name}  ({layout.source})")
    print(f"  regions: {len(layout.regions)}   zones: {dict(zones)}")
    if layout.table_bbox is not None:
        b = layout.table_bbox
        print(f"  table:   ({b.x}, {b.y}) {b.w}x{b.h}")
    for role, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"    {role:<18} {n}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="an image, or a folder of images")
    parser.add_argument("--config", default="config/qwen_config.yaml")
    parser.add_argument("--out", default="results/layout_debug")
    parser.add_argument(
        "--save-binary",
        action="store_true",
        help="also write the binarized image the extractor actually saw",
    )
    args = parser.parse_args()

    source = Path(args.input)
    if source.is_dir():
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    elif source.is_file():
        images = [source]
    else:
        print(f"Path does not exist: {source}")
        return 1

    config = load_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for path in images:
        try:
            display, binary, layout = analyze(path, config)
        except Exception as exc:  # noqa: BLE001 - one bad page must not stop the batch
            print(f"\n{path.name}: FAILED -- {exc}")
            failures += 1
            continue

        report(path, layout)
        cv2.imwrite(str(out_dir / f"{path.stem}_layout.png"), draw(display, layout))
        if args.save_binary:
            cv2.imwrite(str(out_dir / f"{path.stem}_binary.png"), binary)

    print(f"\nWrote {len(images) - failures} render(s) to {out_dir.resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
