"""Side-by-side comparison of the old deskew behaviour against the merged one.

Three deskew implementations used to exist in this project and they disagreed.
Rather than assert which is better, this renders both over a folder of real
invoices so the answer can be looked at.

    python tools/compare_deskew.py data/raw
    python tools/compare_deskew.py data/raw --out results/deskew_comparison

"old" is the behaviour that shipped in `preprocessing/steps/geometric/deskew.py`
before the merge: near-horizontal segments only (anything past `max_angle_deg`
discarded), rotation into the original frame. "merged" is what ships now:
angles folded modulo 90 so vertical rules vote too, and an expanded canvas so
no corner is clipped.

Both are the same class with different parameters, which is the point -- the
old behaviour is still reachable from config if the comparison says it is
better on your forms.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Running this as a script puts tools/ on sys.path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.domain.image_payload import ImagePayload, PipelineContext  # noqa: E402
from preprocessing.steps.geometric.deskew import HoughDeskewStep  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

VARIANTS: dict[str, dict] = {
    # What the pipeline did before: horizontals only, rotate in place.
    "old": {"fold_to_90": False, "expand_canvas": False, "blur_kernel": 3},
    # What it does now: every rule votes, nothing is clipped.
    "merged": {"fold_to_90": True, "expand_canvas": True, "blur_kernel": 3},
}

_LABEL_HEIGHT = 34


def _run(step: HoughDeskewStep, image: np.ndarray) -> tuple[np.ndarray, dict]:
    payload = ImagePayload(image=image.copy())
    payload.metadata["collect_debug_images"] = False
    result = step.apply(PipelineContext(payload=payload))
    return result.payload.image, result.payload.metadata.get(step.name, {})


def _labelled(image: np.ndarray, text: str) -> np.ndarray:
    canvas = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    banner = np.full((_LABEL_HEIGHT, canvas.shape[1], 3), 255, np.uint8)
    cv2.putText(
        banner, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA
    )
    return np.vstack([banner, canvas])


def _side_by_side(panels: list[np.ndarray], gutter: int = 12) -> np.ndarray:
    """Pad every panel to the tallest, then lay them out left to right.

    The variants deliberately produce different output sizes -- that difference
    is one of the things being compared -- so the panels cannot simply be
    hstacked.
    """
    height = max(p.shape[0] for p in panels)
    padded = []
    for panel in panels:
        pad = height - panel.shape[0]
        if pad:
            panel = cv2.copyMakeBorder(
                panel, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(235, 235, 235)
            )
        padded.append(panel)
        padded.append(np.full((height, gutter, 3), (60, 60, 60), np.uint8))
    return np.hstack(padded[:-1])


def compare_image(path: Path, out_dir: Path, max_width: int) -> dict[str, dict]:
    original = cv2.imread(str(path))
    if original is None:
        print(f"  ! could not read {path.name}")
        return {}

    panels = [_labelled(original, f"{path.name}  original  {original.shape[1]}x{original.shape[0]}")]
    report: dict[str, dict] = {}

    for name, params in VARIANTS.items():
        step = HoughDeskewStep(**params)
        deskewed, meta = _run(step, original)
        report[name] = meta

        angle = meta.get("median_angle_deg", 0.0)
        lines = meta.get("num_lines_used", 0)
        note = "  SKIPPED" if "skipped" in meta else ""
        caption = (
            f"{name}  angle={angle:+.2f} deg  lines={lines}  "
            f"{deskewed.shape[1]}x{deskewed.shape[0]}{note}"
        )
        panels.append(_labelled(deskewed, caption))

        cv2.imwrite(str(out_dir / f"{path.stem}_{name}.png"), deskewed)

    sheet = _side_by_side(panels)
    if sheet.shape[1] > max_width:
        scale = max_width / sheet.shape[1]
        sheet = cv2.resize(
            sheet, (max_width, int(sheet.shape[0] * scale)), interpolation=cv2.INTER_AREA
        )
    cv2.imwrite(str(out_dir / f"{path.stem}_comparison.png"), sheet)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="an image, or a folder of images")
    parser.add_argument("--out", default="results/deskew_comparison")
    parser.add_argument(
        "--max-width",
        type=int,
        default=2400,
        help="downscale the side-by-side sheet to this width (the per-variant "
        "PNGs are always written at full size)",
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

    if not images:
        print(f"No images found in {source}")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str, str, str, str]] = []
    for path in images:
        print(f"- {path.name}")
        report = compare_image(path, out_dir, args.max_width)
        for name, meta in report.items():
            size = meta.get("output_size")
            rows.append(
                (
                    path.name,
                    name,
                    f"{meta.get('median_angle_deg', 0.0):+.2f}",
                    str(meta.get("num_lines_used", 0)),
                    f"{size[0]}x{size[1]}" if size else "unchanged",
                )
            )

    header = ("image", "variant", "angle", "lines", "output")
    widths = [
        max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
        for i in range(5)
    ]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(header))
    print(f"\n{line}\n{'-' * len(line)}")
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(5)))

    print(f"\nWrote {len(images)} comparison sheet(s) to {out_dir.resolve()}")
    print("Look at *_comparison.png: original | old | merged, left to right.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
