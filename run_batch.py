# run_batch.py
"""Run the pipeline over a folder of invoices.

main.py handles one image, which matches PipelineOrchestrator.run()'s contract.
This wraps it so a whole folder can go through in one command, and -- crucially
-- builds the orchestrator ONCE so the Qwen worker loads the model a single
time for the entire batch instead of once per invoice (about a minute each).

    python run_batch.py bills/onlyone
    python run_batch.py bills/onlyone --config config/qwen_config.yaml
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import cv2

from config.loader import load_config
from core.domain.image_payload import ImagePayload
from orchestration.pipeline_orchestrator import PipelineOrchestrator

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("bills_dir", nargs="?", default="bills/onlyone",
                   help="folder to scan recursively for invoice images")
    p.add_argument("--config", dest="config_path",
                   default="config/qwen_config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.bills_dir)
    if not root.exists():
        sys.exit(f"Folder not found: {root}")

    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not images:
        sys.exit(f"No images found under {root}")

    print(f"Found {len(images)} image(s) under {root}")
    print(f"Config: {args.config_path}\n")

    # Built once, reused for every image: this is what keeps the model load
    # (and the Surya predictors) out of the per-invoice cost.
    config = load_config(args.config_path)
    orchestrator = PipelineOrchestrator(config)

    failures = []
    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path}")
        try:
            raw = cv2.imread(str(img_path))
            if raw is None:
                raise FileNotFoundError(f"cv2.imread returned None: {img_path}")
            result = orchestrator.run(ImagePayload(image=raw))
            n = len(getattr(result, "elements", []) or [])
            print(f"    ok — invoice_id={getattr(result, 'invoice_id', '?')} "
                  f"elements={n}")
        except Exception as exc:                          # noqa: BLE001
            # One bad invoice must not abort the batch.
            print(f"    !! FAILED: {exc}")
            traceback.print_exc()
            failures.append(str(img_path))

    ok = len(images) - len(failures)
    print(f"\nDone: {ok}/{len(images)} succeeded")
    if failures:
        print("Failed:")
        for f in failures:
            print(f"  {f}")
    print("\nPer-page crops, overlays and results: det/")
    print("Pipeline results:                     results/")


if __name__ == "__main__":
    main()