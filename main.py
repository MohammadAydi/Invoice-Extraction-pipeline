"""Entry point / usage example for the pipeline, one image at a time.

This is the command-line path. The desktop app does not use it -- it talks to
`api.main:app` over HTTP -- but both run the exact same PipelineOrchestrator
from the exact same config file, which is what makes this the right place to
reproduce a bad extraction outside the service.

    python main.py bills/invoice_042.jpg
    python main.py bills/invoice_042.jpg config/qwen_config.yaml
"""

from __future__ import annotations

import json
import sys

import cv2

from config.loader import load_config
from core.domain.image_payload import ImagePayload
from orchestration.pipeline_orchestrator import PipelineOrchestrator


def main(image_path: str, config_path: str = "config/default_config.yaml") -> None:
    config = load_config(config_path)
    orchestrator = PipelineOrchestrator(config)

    raw = cv2.imread(image_path)
    if raw is None:
        sys.exit(f"Could not read image: {image_path}")

    run = orchestrator.run(ImagePayload(image=raw))

    print(json.dumps(run.output, ensure_ascii=False, indent=2))
    print(f"\nResults written under: {orchestrator.result_store.output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python main.py <image> [config.yaml]")
    main(*sys.argv[1:3])
