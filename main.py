"""Entry point / usage example for the backbone.

Wires everything from config/default_config.yaml and shows the intended
call shape. Running this end to end will raise NotImplementedError once
it reaches the first unfilled stub -- that's expected until each stage's
real implementation is filled in by the team.
"""

from __future__ import annotations

import sys

import cv2

from config.loader import load_config
from core.domain.image_payload import ImagePayload
from orchestration.pipeline_orchestrator import PipelineOrchestrator


def main(image_path: str, config_path: str = "config/default_config.yaml") -> None:
    config = load_config(config_path) #load the config file
    orchestrator = PipelineOrchestrator(config) # init pipeline

    raw = cv2.imread(image_path) # read
    payload = ImagePayload(image=raw) #payload

    result = orchestrator.run(payload) # run the pipeline
    print(result)


if __name__ == "__main__":
    main(sys.argv[1])
