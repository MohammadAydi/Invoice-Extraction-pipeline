from __future__ import annotations

from core.interfaces.text_detector import TextDetector
from core.registry import Registry

detector_registry: Registry[TextDetector] = Registry(kind="text detector")
