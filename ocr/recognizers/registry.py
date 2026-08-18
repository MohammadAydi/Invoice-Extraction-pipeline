from __future__ import annotations

from core.interfaces.text_recognizer import TextRecognizer
from core.registry import Registry

recognizer_registry: Registry[TextRecognizer] = Registry(kind="text recognizer")
