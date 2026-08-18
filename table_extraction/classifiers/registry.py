from __future__ import annotations

from core.interfaces.layout_classifier import LayoutClassifier
from core.registry import Registry

classifier_registry: Registry[LayoutClassifier] = Registry(kind="layout classifier")
