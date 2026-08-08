from __future__ import annotations

from core.interfaces.preprocessing import PreprocessingStep
from core.registry import Registry

step_registry: Registry[PreprocessingStep] = Registry(kind="preprocessing step")
