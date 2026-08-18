from __future__ import annotations

from core.interfaces.region_cropper import RegionCropper
from core.registry import Registry

cropper_registry: Registry[RegionCropper] = Registry(kind="region cropper")
