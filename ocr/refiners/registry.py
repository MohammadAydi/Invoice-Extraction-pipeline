from __future__ import annotations

from core.interfaces.region_refiner import RegionRefiner
from core.registry import Registry

refiner_registry: Registry[RegionRefiner] = Registry(kind="region refiner")
