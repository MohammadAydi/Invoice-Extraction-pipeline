from __future__ import annotations

from core.registry import Registry

# Typed loosely on purpose: importing ExtractionFlow here would be circular,
# since every flow module imports this registry to register itself.
flow_registry: Registry = Registry(kind="extraction flow")
