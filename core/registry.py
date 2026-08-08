"""Generic name -> class registry used by every pluggable family
(preprocessing steps, OCR engines, table extractors, string matchers,
output formatters).

Each family gets its own Registry instance. Concrete implementations
self-register via the `@registry.register("name")` decorator at import
time; factories then build instances purely from the string name found in
config. Adding a new implementation means adding one file -- nothing that
already exists needs to change (Open/Closed principle).
"""

from __future__ import annotations

from typing import Dict, Generic, Type, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self._kind = kind
        self._registry: Dict[str, Type[T]] = {}

    def register(self, name: str):
        def decorator(cls: Type[T]) -> Type[T]:
            if name in self._registry:
                raise ValueError(f"{self._kind} '{name}' is already registered.")
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> Type[T]:
        try:
            return self._registry[name]
        except KeyError:
            available = ", ".join(sorted(self._registry)) or "(none registered)"
            raise KeyError(f"Unknown {self._kind} '{name}'. Available: {available}")

    def create(self, name: str, **kwargs) -> T:
        return self.get(name)(**kwargs)

    def available(self) -> list[str]:
        return sorted(self._registry)
