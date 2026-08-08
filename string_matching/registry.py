from __future__ import annotations

from core.interfaces.string_matcher import StringMatcher
from core.registry import Registry

matcher_registry: Registry[StringMatcher] = Registry(kind="string matcher")
