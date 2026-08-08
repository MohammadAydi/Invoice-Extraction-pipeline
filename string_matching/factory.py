from __future__ import annotations

import string_matching.matchers  # noqa: F401  (triggers matcher self-registration)
from config.schema import StringMatchingConfig
from core.interfaces.string_matcher import StringMatcher
from string_matching.registry import matcher_registry


def build_string_matcher(config: StringMatchingConfig) -> StringMatcher:
    return matcher_registry.create(config.algorithm, **config.algorithm_params)
