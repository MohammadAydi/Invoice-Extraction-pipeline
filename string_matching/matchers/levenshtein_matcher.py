from __future__ import annotations

from core.domain.matching import KeywordDictionary, MatchResult
from string_matching.registry import matcher_registry


@matcher_registry.register("levenshtein")
class LevenshteinMatcher:
    def __init__(self, max_distance: int = 2, **params):
        self.max_distance = max_distance
        self.params = params

    def match(self, text: str, dictionary: KeywordDictionary) -> MatchResult:
        raise NotImplementedError("Find the closest keyword by edit distance.")
