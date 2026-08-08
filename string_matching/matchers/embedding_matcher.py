from __future__ import annotations

from core.domain.matching import KeywordDictionary, MatchResult
from string_matching.registry import matcher_registry


@matcher_registry.register("embedding")
class EmbeddingMatcher:
    """Alternative strategy: semantic nearest-neighbor match instead of
    edit distance -- demonstrates that swapping algorithms only requires
    changing `string_matching.algorithm` in config.
    """

    def __init__(self, **params):
        self.params = params

    def match(self, text: str, dictionary: KeywordDictionary) -> MatchResult:
        raise NotImplementedError("Embed text + keywords, return nearest match.")
