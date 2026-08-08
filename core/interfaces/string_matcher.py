from __future__ import annotations

from typing import Protocol

from core.domain.matching import KeywordDictionary, MatchResult


class StringMatcher(Protocol):
    def match(self, text: str, dictionary: KeywordDictionary) -> MatchResult: ...
