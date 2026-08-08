from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.domain.document import DocumentElement


@dataclass(frozen=True)
class KeywordDictionary:
    """Wraps the keyword list every element's text is matched against.

    Loaded once from StringMatchingConfig.dictionary_path and shared across
    BOTH table-cell and free-field elements -- there is only one dictionary,
    per project decision.
    """

    keywords: list[str]
    source_path: str


@dataclass(frozen=True)
class MatchResult:
    corrected_text: str
    confidence: float
    alternatives: list[str] = field(default_factory=list)


@dataclass
class MatchedElement:
    element: DocumentElement
    match: MatchResult
