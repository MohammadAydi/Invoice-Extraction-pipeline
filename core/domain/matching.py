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
    """What a StringMatcher made of one element's text.

    `corrected_text` is the value to show; it falls back to the original text
    when nothing scored well enough to replace it, so a bad dictionary never
    destroys what OCR read.

    `candidates` carries the full ranked list -- `{"matched_value", "similarity_score"}`
    per entry, highest first -- which is what the desktop app binds its editable
    dropdown to. `alternatives` is the same list reduced to bare strings, kept
    because it is the shape the persisted result files already use.
    """

    corrected_text: str
    confidence: float
    alternatives: list[str] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)


@dataclass
class MatchedElement:
    element: DocumentElement
    match: MatchResult
