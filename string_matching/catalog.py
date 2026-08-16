"""Record-level matching against the catalogs the desktop app sends.

:mod:`string_matching.algorithms` matches against loose strings. This module
matches against *records* -- a customer, a product, a city -- which differs in
two ways that matter:

* A record can answer to several names. A customer has a canonical ``Name`` and
  an ``AliasName``, and an invoice may be printed with either. Both compete on
  equal terms, the record wins on its best-scoring name, and the answer is
  always reported under the canonical name so the invoice is filed consistently.
* A record has a primary key. Returning it lets the app bind the extracted value
  straight to the ``Customers`` / ``Products`` row instead of resolving the name
  a second time and possibly resolving it differently.

The result is always a ranked list plus a ``requires_manual_review`` flag rather
than a single answer, because a wrong confident match silently corrupts an
invoice while a ranked list costs the user one click.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol, runtime_checkable

from string_matching.algorithms import (
    DEFAULT_TOP_K,
    ScoredCandidate,
    levenshtein_similarity,
    product_match_similarity,
)

# Below this the top candidate is not trustworthy enough to accept unseen, and
# the field is flagged for the user to confirm. Same number as the API
# contract's confidence review threshold, deliberately: the verification screen
# highlights both kinds of uncertainty the same way.
REVIEW_SCORE_THRESHOLD = 0.75


@runtime_checkable
class CatalogEntry(Protocol):
    """A match target with a primary key and one or more equivalent names.

    Structural rather than nominal, so this module does not depend on the
    Pydantic wire models: anything exposing these three members can be matched.
    """

    @property
    def entry_id(self) -> int | None: ...

    @property
    def name(self) -> str: ...

    @property
    def names(self) -> list[str]: ...


@dataclass(frozen=True)
class NamedEntry:
    """Plain in-process catalog entry, for callers that are not on the wire."""

    name: str
    entry_id: int | None = None
    aliases: list[str] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in (self.name, *self.aliases):
            text = (value or "").strip()
            key = text.casefold()
            if text and key not in seen:
                seen.add(key)
                result.append(text)
        return result


@dataclass(frozen=True)
class FieldMatch:
    """The ranked answer for one field, ready to be put on the wire."""

    ocr_raw_text: str
    candidates: list[ScoredCandidate]

    @property
    def best(self) -> ScoredCandidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def requires_manual_review(self) -> bool:
        """True when the top candidate is too weak to accept without a human.

        No candidates at all also counts: an empty catalog or an unreadable cell
        both leave the user with the raw OCR text to resolve.
        """
        best = self.best
        return best is None or best.score_fraction < REVIEW_SCORE_THRESHOLD


def top_entry_matches(
    text: str | None,
    entries: Iterable[CatalogEntry],
    score_fn: Callable[[str | None, str | None], float],
    top_k: int = DEFAULT_TOP_K,
) -> list[ScoredCandidate]:
    """Rank catalog *records* for ``text``, best first.

    Every name a record carries is scored and the record keeps only its best,
    so a customer whose alias and canonical name both score does not occupy two
    slots in a five-slot dropdown.
    """
    if not text:
        return []

    best_per_entry: list[ScoredCandidate] = []

    for entry in entries:
        entry_best: ScoredCandidate | None = None

        for name in entry.names:
            score = round(score_fn(text, name) * 100, 2)
            if entry_best is not None and score <= entry_best.similarity_score:
                continue

            entry_best = ScoredCandidate(
                # A record carrying only aliases and no canonical name still
                # needs something to be filed under.
                matched_value=(entry.name or "").strip() or name,
                similarity_score=score,
                matched_id=entry.entry_id,
                matched_name=name,
            )

        if entry_best is not None:
            best_per_entry.append(entry_best)

    best_per_entry.sort(key=lambda candidate: candidate.similarity_score, reverse=True)
    return best_per_entry[:top_k]


def match_merchant(
    text: str | None,
    merchants: Iterable[CatalogEntry],
    top_k: int = DEFAULT_TOP_K,
) -> FieldMatch:
    """Merchant name: character-level edit distance across name and aliases."""
    return FieldMatch(
        ocr_raw_text=text or "",
        candidates=top_entry_matches(text, merchants, levenshtein_similarity, top_k),
    )


def match_product(
    text: str | None,
    products: Iterable[CatalogEntry],
    top_k: int = DEFAULT_TOP_K,
) -> FieldMatch:
    """Product name: order-independent word matching.

    Products carry a name and nothing else, so the name is the entire matching
    surface; the id only travels back so the app can link the line to the row.
    """
    return FieldMatch(
        ocr_raw_text=text or "",
        candidates=top_entry_matches(text, products, product_match_similarity, top_k),
    )


def match_city(
    text: str | None,
    cities: Iterable[CatalogEntry],
    top_k: int = DEFAULT_TOP_K,
) -> FieldMatch:
    """City or governorate: same character-level matching as a merchant name."""
    return FieldMatch(
        ocr_raw_text=text or "",
        candidates=top_entry_matches(text, cities, levenshtein_similarity, top_k),
    )
