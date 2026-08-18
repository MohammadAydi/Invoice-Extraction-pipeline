"""Record-level matching against the catalogs the desktop app sends.

:mod:`string_matching.algorithms` matches against loose strings; these functions
match against *records*, which is what the desktop app actually needs to bind a
result to a row of its Customers or Products table.

The records themselves and the shape of the answer live in
:mod:`core.domain.catalog` with the rest of the domain. What is here is the
matching.
"""

from __future__ import annotations

from typing import Callable, Iterable

from core.domain.catalog import (
    REVIEW_SCORE_THRESHOLD,
    CatalogEntry,
    Catalogs,
    FieldMatch,
    NamedEntry,
)
from string_matching.algorithms import (
    DEFAULT_TOP_K,
    ScoredCandidate,
    levenshtein_similarity,
    product_match_similarity,
)

__all__ = [
    "REVIEW_SCORE_THRESHOLD",
    "CatalogEntry",
    "Catalogs",
    "FieldMatch",
    "NamedEntry",
    "match_city",
    "match_merchant",
    "match_product",
    "top_entry_matches",
]


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
