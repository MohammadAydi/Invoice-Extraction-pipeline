"""Catalog records and the ranked answer a field match produces.

These are the *targets* matching runs against -- a customer, a product, a
city -- and the shape of its answer. They differ from a loose string in two ways
that matter:

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

The matching *functions* live in :mod:`string_matching.catalog`; only the data
lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from string_matching.algorithms import DEFAULT_TOP_K, ScoredCandidate

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


@dataclass
class Catalogs:
    """The match targets the desktop app sends with each request.

    They are a property of the *request*, not of the configuration: they are the
    C# side's live Customers and Products tables, which change between one
    invoice and the next. Empty lists are normal and simply disable matching for
    that field, leaving the raw OCR text and an empty candidate array.
    """

    merchants: Sequence[CatalogEntry] = field(default_factory=list)
    products: Sequence[CatalogEntry] = field(default_factory=list)
    cities: Sequence[CatalogEntry] = field(default_factory=list)

    # How many ranked alternatives each matched field carries back.
    top_k: int = DEFAULT_TOP_K
