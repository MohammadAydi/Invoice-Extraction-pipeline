"""The two similarity algorithms, and the top-K search built on them.

Both algorithms return a *ranked list*, not a single winner. That is the whole
point: the desktop app shows the best candidate pre-selected in an editable
combo box and lets the user pick a different one from the list, so a
near-miss costs one click instead of a re-typed line. A matcher that returned
only its favourite would throw that away.

Two algorithms, because names fail in two different ways:

* :func:`levenshtein_similarity` -- character edits. Right for merchant, city
  and governorate names, which are read as one run of characters and go wrong
  one letter at a time.
* :func:`product_match_similarity` -- word-set overlap scored with the same
  edit distance underneath. Right for product names, where the words are
  correct but their order is not: "جاكيت صوف أزرق" and "جاكيت أزرق صوف" are
  the same product and score ~1.0 here while scoring poorly on raw edit
  distance.

Every input passes through :mod:`string_matching.normalization` first, so
diacritics, letter-form variants and Arabic-Indic digits never decide a match.
"""

from __future__ import annotations

from dataclasses import dataclass

from string_matching.normalization import normalize_text, normalize_words

# The desktop app renders a dropdown; five entries is what fits without
# scrolling, and past the fifth the scores are noise anyway.
DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class ScoredCandidate:
    """One catalog value and how well it scored, as a percentage 0-100."""

    matched_value: str
    similarity_score: float

    # Primary key of the catalog record, when matching against records rather
    # than loose strings. None for dictionary/keyword matches.
    matched_id: int | None = None

    # The name that actually scored, when it differs from the canonical
    # ``matched_value`` -- i.e. the alias an invoice was printed with.
    matched_name: str | None = None

    def as_dict(self) -> dict:
        """The wire shape from the API contract's candidate array."""
        payload: dict = {
            "matched_value": self.matched_value,
            "similarity_score": self.similarity_score,
        }
        if self.matched_id is not None:
            payload["matched_id"] = self.matched_id
        if self.matched_name is not None and self.matched_name != self.matched_value:
            payload["matched_name"] = self.matched_name
        return payload

    @property
    def score_fraction(self) -> float:
        """The score as a 0-1 fraction, which is what the field envelope uses."""
        return self.similarity_score / 100.0


# --------------------------------------------------------------------------
# Similarity
# --------------------------------------------------------------------------


def levenshtein_distance(s1: str, s2: str) -> int:
    """Edit distance between two strings, two-row dynamic programming.

    O(len(s1) x len(s2)) time and O(min) space. Invoice strings are short, so
    the straightforward implementation is the right one -- and keeping it here
    rather than pulling in a C extension keeps the service installable on a
    machine with no build toolchain.
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    if not s2:
        return len(s1)

    previous_row = list(range(len(s2) + 1))

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def levenshtein_similarity(s1: str | None, s2: str | None) -> float:
    """Edit-distance similarity in [0, 1], where 1.0 is an exact match.

    ``(max_len - distance) / max_len``: the fraction of the longer string that
    did not need editing. Both arguments are normalized first, so the caller
    passes raw OCR text and raw catalog text and does not have to remember to
    fold them.

    Empty strings: *one* side empty scores 0.0 -- nothing matched. *Both* sides
    empty scores 1.0, because the two strings really are identical; the
    reference implementation returns 0.0 there only as a division-by-zero
    guard, and the C# side depends on the 1.0 for "two invoices with no
    merchant read are equally unknown, not different". The case never arises in
    catalog matching, where empty input returns an empty candidate list before
    any scoring happens.
    """
    left = normalize_text(s1)
    right = normalize_text(s2)

    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0

    max_len = max(len(left), len(right))
    return (max_len - levenshtein_distance(left, right)) / max_len


def product_match_similarity(ocr_product: str | None, db_product: str | None) -> float:
    """Order-independent similarity in [0, 1], for product names.

    Each word of the OCR text is scored against its best match among the
    catalog words, and the total is divided by the *larger* word count. Using
    the larger count is what stops a two-word OCR read from scoring 1.0 against
    a five-word catalog entry just because both its words appear in it -- extra
    words on either side dilute the score, as they should.

    Word-level scoring uses :func:`levenshtein_similarity`, so a single mis-read
    letter inside one word costs a fraction of that word rather than the match.
    """
    ocr_words = normalize_words(ocr_product)
    db_words = normalize_words(db_product)

    if not ocr_words or not db_words:
        return 0.0

    total_score = 0.0
    for ocr_word in ocr_words:
        best = 0.0
        for db_word in db_words:
            # Normalization already ran on the whole string, so compare the
            # words directly rather than re-folding each one.
            score = _word_similarity(ocr_word, db_word)
            if score > best:
                best = score
                if best == 1.0:
                    break
        total_score += best

    return total_score / max(len(ocr_words), len(db_words))


def _word_similarity(left: str, right: str) -> float:
    """Edit-distance similarity between two already-normalized words."""
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0

    max_len = max(len(left), len(right))
    return (max_len - levenshtein_distance(left, right)) / max_len


# --------------------------------------------------------------------------
# Top-K search
# --------------------------------------------------------------------------


def _to_percentage(score: float) -> float:
    return round(score * 100, 2)


def get_top_exact_matches(
    ocr_text: str | None,
    db_list: list[str],
    top_k: int = DEFAULT_TOP_K,
) -> list[ScoredCandidate]:
    """Best ``top_k`` entries of ``db_list`` by edit distance, highest first.

    This is the entry point for merchant, city and governorate names.
    """
    return _rank(ocr_text, db_list, levenshtein_similarity, top_k)


def get_top_product_matches(
    ocr_product: str | None,
    db_products_list: list[str],
    top_k: int = DEFAULT_TOP_K,
) -> list[ScoredCandidate]:
    """Best ``top_k`` products by order-independent word matching."""
    return _rank(ocr_product, db_products_list, product_match_similarity, top_k)


def _rank(
    text: str | None,
    values: list[str],
    score_fn,
    top_k: int,
) -> list[ScoredCandidate]:
    if not text or not values:
        return []

    results = [
        ScoredCandidate(
            matched_value=value,
            similarity_score=_to_percentage(score_fn(text, value)),
        )
        for value in values
        if value
    ]

    # Ties break on the catalog's own order, which is alphabetical by name, so
    # repeated runs over an unchanged catalog rank identically. `sorted` is
    # stable, so listing the sort key alone is enough to guarantee that.
    results.sort(key=lambda candidate: candidate.similarity_score, reverse=True)
    return results[:top_k]
