from __future__ import annotations

from core.domain.matching import KeywordDictionary, MatchResult
from string_matching.algorithms import DEFAULT_TOP_K, get_top_exact_matches
from string_matching.registry import matcher_registry


@matcher_registry.register("levenshtein")
class LevenshteinMatcher:
    """Corrects an element's text to the closest keyword by edit distance.

    Used for the connected-word fields: merchant names, cities and
    governorates. `max_distance` is expressed in edit operations, as the config
    contract specifies, and is converted to a per-comparison similarity floor
    against the length of the text being matched -- a two-edit allowance means
    something very different on a 4-character word than on a 40-character one.
    """

    def __init__(self, max_distance: int = 2, top_k: int = DEFAULT_TOP_K, **params):
        self.max_distance = max_distance
        self.top_k = top_k
        self.params = params

    def match(self, text: str, dictionary: KeywordDictionary) -> MatchResult:
        candidates = get_top_exact_matches(text, dictionary.keywords, self.top_k)

        if not candidates:
            # No dictionary, or nothing readable in the cell: keep what OCR read
            # rather than blanking a value the user could still have corrected.
            return MatchResult(corrected_text=text, confidence=0.0)

        best = candidates[0]
        accepted = best.score_fraction >= self._minimum_score(text)

        return MatchResult(
            corrected_text=best.matched_value if accepted else text,
            confidence=best.score_fraction,
            alternatives=[c.matched_value for c in candidates],
            candidates=[c.as_dict() for c in candidates],
        )

    def _minimum_score(self, text: str) -> float:
        """The similarity that `max_distance` edits correspond to for this text.

        Below it the best keyword is further away than the configured allowance
        and the OCR text stands.
        """
        length = len(text or "")
        if length == 0:
            return 1.0
        return max(0.0, (length - self.max_distance) / length)
