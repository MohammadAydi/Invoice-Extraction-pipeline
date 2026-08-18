from __future__ import annotations

from core.domain.matching import KeywordDictionary, MatchResult
from string_matching.algorithms import DEFAULT_TOP_K, get_top_product_matches
from string_matching.registry import matcher_registry


@matcher_registry.register("order_independent")
class OrderIndependentMatcher:
    """Word-set matcher, for names whose words are right but whose order is not.

    "جاكيت صوف أزرق" and "جاكيت أزرق صوف" name the same product; edit distance
    over the whole string scores them poorly because it counts the moved words
    as a long run of substitutions. Scoring each word against its best partner
    and dividing by the larger word count removes the ordering from the answer
    while still penalising words that are on one side only.

    Selected by setting `string_matching.algorithm: order_independent`; the
    same algorithm backs product matching against the Products catalog
    regardless of what this option is set to, since a product name is the only
    thing it is ever right for.
    """

    def __init__(self, min_score: float = 0.75, top_k: int = DEFAULT_TOP_K, **params):
        self.min_score = min_score
        self.top_k = top_k
        self.params = params

    def match(self, text: str, dictionary: KeywordDictionary) -> MatchResult:
        candidates = get_top_product_matches(text, dictionary.keywords, self.top_k)

        if not candidates:
            return MatchResult(corrected_text=text, confidence=0.0)

        best = candidates[0]

        return MatchResult(
            corrected_text=(
                best.matched_value if best.score_fraction >= self.min_score else text
            ),
            confidence=best.score_fraction,
            alternatives=[c.matched_value for c in candidates],
            candidates=[c.as_dict() for c in candidates],
        )
