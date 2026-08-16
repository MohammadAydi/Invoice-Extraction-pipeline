"""The two matching algorithms and the ranked top-K search built on them."""

from __future__ import annotations

import pytest

from string_matching.algorithms import (
    get_top_exact_matches,
    get_top_product_matches,
    levenshtein_distance,
    levenshtein_similarity,
    product_match_similarity,
)
from string_matching.catalog import (
    NamedEntry,
    match_city,
    match_merchant,
    match_product,
)


class TestLevenshtein:
    def test_distance_basics(self):
        assert levenshtein_distance("kitten", "sitting") == 3
        assert levenshtein_distance("", "abc") == 3
        assert levenshtein_distance("abc", "abc") == 0

    def test_identical_after_normalization(self):
        assert levenshtein_similarity("مؤسسة النـور", "مؤسسه النور") == 1.0

    def test_one_edit_in_a_short_word(self):
        # "نور" vs "نوز": one substitution out of three characters.
        assert levenshtein_similarity("نور", "نوز") == pytest.approx(2 / 3)

    def test_one_side_empty_scores_zero(self):
        assert levenshtein_similarity("متجر", "") == 0.0
        assert levenshtein_similarity(None, "متجر") == 0.0

    def test_both_sides_empty_are_identical_not_different(self):
        # The C# duplicate detector relies on this: two invoices with no merchant
        # read are equally unknown, not different businesses.
        assert levenshtein_similarity("", "") == 1.0
        assert levenshtein_similarity(None, "   ") == 1.0

    def test_unrelated_strings_score_low(self):
        assert levenshtein_similarity("متجر النور", "Ahmad Trading") < 0.3


class TestProductMatchSimilarity:
    def test_word_order_does_not_matter(self):
        assert product_match_similarity("جاكيت ازرق صوف", "جاكيت صوف أزرق") == 1.0

    def test_beats_edit_distance_on_reordered_words(self):
        reordered = ("جاكيت ازرق صوف", "جاكيت صوف أزرق")
        assert product_match_similarity(*reordered) > levenshtein_similarity(*reordered)

    def test_one_wrong_letter_costs_a_fraction_of_one_word(self):
        score = product_match_similarity("جاكيت صوف ازرق", "جاكيت صوف ازرك")
        assert 0.85 < score < 1.0

    def test_extra_words_dilute_the_score(self):
        # Two words fully contained in a four-word entry must not score 1.0.
        score = product_match_similarity("جاكيت ازرق", "جاكيت صوف ازرق كبير")
        assert score == pytest.approx(0.5)

    def test_different_products_score_low(self):
        assert product_match_similarity("جاكيت صوف ازرق", "حذاء جلد اسود") < 0.4

    def test_empty(self):
        assert product_match_similarity("", "جاكيت") == 0.0
        assert product_match_similarity("جاكيت", None) == 0.0


class TestTopMatches:
    CUSTOMERS = ["مؤسسة النور التجارية", "متجر السلام", "شركة الأمل", "Ahmad Trading Est."]

    def test_returns_at_most_five(self):
        candidates = get_top_exact_matches("متجر", [f"متجر {i}" for i in range(20)])
        assert len(candidates) == 5

    def test_ranked_highest_first(self):
        candidates = get_top_exact_matches("متجر السلام", self.CUSTOMERS)
        scores = [c.similarity_score for c in candidates]
        assert scores == sorted(scores, reverse=True)
        assert candidates[0].matched_value == "متجر السلام"
        assert candidates[0].similarity_score == 100.0

    def test_scores_are_percentages_rounded_to_two_places(self):
        candidates = get_top_exact_matches("نور", ["النور"])
        score = candidates[0].similarity_score
        assert 0.0 <= score <= 100.0
        assert round(score, 2) == score

    def test_empty_inputs(self):
        assert get_top_exact_matches("", self.CUSTOMERS) == []
        assert get_top_exact_matches("متجر", []) == []

    def test_product_search_uses_word_matching(self):
        products = ["جاكيت صوف أزرق", "جاكيت قطن أزرق", "جاكيت صوف أسود"]
        candidates = get_top_product_matches("جاكيت ازرق صوف", products)

        assert candidates[0].matched_value == "جاكيت صوف أزرق"
        assert candidates[0].similarity_score == 100.0
        assert len(candidates) == 3

    def test_top_k_is_configurable(self):
        assert len(get_top_exact_matches("متجر", self.CUSTOMERS, top_k=2)) == 2


class TestCatalogMatching:
    def test_alias_resolves_to_the_canonical_record(self):
        customers = [
            NamedEntry(name="مؤسسة النور التجارية", entry_id=12, aliases=["متجر النور"]),
            NamedEntry(name="شركة الأمل", entry_id=13),
        ]

        match = match_merchant("متجر النور", customers)
        best = match.best

        # Scored on the alias, but filed under the canonical name and its id.
        assert best.matched_value == "مؤسسة النور التجارية"
        assert best.matched_name == "متجر النور"
        assert best.matched_id == 12
        assert not match.requires_manual_review

    def test_a_record_occupies_one_slot_however_many_names_it_has(self):
        customers = [
            NamedEntry(name="متجر النور", entry_id=1, aliases=["متجر النور", "النور"]),
        ]
        assert len(match_merchant("متجر النور", customers).candidates) == 1

    def test_weak_match_is_flagged_for_review(self):
        customers = [NamedEntry(name="شركة الأمل", entry_id=1)]
        match = match_merchant("مؤسسة النور التجارية", customers)

        assert match.requires_manual_review
        # The candidate still comes back: the user picks from the dropdown.
        assert len(match.candidates) == 1

    def test_empty_catalog_requires_review(self):
        match = match_merchant("متجر النور", [])
        assert match.candidates == []
        assert match.requires_manual_review

    def test_products_match_order_independently(self):
        products = [
            NamedEntry(name="جاكيت صوف أزرق", entry_id=31),
            NamedEntry(name="حذاء جلد أسود", entry_id=32),
        ]
        match = match_product("جاكيت ازرق صوف", products)

        assert match.best.matched_id == 31
        assert not match.requires_manual_review

    def test_cities_use_character_matching(self):
        cities = [NamedEntry(name="دمشق"), NamedEntry(name="حلب"), NamedEntry(name="حمص")]
        match = match_city("دمشق", cities)

        assert match.best.matched_value == "دمشق"
        assert match.best.similarity_score == 100.0
