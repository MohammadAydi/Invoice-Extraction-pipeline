"""Normalization rules, per docs/settings-config-contract.md section 2."""

from __future__ import annotations

from datetime import date

import pytest

from string_matching.normalization import (
    digit_sequence,
    fold_digits,
    fold_latin_lookalikes,
    format_date,
    normalize_date_text,
    normalize_price,
    normalize_quantity,
    normalize_text,
    normalize_words,
    parse_date,
    parse_number_strict,
)


class TestNormalizeText:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("أحمد", "احمد"),
            ("إبراهيم", "ابراهيم"),
            ("آمنة", "امنه"),
            ("مصطفى", "مصطفي"),
            ("مؤسسة", "مؤسسه"),
        ],
    )
    def test_folds_letter_variants(self, raw, expected):
        assert normalize_text(raw) == expected

    def test_strips_diacritics(self):
        assert normalize_text("مُحَمَّدٌ") == "محمد"

    def test_strips_tatweel(self):
        assert normalize_text("النـــور") == "النور"

    def test_collapses_whitespace(self):
        assert normalize_text("  متجر    النور  ") == "متجر النور"

    def test_strips_punctuation_but_keeps_digits(self):
        # Digits belong to real product names; punctuation never does.
        assert normalize_text("سكر, 1كغ.") == "سكر 1كغ"

    def test_folds_arabic_indic_digits(self):
        assert normalize_text("رقم ٢٢٩١") == "رقم 2291"

    def test_casefolds_latin(self):
        assert normalize_text("Ahmad Trading EST.") == "ahmad trading est"

    def test_empty_input(self):
        assert normalize_text(None) == ""
        assert normalize_text("   ") == ""

    def test_punctuation_alone_normalizes_away(self):
        # This is what makes "سوبرماركت الأمل!" and "سوبرماركت الأمل" one name.
        assert normalize_text("!") == ""

    def test_two_spellings_of_one_name_converge(self):
        assert normalize_text("مؤسسة النـور") == normalize_text("مؤسسه النور")


class TestNormalizeWords:
    def test_splits_after_normalizing(self):
        assert normalize_words("جاكيت  صوف   أزرق") == ["جاكيت", "صوف", "ازرق"]

    def test_empty(self):
        assert normalize_words("") == []


class TestFoldDigits:
    def test_arabic_indic(self):
        assert fold_digits("٠١٢٣٤٥٦٧٨٩") == "0123456789"

    def test_extended_arabic_indic(self):
        assert fold_digits("۰۱۲۳۴۵۶۷۸۹") == "0123456789"

    def test_leaves_other_characters(self):
        assert fold_digits("سعر ٢٥ ريال") == "سعر 25 ريال"


class TestNormalizeQuantity:
    def test_plain_integer(self):
        assert normalize_quantity("12") == 12

    def test_arabic_indic(self):
        assert normalize_quantity("٢٥") == 25

    @pytest.mark.parametrize("raw", ["٢.٠", "٢،٠", "2.0", "2,0"])
    def test_separators_are_misread_strokes_not_decimal_points(self, raw):
        # The agreed rule: a quantity is a whole number, so everything that is
        # not a digit is dropped rather than interpreted.
        assert normalize_quantity(raw) == 20

    def test_strips_surrounding_noise(self):
        assert normalize_quantity("عدد 7") == 7

    def test_no_digits(self):
        assert normalize_quantity("قطعة") is None
        assert normalize_quantity("") is None
        assert normalize_quantity(None) is None


class TestNormalizePrice:
    def test_plain_decimal(self):
        assert normalize_price("12.50") == 12.50

    def test_arabic_decimal_separator(self):
        assert normalize_price("١٢٫٥٠") == 12.50

    def test_comma_as_decimal_point(self):
        assert normalize_price("517,65") == 517.65

    def test_thousands_separator_with_decimal_point(self):
        assert normalize_price("1,234.50") == 1234.50

    def test_european_grouping(self):
        assert normalize_price("1.234,50") == 1234.50

    def test_multiple_dots_are_all_thousands(self):
        assert normalize_price("1.234.500") == 1234500.0

    def test_strips_currency(self):
        assert normalize_price("128.75 ر.س") == 128.75
        assert normalize_price("SAR 99.00") == 99.00

    def test_no_number(self):
        assert normalize_price("مجاني") is None
        assert normalize_price(None) is None


class TestParseNumberStrict:
    def test_whole_string_number(self):
        assert parse_number_strict("2.50") == 2.50

    def test_rejects_number_embedded_in_text(self):
        # This is the guard that keeps "سكر 1كغ" a product name rather than the
        # number 1, which would both shift the column mapping and erase the name.
        assert parse_number_strict("سكر 1كغ") is None
        assert parse_number_strict("A4 Paper") is None

    def test_rejects_empty(self):
        assert parse_number_strict("") is None


class TestDates:
    def test_unifies_separators(self):
        assert normalize_date_text("14-03-2026") == "14/03/2026"
        assert normalize_date_text("14.03.2026") == "14/03/2026"
        assert normalize_date_text("14 / 03 / 2026") == "14/03/2026"

    def test_folds_arabic_indic_digits(self):
        assert normalize_date_text("١٤-٠٣-٢٠٢٦") == "14/03/2026"

    def test_collapses_duplicate_separators(self):
        assert normalize_date_text("14//03///2026") == "14/03/2026"

    def test_parses_day_first(self):
        assert parse_date("03/04/2026") == date(2026, 4, 3)

    def test_parses_iso(self):
        assert parse_date("2026-03-14") == date(2026, 3, 14)

    def test_corrects_unambiguous_reversal(self):
        # 14 cannot be a month, so the components must be the other way round.
        assert parse_date("03/14/2026") == date(2026, 3, 14)

    def test_two_digit_year(self):
        assert parse_date("14/03/26") == date(2026, 3, 14)

    def test_rejects_impossible_date(self):
        assert parse_date("31/02/2026") is None

    def test_finds_date_inside_a_label(self):
        assert parse_date("التاريخ: ١٤/٠٣/٢٠٢٦") == date(2026, 3, 14)

    def test_format_is_iso(self):
        assert format_date(date(2026, 3, 14)) == "2026-03-14"
        assert format_date(None) is None


class TestLatinLookalikes:
    """The three digit-to-letter mis-readings the VLM recognizers make.

    Measured on this project's own invoices and documented in the project log;
    the map is deliberately not extended past them. Every product name passes
    through the same normalizer, and folding a fourth letter on a hunch would
    corrupt real text to fix a mis-reading nobody has seen.
    """

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("O", "5"),
            ("o", "5"),
            ("V", "7"),
            ("v", "7"),
            ("A", "8"),
            ("1O5", "155"),
            # Anything else is left exactly as it was.
            ("B", "B"),
            ("سكر", "سكر"),
            ("", ""),
        ],
    )
    def test_folds_only_the_documented_three(self, raw, expected):
        assert fold_latin_lookalikes(raw) == expected

    def test_handles_none(self):
        assert fold_latin_lookalikes(None) == ""


class TestDigitSequence:
    """The reading a numeric cell is most likely to have got right.

    These recognizers misplace separators far more often than they misread a
    digit, so the bare sequence is what invoice.reconciliation searches decimal
    positions over.
    """

    def test_strips_the_noise_a_dot_pattern_adds(self):
        # A real reading: the crop caught the printed dot leader between the
        # digits. The sequence is 517.65 exactly right.
        assert digit_sequence("٥-١-٧,٦,٥") == "51765"

    def test_folds_arabic_indic_digits(self):
        assert digit_sequence("١٢٣") == "123"

    def test_folds_latin_lookalikes_first(self):
        assert digit_sequence("1O5") == "155"

    def test_a_cell_with_no_digits_is_empty(self):
        assert digit_sequence("ثلاثون") == ""
        assert digit_sequence(None) == ""
