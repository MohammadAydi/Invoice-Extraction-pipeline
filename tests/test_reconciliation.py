"""Arithmetic reconciliation of line items, per invoice/reconciliation.py.

Each case here is a reading these recognizers actually produce. The stage's
whole premise is that the *digits* are usually right and the separator usually
is not, so the tests are written as "this is what came back, this is what the
row must end up saying".
"""

from __future__ import annotations

import pytest

from core.domain.invoice import (
    ExtractedValue,
    InvoiceDraft,
    InvoiceLineItem,
    WarningCodes,
)
from invoice.reconciliation import (
    decimal_candidates,
    dot_run_candidates,
    observed_decimals,
    reconcile,
)


def row(quantity=None, unit_price=None, total=None, index=0) -> InvoiceLineItem:
    """One line item, built from raw cell readings.

    `raw` is what the recognizer returned and `value` what the parser made of
    it -- None where the strict number rules refused the reading, which is
    exactly the case reconciliation exists for.
    """

    def cell(reading):
        if reading is None:
            return ExtractedValue()

        raw, value = reading if isinstance(reading, tuple) else (str(reading), reading)
        return ExtractedValue(value=value, confidence=0.9, raw=raw)

    item = InvoiceLineItem(row_index=index)
    item.quantity = cell(quantity)
    item.unit_price = cell(unit_price)
    item.total_price = cell(total)
    return item


def reconciled(item: InvoiceLineItem) -> InvoiceLineItem:
    reconcile(InvoiceDraft(line_items=[item]))
    return item


class TestDecimalCandidates:
    def test_every_legal_separator_position(self):
        assert decimal_candidates("51765") == [51765.0, 5176.5, 517.65]

    def test_a_short_sequence_runs_out_of_positions(self):
        assert decimal_candidates("5") == [5.0]

    def test_an_empty_sequence_offers_nothing(self):
        assert decimal_candidates("") == []


class TestDotRunCandidates:
    """These models write trailing zeros as dots."""

    def test_expands_a_trailing_dot_run(self):
        assert dot_run_candidates("212...") == ["212", "2120", "21200", "212000", "2120000"]

    def test_a_reading_with_no_trailing_dots_is_left_alone(self):
        assert dot_run_candidates("212") == ["212"]

    def test_a_cell_with_no_digits_offers_nothing(self):
        assert dot_run_candidates("ثلاثون") == []


class TestObservedDecimals:
    def test_reads_the_fraction_length_off_a_clean_reading(self):
        assert observed_decimals("517.65") == 2
        assert observed_decimals("5176.5") == 1

    def test_a_multi_separator_reading_carries_no_signal(self):
        # Dot-pattern noise, not formatting. Trusting it would mislead.
        assert observed_decimals("٥-١-٧,٦,٥") is None

    def test_no_separator_at_all(self):
        assert observed_decimals("51765") is None


class TestRepairingSeparators:
    def test_finds_the_decimal_position_the_equation_agrees_with(self):
        """The classic case: three readings, one consistent placement."""
        item = reconciled(row(quantity=("3", 3), unit_price=("517.65", 517.65),
                              total=("155295", 155295.0)))

        assert item.unit_price.value == pytest.approx(517.65)
        assert item.quantity.value == 3
        assert item.total_price.value == pytest.approx(1552.95)
        assert item.arithmetic_ok

    def test_recovers_a_cell_the_strict_parser_refused(self):
        """"٥-١-٧,٦,٥" parses to nothing but its digits are 517.65 exactly."""
        item = reconciled(row(quantity=("3", 3), unit_price=("٥-١-٧,٦,٥", None),
                              total=("1552.95", 1552.95)))

        assert item.unit_price.value == pytest.approx(517.65)
        assert item.arithmetic_ok

    def test_a_row_that_already_multiplies_out_is_left_alone(self):
        item = reconciled(row(quantity=("2", 2), unit_price=("150.00", 150.0),
                              total=("300.00", 300.0)))

        assert item.unit_price.value == pytest.approx(150.0)
        assert item.quantity.value == 2
        assert item.total_price.value == pytest.approx(300.0)
        assert item.arithmetic_ok


class TestDerivingAMissingCell:
    def test_computes_the_total_from_price_and_quantity(self):
        item = reconciled(row(quantity=("4", 4), unit_price=("25.50", 25.5)))

        assert item.total_price.value == pytest.approx(102.0)

    def test_computes_the_price_from_total_and_quantity(self):
        item = reconciled(row(quantity=("4", 4), total=("102.00", 102.0)))

        assert item.unit_price.value == pytest.approx(25.5)

    def test_one_known_cell_derives_nothing(self):
        """Two knowns is the minimum. One number cannot imply the other two."""
        item = reconciled(row(quantity=("4", 4)))

        assert item.unit_price.value is None
        assert item.total_price.value is None


class TestRefusingToGuess:
    def test_an_irreconcilable_row_keeps_what_was_read(self):
        """No decimal placement works, and the derived value explains nothing.

        The row is left exactly as the recognizer produced it and flagged. A
        confident wrong number is worse than a visibly unresolved one -- the
        verification screen is where this gets settled.
        """
        item = reconciled(row(quantity=("2", 2), unit_price=("7", 7.0),
                              total=("9999999", 9999999.0)))

        assert item.quantity.value == 2
        assert item.unit_price.value == pytest.approx(7.0)
        assert not item.arithmetic_ok

    def test_an_empty_row_is_untouched(self):
        item = reconciled(row())

        assert item.quantity.value is None
        assert item.unit_price.value is None
        assert item.total_price.value is None
        # Nothing to disagree about.
        assert item.arithmetic_ok


class TestWarnings:
    def test_a_repaired_row_no_longer_reports_a_mismatch(self):
        """The parser raised the warning against the row as it was read."""
        item = row(quantity=("3", 3), unit_price=("517.65", 517.65),
                   total=("155295", 155295.0))
        item.arithmetic_ok = False

        draft = InvoiceDraft(line_items=[item])
        reconcile(draft)

        codes = [warning.code for warning in draft.warnings]
        assert WarningCodes.ARITHMETIC_MISMATCH not in codes

    def test_an_unresolved_row_reports_one_exactly_once(self):
        item = row(quantity=("2", 2), unit_price=("7", 7.0), total=("9999999", 9999999.0))

        draft = InvoiceDraft(line_items=[item])
        reconcile(draft)
        reconcile(draft)  # Re-running must not accumulate duplicates.

        codes = [warning.code for warning in draft.warnings]
        assert codes.count(WarningCodes.ARITHMETIC_MISMATCH) == 1
