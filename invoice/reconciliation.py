"""Arithmetic reconciliation of a parsed invoice's line items.

The recognizers on this project misplace the decimal separator far more often
than they misread a digit: a cell photographed across a printed dot pattern
comes back as ``"٥-١-٧,٦,٥"`` whose digit sequence, ``51765``, is exactly right.
So instead of trusting the separator, this stage tries every legal decimal
position and keeps the one combination that satisfies

    unit price x quantity = line total

That single equation both repairs the separators and verifies the reading in one
step, and it fills in a missing cell from the two that were read.

Two safety rules, each of which was the difference between a useful correction
and a confidently wrong number:

* A computed value replaces a *read* one only when the two are close --
  :func:`_explains_reading`. A derived value that looks nothing like what the
  model saw means the model read a different cell, not a mis-scaled one.
* Nothing here ever invents a value out of one number. Two knowns are the
  minimum, and a value derived from two unconfirmed cells is marked for review
  rather than accepted.

This runs after :class:`~invoice.parser.InvoiceParser`, on the draft it
produced, and is called once from
:meth:`~orchestration.pipeline_orchestrator.PipelineOrchestrator.run`. It
replaced the standalone ``postprocess/reconcile.py`` CLI, which reimplemented
its own digit normalization and never ran inside the pipeline at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.domain.invoice import (
    InvoiceDraft,
    InvoiceLineItem,
    InvoiceWarning,
    WarningCodes,
)
from string_matching.normalization import digit_sequence, fold_digits, fold_latin_lookalikes

# Slack for the decimal division, in currency units.
TOLERANCE = 0.02

# Amounts on these invoices carry at most two decimals, so the separator can sit
# in one of three places.
MAX_DECIMALS = 2

# How different a derived value may be from what was read, as a fraction of the
# longer digit sequence, before it stops being an explanation of that reading.
MAX_ERROR_RATE = 0.50

_SEPARATORS = re.compile(r"[,.٫،\-_/\\|]")


@dataclass
class _Cell:
    """One numeric cell of a row, as reconciliation sees it."""

    name: str
    raw: str
    digits: str
    value: float | None

    @property
    def known(self) -> bool:
        return bool(self.digits)


# --------------------------------------------------------------------------
# Candidate generation
# --------------------------------------------------------------------------


def decimal_candidates(digits: str, max_decimals: int = MAX_DECIMALS) -> list[float]:
    """Every value the digit sequence could be, by decimal position."""
    if not digits:
        return []

    values: list[float] = []
    for places in range(0, max_decimals + 1):
        if places == 0:
            candidate = float(digits)
        elif len(digits) > places:
            candidate = float(f"{digits[:-places]}.{digits[-places:]}")
        else:
            continue

        if candidate not in values:
            values.append(candidate)

    return values


def dot_run_candidates(raw: str | None) -> list[str]:
    """Digit sequences for a cell whose reading ends in a run of dots.

    These models write trailing zeros as dots: ``"212..."`` is 212000 and
    ``"١٢.."`` is 1200. The count does not reliably match the number of zeros,
    so every plausible expansion is offered and the equation picks.
    """
    digits = digit_sequence(raw)
    if not digits:
        return []

    tail = re.search(r"[.…]+\s*$", (raw or "").strip())
    if not tail:
        return [digits]

    dots = len(tail.group(0).replace(" ", ""))

    # One past the dot count: a three-zero amount is sometimes abbreviated to
    # two dots, so the run length is a hint rather than a count.
    sequences = [digits]
    for zeros in range(1, dots + 2):
        expanded = digits + "0" * zeros
        if expanded not in sequences:
            sequences.append(expanded)

    return sequences


def _pool(cell: _Cell, integral: bool) -> list[float]:
    """Every value this cell could hold, best-guess order.

    `integral` for the quantity column: a quantity is a whole number by the
    project's normalization rules, so there is no separator to search for.
    """
    sequences = dot_run_candidates(cell.raw) or ([cell.digits] if cell.digits else [])

    if integral:
        return [float(sequence) for sequence in sequences if sequence]

    values: list[float] = []
    for sequence in sequences:
        for value in decimal_candidates(sequence):
            if value not in values:
                values.append(value)

    return values


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------


def observed_decimals(raw: str | None) -> int | None:
    """How many digits followed the separator in the raw reading, if legible.

    The *position* of a separator is a clearer visual mark than the shape of a
    digit, so the models get it right more often than they get the digits right
    -- which is what disambiguates 517.65 from 5176.5 when both satisfy the
    equation. A reading with several separators is dot-pattern noise, not
    formatting, and its signal is discarded rather than trusted.
    """
    text = fold_latin_lookalikes(fold_digits(raw or ""))
    parts = [part for part in _SEPARATORS.split(text) if part.strip()]

    if len(parts) != 2:
        return None

    tail = re.sub(r"\D", "", parts[-1])
    return len(tail) if 1 <= len(tail) <= 2 else None


def _separator_target(raw: str | None) -> float | None:
    """The value the reading claims when its separator is taken at face value.

    Built from the digit sequence plus the separator position rather than by
    parsing, so it survives the noise around the number -- and so that a printed
    "150.00" claims 150.0 rather than the 0-decimal 150 that stripping trailing
    zeros would suggest.
    """
    wanted = observed_decimals(raw)
    if wanted is None:
        return None

    digits = digit_sequence(raw)
    if not digits:
        return None

    return int(digits) / (10**wanted)


def _matches_separator(value: float, raw: str | None) -> bool | None:
    """Whether `value` sits where the reading's separator says. None = no signal."""
    target = _separator_target(raw)
    if target is None:
        return None
    return abs(value - target) <= 1e-9


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current

    return previous[-1]


def _explains_reading(derived: float, observed: str) -> bool:
    """Whether `derived` is a plausible correction of what was actually read."""
    if not observed:
        return False

    digits = re.sub(r"\D", "", f"{derived:.2f}".rstrip("0").rstrip("."))
    span = max(len(digits), len(observed))
    return _levenshtein(digits, observed) / span <= MAX_ERROR_RATE


def _pick_by_separator(options: list[float], raw: str | None) -> float | None:
    """Break a tie between same-sequence values using the separator's position."""
    unique = sorted(set(options))
    if len(unique) == 1:
        return unique[0]

    target = _separator_target(raw)
    if target is None:
        return None

    matched = [value for value in unique if abs(value - target) <= 1e-9]
    return matched[0] if len(matched) == 1 else None


def _settled_value(cell: _Cell, pool: list[float]) -> float | None:
    """The one value this cell can be read as, or None if it is ambiguous.

    The parser's own reading wins when it had one -- it already applied the
    number rules to a cell those rules accepted. Only a cell they refused falls
    back to the separator evidence.
    """
    if cell.value is not None:
        return cell.value
    return _pick_by_separator(pool, cell.raw)


def _solve_missing(
    missing: str,
    prices: list[float],
    quantities: list[float],
    totals: list[float],
) -> set[float]:
    if missing == "total_price":
        return {round(p * q, 2) for p in prices for q in quantities}
    if missing == "unit_price":
        return {round(t / q, 2) for q in quantities for t in totals if q}
    if missing == "quantity":
        return {round(t / p, 2) for p in prices for t in totals if p}
    return set()


# --------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------


def reconcile(draft: InvoiceDraft) -> InvoiceDraft:
    """Repair each line's numbers against ``price x quantity = total``.

    Mutates and returns `draft`. Rows the equation cannot settle are left
    exactly as the parser read them: an unresolved row belongs on the
    verification screen, and overwriting it with a guess is what this stage
    exists to avoid.
    """
    for item in draft.line_items:
        _reconcile_row(item)

    _refresh_arithmetic_warnings(draft)
    return draft


def _refresh_arithmetic_warnings(draft: InvoiceDraft) -> None:
    """Re-issue ARITHMETIC_MISMATCH against the reconciled numbers.

    The parser raised those warnings against the rows as they were read. A row
    this stage repaired is no longer mismatched, and leaving the old warning
    standing would put a contradiction into the run's own record.
    """
    draft.warnings = [
        warning
        for warning in draft.warnings
        if warning.code != WarningCodes.ARITHMETIC_MISMATCH
    ]

    for item in draft.line_items:
        if item.arithmetic_ok:
            continue

        draft.warnings.append(
            InvoiceWarning(
                code=WarningCodes.ARITHMETIC_MISMATCH,
                field=f"line_items[{item.row_index}]",
                message=(
                    f"Row {item.row_index + 1}: quantity x unit price does not "
                    "equal the line total, and no decimal placement reconciles them."
                ),
            )
        )


def _cells(item: InvoiceLineItem) -> dict[str, _Cell]:
    def cell(name: str, field) -> _Cell:
        raw = field.raw if field.raw is not None else field.value
        text = "" if raw is None else str(raw)
        return _Cell(
            name=name,
            raw=text,
            digits=digit_sequence(text),
            value=float(field.value) if isinstance(field.value, (int, float)) else None,
        )

    return {
        "unit_price": cell("unit_price", item.unit_price),
        "quantity": cell("quantity", item.quantity),
        "total_price": cell("total_price", item.total_price),
    }


def _write_back(item: InvoiceLineItem, name: str, value: float) -> None:
    field = getattr(item, name)

    # Quantity stays an integer, matching normalize_quantity: a fractional
    # quantity here is a division artefact, not something the paper said.
    field.value = int(round(value)) if name == "quantity" else round(value, 2)


def _reconcile_row(item: InvoiceLineItem) -> None:
    cells = _cells(item)
    known = {name: cell for name, cell in cells.items() if cell.known}

    pools = {
        name: _pool(cell, integral=(name == "quantity")) for name, cell in cells.items()
    }

    if len(known) == 3:
        _reconcile_complete_row(item, cells, pools)
    elif len(known) == 2:
        _derive_missing_cell(item, cells, pools)

    item.arithmetic_ok = _row_is_consistent(item)


def _reconcile_complete_row(item, cells: dict[str, _Cell], pools) -> None:
    # A row whose own readings already multiply out needs nothing. Searching one
    # anyway would let a differently-scaled combination that also satisfies the
    # equation replace three correct numbers.
    if _row_multiplies_out(item):
        return

    combinations = [
        (price, quantity, total)
        for price in pools["unit_price"]
        for quantity in pools["quantity"]
        for total in pools["total_price"]
        if abs(price * quantity - total) <= TOLERANCE
    ]

    if not combinations:
        _repair_least_trusted_cell(item, cells, pools)
        return

    if len(combinations) > 1:
        chosen = _best_scaled(combinations, cells)
        if chosen is None:
            return
        combinations = [chosen]

    price, quantity, _ = combinations[0]

    # The product, not the read total: the match was accepted within tolerance,
    # so the read total may carry a stray digit the multiplication does not.
    _write_back(item, "unit_price", price)
    _write_back(item, "quantity", quantity)
    _write_back(item, "total_price", round(price * quantity, 2))


def _best_scaled(combinations, cells: dict[str, _Cell]):
    """Pick among combinations the equation cannot separate.

    5.95 x 87, 59.5 x 87 and 595 x 87 are all internally consistent; the
    equation fixes the ratios and says nothing about the scale. Two independent
    signals break the tie, and each cell contributes exactly one of them:

    * A cell whose reading carries **one** legible separator has already said
      where the point goes, and that is the stronger evidence.
    * A cell with no separator signal can only offer its literal reading -- the
      bare digit sequence, unscaled. "13250 x 16 = 212000" beats
      "1325 x 16 = 21200" on that basis.

    A cell never votes twice. Counting a legible cell's bare-digit reading as
    well would have it argue against its own separator and cancel out.
    """
    scored = []

    for price, quantity, total in combinations:
        score = 0

        for name, value in (
            ("unit_price", price),
            ("quantity", quantity),
            ("total_price", total),
        ):
            cell = cells[name]
            agrees = _matches_separator(value, cell.raw)

            if agrees is not None:
                score += 1 if agrees else -1
            elif cell.digits and abs(value - float(cell.digits)) <= TOLERANCE:
                score += 1

        scored.append((score, (price, quantity, total)))

    best = max(score for score, _ in scored)
    winners = [combo for score, combo in scored if score == best]

    # Ambiguous and unrankable: leave the row exactly as it was read. A coin
    # flip between two scales is the one case where the user must decide.
    return winners[0] if best > 0 and len(winners) == 1 else None


def _repair_least_trusted_cell(item, cells: dict[str, _Cell], pools) -> None:
    """No combination works, so re-derive the least trustworthy cell.

    Long digit runs are where these models fail and short ones are where they
    succeed, so the cell with the longest sequence is the first suspect, broken
    by how clean its crop was.
    """
    known = {name: cell for name, cell in cells.items() if cell.known}
    suspect = max(known, key=lambda name: (len(known[name].digits), -_purity(known[name])))

    remaining = {name: values for name, values in pools.items() if name != suspect}
    options = _solve_missing(
        suspect,
        remaining.get("unit_price", []),
        remaining.get("quantity", []),
        remaining.get("total_price", []),
    )

    viable = [value for value in options if _explains_reading(value, known[suspect].digits)]
    chosen = _pick_by_separator(viable, known[suspect].raw) if viable else None

    if chosen is not None:
        _write_back(item, suspect, chosen)


def _derive_missing_cell(item, cells: dict[str, _Cell], pools) -> None:
    """Fill the one unread cell from the two that were read.

    Both sources have to settle on a single value first: multiplying two
    ambiguous readings together would produce a whole spread of "answers" and
    then pick one, which is guessing dressed as arithmetic.

    The result is mathematically sound and factually unconfirmed -- neither
    source cell was verified by anything. It is offered so the row is not left
    blank; the verification screen is where it gets confirmed.
    """
    missing = next(
        name
        for name in ("unit_price", "quantity", "total_price")
        if not cells[name].known
    )

    known = {
        name: _settled_value(cell, pools[name])
        for name, cell in cells.items()
        if name != missing
    }

    if any(value is None for value in known.values()):
        return

    options = _solve_missing(
        missing,
        [known["unit_price"]] if "unit_price" in known else [],
        [known["quantity"]] if "quantity" in known else [],
        [known["total_price"]] if "total_price" in known else [],
    )

    if len(options) == 1:
        _write_back(item, missing, options.pop())


def _purity(cell: _Cell) -> float:
    """Digits as a fraction of the raw reading -- a clean crop approaches 1.0."""
    raw = cell.raw.strip()
    return len(cell.digits) / len(raw) if raw else 0.0


def _row_multiplies_out(item: InvoiceLineItem) -> bool:
    """All three numbers present, and the equation already holds."""
    values = [item.quantity.value, item.unit_price.value, item.total_price.value]
    if any(value is None for value in values):
        return False

    quantity, price, total = (float(value) for value in values)
    return abs(quantity * price - total) <= TOLERANCE


def _row_is_consistent(item: InvoiceLineItem) -> bool:
    """What `arithmetic_ok` reports.

    An incomplete row counts as consistent: there is no disagreement to flag
    when one of the three numbers was never read, and reporting one would put an
    amber highlight on every blank cell of an unfilled form.
    """
    values = [item.quantity.value, item.unit_price.value, item.total_price.value]
    if any(value is None for value in values):
        return True

    return _row_multiplies_out(item)
