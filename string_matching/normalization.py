"""Pre-processing applied to every string before it reaches a matcher.

Matching Arabic OCR output against a catalog only works if both sides are first
reduced to the same canonical form. These rules are the only implementation of
them: the desktop app used to mirror them in
``InvoiceDigitizationApp/Helpers/TextNormalizer.cs`` so it could re-match locally,
and that copy is deleted. Nothing here has to be kept in step with a second
version any more, which is the point -- the two drifted, invisibly, and a
merchant the service had matched came back unmatched in the app.

Three separate normalizations live here because they have genuinely different
rules:

* :func:`normalize_text`   -- for names (merchants, products, cities)
* :func:`normalize_quantity` / :func:`normalize_price` -- for numeric cells
* :func:`normalize_date`   -- for date cells

Nothing here is ever stored as the user-visible value. Normalization is for
comparison and parsing only; the raw OCR text always travels alongside so the
verification screen can show what the paper actually said.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date

# --------------------------------------------------------------------------
# Digits
# --------------------------------------------------------------------------

# Arabic-Indic (U+0660-U+0669) and extended/Persian Arabic-Indic (U+06F0-U+06F9).
_ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
_EXTENDED_ARABIC_INDIC = "۰۱۲۳۴۵۶۷۸۹"

_DIGIT_MAP = {
    ord(ch): str(index)
    for group in (_ARABIC_INDIC, _EXTENDED_ARABIC_INDIC)
    for index, ch in enumerate(group)
}


def fold_digits(text: str) -> str:
    """Arabic-Indic digits to ASCII, leaving every other character alone."""
    return text.translate(_DIGIT_MAP)


# The VLM recognizers translate a handwritten digit into the Latin letter it
# looks like. Only the three that were actually measured on this project's
# invoices are folded -- adding more on a hunch would corrupt product names,
# which pass through the same normalizer.
_LATIN_LOOKALIKES = str.maketrans({"O": "5", "o": "5", "V": "7", "v": "7", "A": "8"})


def fold_latin_lookalikes(text: str | None) -> str:
    """Undo the three documented digit-to-Latin-letter mis-readings."""
    return (text or "").translate(_LATIN_LOOKALIKES)


def digit_sequence(text: str | None) -> str:
    """Every digit in `text`, in order, with nothing else.

    The reading a numeric cell is most likely to have got *right*: these models
    misplace separators far more often than they misread a digit, so the bare
    sequence is what the arithmetic reconciliation searches decimal positions
    over. See :mod:`invoice.reconciliation`.
    """
    return re.sub(r"\D", "", fold_latin_lookalikes(fold_digits(text or "")))


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------

# Harakat (U+064B-U+0652) plus the tatweel/kashida (U+0640). Written as an
# explicit range because combining marks are invisible in an editor and are
# easily mangled by a copy-paste. The range deliberately stops above U+064A
# (yeh), which is a real letter and must survive.
_DIACRITICS = re.compile("[ً-ْـ]")

# Letter forms that vary freely between writers for the same word.
_LETTER_FOLD = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ة": "ه",
    }
)

# Punctuation, Arabic and Latin alike. Digits are deliberately NOT stripped:
# they are part of real product names ("سكر 1كغ", "A4 Paper").
_PUNCTUATION = re.compile(r"[.,;:!؟?،؛(){}\[\]<>\"'`«»/\\|_\-+*=&^%#@~$]")

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str | None) -> str:
    """Fold a name to the canonical form used for every similarity comparison.

    Applies, in order: NFKC (collapses presentation forms such as ﻻ back to
    their canonical letters), Arabic-Indic digits to ASCII, diacritic and
    tatweel removal, alef/yeh/teh-marbuta folding, punctuation removal,
    whitespace collapse, and case folding.
    """
    if not text:
        return ""

    result = unicodedata.normalize("NFKC", text)
    result = fold_digits(result)
    result = _DIACRITICS.sub("", result)
    result = result.translate(_LETTER_FOLD)
    result = _PUNCTUATION.sub(" ", result)
    result = _WHITESPACE.sub(" ", result).strip()
    return result.casefold()


def normalize_words(text: str | None) -> list[str]:
    """:func:`normalize_text` split into words, for order-independent matching."""
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------

# Currency markers that sit next to amounts on regional invoices and must not
# be read as part of the number.
_CURRENCY = re.compile(
    r"(?:ر\.س|ر\.ع|د\.أ|د\.ك|ج\.م|ل\.س|SAR|AED|JOD|KWD|EGP|SYP|USD|"
    r"ريال|درهم|دينار|جنيه|ليره|ليرة|\$|€|£)",
    re.IGNORECASE,
)

_NON_DIGIT = re.compile(r"[^0-9]")


def _clean_numeric_text(text: str) -> str:
    """Shared pre-cleaning: NFKC, fold digits, drop currency, unify separators."""
    cleaned = fold_digits(unicodedata.normalize("NFKC", text))
    cleaned = _CURRENCY.sub(" ", cleaned)
    # Arabic decimal separator U+066B and thousands separator U+066C.
    return cleaned.replace("٫", ".").replace("٬", ",")


def normalize_quantity(text: str | None) -> int | None:
    """Read a quantity cell as a whole number.

    Quantities on these invoices are always integers, so everything that is not
    a digit is discarded rather than interpreted. This is deliberate and is the
    rule the AI team specified: a comma or a dot in a handwritten quantity is
    almost always a mis-read stroke, not a decimal point, and the digit it was
    printed next to is nearly always a zero -- so ``٢.٠`` and ``٢،٠`` both read
    as ``20``, not ``2``.

    Returns None when the cell carries no digit at all.
    """
    if not text:
        return None

    digits = _NON_DIGIT.sub("", _clean_numeric_text(text))
    if not digits:
        return None

    try:
        return int(digits)
    except ValueError:  # pragma: no cover - unreachable while the regex holds
        return None


def normalize_price(text: str | None) -> float | None:
    """Read a price cell as a decimal number.

    Unlike a quantity, a price genuinely has a fractional part, so separators
    are resolved rather than stripped: the rightmost separator is the decimal
    point (a thousands separator is never last) and every other one is dropped.
    """
    if not text:
        return None

    cleaned = _clean_numeric_text(text)

    match = re.search(r"-?\d[\d,.\s]*", cleaned)
    if not match:
        return None

    candidate = _resolve_separators(match.group(0).strip().replace(" ", ""))

    try:
        return float(candidate)
    except ValueError:
        return None


def parse_number_strict(text: str | None) -> float | None:
    """Parse text that is *entirely* a number, else None.

    Distinct from :func:`normalize_price`, which searches for a number anywhere
    in the string. Use this to answer "is this token a numeric cell?" -- a
    product name like ``سكر 1كغ`` or ``A4 Paper`` must not be classified as the
    number 1 or 4, which would both shift the column mapping and erase the
    product name from the row.
    """
    if not text:
        return None

    cleaned = _clean_numeric_text(text).strip()
    if not cleaned:
        return None

    if not re.fullmatch(r"-?\d[\d,.\s]*", cleaned):
        return None

    candidate = _resolve_separators(cleaned.replace(" ", ""))

    try:
        return float(candidate)
    except ValueError:
        return None


def _resolve_separators(value: str) -> str:
    """Decide which of ``.`` and ``,`` is the decimal point.

    OCR reads "1,234.50" and "1.234,50" for the same amount depending on how the
    invoice was printed. The rightmost separator wins.
    """
    last_dot = value.rfind(".")
    last_comma = value.rfind(",")

    if last_dot == -1 and last_comma == -1:
        return value

    if last_comma > last_dot:
        # The comma is the decimal point: drop dots, then swap it for one.
        return value.replace(".", "").replace(",", ".")

    result = value.replace(",", "")

    # "1.234.500" -- more than one dot means they are all thousands separators.
    if result.count(".") > 1:
        result = result.replace(".", "")

    return result


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------

# Any separator a writer might use between date components.
_DATE_SEPARATORS = re.compile(r"[-.\\_\s]+")


def normalize_date_text(text: str | None) -> str:
    """Canonicalize a date cell's *text* to ``DD/MM/YYYY``-shaped input.

    Folds Arabic-Indic digits, rewrites every separator to ``/`` and removes the
    whitespace writers leave around the slashes. This does not validate the date
    -- :func:`parse_date` does that -- it only makes the string parseable.
    """
    if not text:
        return ""

    cleaned = fold_digits(unicodedata.normalize("NFKC", text)).strip()
    cleaned = _DATE_SEPARATORS.sub("/", cleaned)
    # Collapse the runs a stray separator can leave behind ("14//03///2026").
    cleaned = re.sub(r"/{2,}", "/", cleaned)
    return cleaned.strip("/")


# Ordered most-specific first, so an unambiguous ISO date is never read as D/M/Y.
_DATE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})"), "ymd"),
    (re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})"), "dmy"),
    (re.compile(r"(\d{1,2})/(\d{1,2})/(\d{2})"), "dmy2"),
]


def parse_date(text: str | None) -> date | None:
    """Find a date in OCR text and return it, or None.

    Day/month order is ambiguous in ``03/04/2026``. This assumes **day first**,
    the regional convention. When one component exceeds 12 the order is
    unambiguous and is corrected automatically.
    """
    if not text:
        return None

    cleaned = normalize_date_text(text)

    for pattern, order in _DATE_PATTERNS:
        match = pattern.search(cleaned)
        if not match:
            continue

        a, b, c = (int(group) for group in match.groups())

        if order == "ymd":
            year, month, day = a, b, c
        else:
            day, month = a, b
            year = c if order == "dmy" else 2000 + c

            # A "day" over 12 must actually be the month.
            if month > 12 and day <= 12:
                day, month = month, day

        try:
            return date(year, month, day)
        except ValueError:
            continue  # e.g. 31/02 -- keep looking.

    return None


def format_date(value: date | None) -> str | None:
    """ISO-8601, the only date format the API contract emits."""
    return value.isoformat() if value else None
