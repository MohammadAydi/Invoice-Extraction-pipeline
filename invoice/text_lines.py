"""Reconstructing text lines from loose OCR fragments.

Every OCR engine here returns fragments with a box and no notion of which line
they belong to. The header parser reads labelled lines ("التاريخ: ..."), so the
lines have to be rebuilt from geometry first.

Kept separate from the parser because the table path does not need it: when the
table extractor found a grid, rows come from the grid and this module is only
used for the header region above it.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from core.domain.geometry import BoundingBox
from core.domain.ocr import OCRFragment
from string_matching.normalization import normalize_text, parse_number_strict


@dataclass(frozen=True)
class Word:
    """One OCR fragment, with the positional accessors line-building needs."""

    text: str
    bbox: BoundingBox
    confidence: float

    @classmethod
    def from_fragment(cls, fragment: OCRFragment) -> "Word":
        # Detector-only engines report no recognition confidence and hand back
        # None. Treating that as 0.0 would flag every field on the page for
        # review; 0.5 says "unknown", which is what it is.
        confidence = fragment.confidence
        return cls(
            text=fragment.text,
            bbox=fragment.bbox,
            confidence=0.5 if confidence is None else float(confidence),
        )

    @property
    def center_x(self) -> float:
        return self.bbox.x + self.bbox.w / 2.0

    @property
    def center_y(self) -> float:
        return self.bbox.y + self.bbox.h / 2.0

    @property
    def height(self) -> int:
        return self.bbox.h


@dataclass
class TextLine:
    """A horizontal band of words, reconstructed from their positions."""

    words: list[Word]

    @property
    def text(self) -> str:
        # Right to left: this service is Arabic-primary.
        return " ".join(w.text for w in sorted(self.words, key=lambda w: -w.center_x))

    @property
    def ltr_text(self) -> str:
        """Left-to-right reading, for lines that are mostly Latin or numeric."""
        return " ".join(w.text for w in sorted(self.words, key=lambda w: w.center_x))

    @property
    def normalized(self) -> str:
        return normalize_text(self.text)

    @property
    def confidence(self) -> float:
        return (
            sum(w.confidence for w in self.words) / len(self.words) if self.words else 0.0
        )

    @property
    def top(self) -> float:
        return min(w.bbox.y for w in self.words) if self.words else 0.0

    @property
    def bbox(self) -> BoundingBox | None:
        if not self.words:
            return None

        x1 = min(w.bbox.x for w in self.words)
        y1 = min(w.bbox.y for w in self.words)
        x2 = max(w.bbox.x + w.bbox.w for w in self.words)
        y2 = max(w.bbox.y + w.bbox.h for w in self.words)
        return BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)

    def numbers(self) -> list[float]:
        """Every wholly-numeric token on the line, in left-to-right order.

        Strict parsing matters here: "سكر 1كغ" is a product name, not the
        number 1, and treating it as numeric would both shift the column
        mapping and erase the product name from the row.
        """
        found: list[float] = []
        for word in sorted(self.words, key=lambda w: w.center_x):
            value = parse_number_strict(word.text)
            if value is not None:
                found.append(value)
        return found

    def has_any(self, hints: tuple[str, ...]) -> bool:
        norm = self.normalized
        return any(hint in norm for hint in hints)

    def segments(self, max_words: int = 6) -> list[str]:
        """Contiguous runs of words within the line, longest first.

        Invoice letterheads commonly put the merchant name at one side of a line
        and the invoice number at the other, so the *line* text is often two
        unrelated fields concatenated. Matching against runs recovers the
        merchant from such a line instead of scoring the whole merged string.
        """
        ordered = sorted(self.words, key=lambda w: -w.center_x)
        results: list[str] = []

        for size in range(min(max_words, len(ordered)), 0, -1):
            for start in range(len(ordered) - size + 1):
                text = " ".join(w.text for w in ordered[start : start + size]).strip()
                if text:
                    results.append(text)

        return results


def group_lines(fragments: list[OCRFragment]) -> list[TextLine]:
    """Cluster fragments into lines by vertical position, top line first.

    The tolerance is a fraction of the median glyph height rather than a fixed
    pixel count, so the same code works on a 900px phone photo and a 3000px
    flatbed scan.
    """
    words = [Word.from_fragment(f) for f in fragments if (f.text or "").strip()]
    if not words:
        return []

    median_height = statistics.median(w.height for w in words) or 10.0
    tolerance = median_height * 0.6

    lines: list[list[Word]] = []
    for word in sorted(words, key=lambda w: w.center_y):
        if lines:
            last_center = sum(w.center_y for w in lines[-1]) / len(lines[-1])
            if abs(word.center_y - last_center) <= tolerance:
                lines[-1].append(word)
                continue
        lines.append([word])

    return [TextLine(words=line) for line in lines]
