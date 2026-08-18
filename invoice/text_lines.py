"""Reconstructing text lines from loose OCR fragments.

The `Word` and `TextLine` models this builds live in `core.domain.text_lines`
with the rest of the domain; what stays here is the clustering itself, which
is an algorithm and not a model.

Only the header path needs it. When the table extractor found a grid, rows
come from the grid; this rebuilds lines for the region above it, and for the
borderless documents that have no grid at all.
"""

from __future__ import annotations

import statistics

from core.domain.ocr import OCRFragment
from core.domain.text_lines import TextLine, Word

__all__ = ["TextLine", "Word", "group_lines"]


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
