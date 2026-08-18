"""Square a detector's boxes up against a declared column layout.

Extracted from `HybridSuryaQwenEngine`'s `_split_box`, `_fill_missing`,
`_boundaries` and `_kind_for`. It does three things a raw detector cannot:

1. **Split.** Handwriting that runs together makes the detector draw one box
   across several columns. Cutting it at the declared boundaries recovers the
   individual values.
2. **Fill.** The detector misses short isolated digits entirely -- a quantity of
   "٢" is often never boxed. For a row where other columns *were* found, a box
   is synthesized at the missing column so the recognizer at least looks there.
3. **Type.** A region's horizontal position tells you what kind of characters it
   holds, which selects the recognition prompt. That single change suppressed
   the digit-to-Latin confusion (٥->O, ٧->V, ٨->A) entirely.

**This whole class is a workaround for not knowing where the columns are.** The
column boundaries are declared as fractions of the page width -- 0.163, 0.285,
0.775 -- which are a property of one printed form measured once by hand, and
which are wrong the moment the page is cropped differently. The layout-driven
flow does not use it: there, the detected grid supplies the real boundaries.
Prefer that flow where the form is ruled; this one earns its place when text
sits outside any printed box.
"""

from __future__ import annotations

from typing import Sequence

from core.domain.geometry import BoundingBox
from core.domain.ocr import TextRegion
from core.domain.roles import ContentKind
from ocr.refiners.registry import refiner_registry


@refiner_registry.register("column_fraction")
class ColumnFractionRefiner:
    name = "column_fraction"

    def __init__(
        self,
        column_kinds: list[dict] | None = None,
        split_on_columns: bool = True,
        min_split_width: int = 45,
        split_inset: int = 10,
        fill_missing_cells: bool = True,
        row_tolerance: float = 0.35,
        default_kind: str = ContentKind.ANY.value,
        **params,
    ):
        self.columns = list(column_kinds or [])
        for c in self.columns:
            # Fail here rather than mid-page: an unknown kind would silently
            # fall back to the generic prompt for every cell in that column.
            ContentKind(c["kind"])
            if not 0.0 <= c["from"] < c["to"] <= 1.0:
                raise ValueError(
                    f"column_fraction: column {c!r} must satisfy 0 <= from < to <= 1."
                )

        self.split_on_columns = split_on_columns
        self.min_split_width = min_split_width
        self.split_inset = split_inset
        self.fill_missing_cells = fill_missing_cells
        self.row_tolerance = row_tolerance
        self.default_kind = ContentKind(default_kind)
        self.params = params

    # ------------------------------------------------------------------

    def refine(
        self, regions: Sequence[TextRegion], width: int, height: int
    ) -> list[TextRegion]:
        split: list[TextRegion] = []
        for region in regions:
            for part in self._split(region, width):
                split.append(part)

        filled = split + self._missing(split, width)

        # Reading order: top to bottom, then right to left. The y is rounded
        # into bands so two boxes on the same printed line are not ordered by a
        # few pixels of vertical jitter.
        filled.sort(key=lambda r: (round(r.bbox.y / 20), -r.bbox.x))
        return [self._typed(r, width) for r in filled]

    # ------------------------------------------------------------------

    def _boundaries(self, width: int) -> list[int]:
        """Interior column boundaries in pixels, sorted."""
        edges: set[int] = set()
        for c in self.columns:
            for frac in (c["from"], c["to"]):
                if 0.0 < frac < 1.0:
                    edges.add(int(round(frac * width)))
        return sorted(edges)

    def _split(self, region: TextRegion, width: int) -> list[TextRegion]:
        box = region.bbox
        x1, x2 = box.x, box.x + box.w

        if not self.split_on_columns or not self.columns:
            return [region]

        m = self.min_split_width
        cuts = [b for b in self._boundaries(width) if x1 + m < b < x2 - m]
        if not cuts:
            return [region]

        parts: list[TextRegion] = []
        prev = x1
        ins = self.split_inset
        for i, b in enumerate(cuts + [x2]):
            if b - prev >= m:
                # Inset away from the boundary so a piece does not carry the
                # neighbouring column's printed rule into its crop.
                left = prev + ins if i > 0 else prev
                right = b - ins if b != x2 else b
                if right - left >= m:
                    parts.append(
                        TextRegion(
                            bbox=BoundingBox(x=left, y=box.y, w=right - left, h=box.h),
                            confidence=region.confidence,
                            content_kind=region.content_kind,
                            source_id=region.source_id,
                        )
                    )
            prev = b

        return parts or [region]

    def _missing(self, regions: list[TextRegion], width: int) -> list[TextRegion]:
        """Synthesize a region for each column a detected row left empty.

        Only for rows that already have at least two detections: one lone box is
        as likely to be a stray mark as a row, and inventing five neighbours for
        it would cost five model calls to read blank paper.
        """
        if not self.fill_missing_cells or not self.columns:
            return []

        lo = min(c["from"] for c in self.columns) * width
        hi = max(c["to"] for c in self.columns) * width
        inside = [r for r in regions if lo <= r.bbox.x + r.bbox.w / 2.0 <= hi]
        if not inside:
            return []

        added: list[TextRegion] = []
        for row in self._rows(inside):
            if len(row) < 2:
                continue

            y1 = min(r.bbox.y for r in row)
            y2 = max(r.bbox.y + r.bbox.h for r in row)

            for c in self.columns:
                cx1, cx2 = int(c["from"] * width), int(c["to"] * width)
                covered = any(cx1 < r.bbox.x + r.bbox.w / 2.0 < cx2 for r in row)
                if covered:
                    continue

                x1 = cx1 + self.split_inset
                x2 = cx2 - self.split_inset
                if x2 - x1 >= self.min_split_width:
                    added.append(
                        TextRegion(
                            bbox=BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1),
                            # Synthesized, so there is no detection score to
                            # report. None means unknown, which is the truth.
                            confidence=None,
                            content_kind=ContentKind(c["kind"]),
                        )
                    )

        return added

    def _rows(self, regions: list[TextRegion]) -> list[list[TextRegion]]:
        ordered = sorted(regions, key=lambda r: r.bbox.y + r.bbox.h / 2.0)
        rows: list[list[TextRegion]] = []
        current = [ordered[0]]

        for r in ordered[1:]:
            prev = current[-1].bbox
            tol = self.row_tolerance * max(prev.h, r.bbox.h)
            if abs((r.bbox.y + r.bbox.h / 2.0) - (prev.y + prev.h / 2.0)) <= tol:
                current.append(r)
            else:
                rows.append(current)
                current = [r]

        rows.append(current)
        return rows

    def _typed(self, region: TextRegion, width: int) -> TextRegion:
        """Attach the content kind implied by the region's column."""
        if not self.columns:
            return region
        if region.content_kind is not ContentKind.ANY:
            return region  # a synthesized region already knows its column

        frac = (region.bbox.x + region.bbox.w / 2.0) / max(1, width)
        kind = next(
            (ContentKind(c["kind"]) for c in self.columns if c["from"] <= frac < c["to"]),
            self.default_kind,
        )
        return TextRegion(
            bbox=region.bbox,
            confidence=region.confidence,
            content_kind=kind,
            source_id=region.source_id,
        )
