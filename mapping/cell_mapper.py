# """Maps raw OCR fragments onto the page's labelled regions.
#
# Because geometric correction runs once, upstream of both the OCR and table
# branches (see orchestration.PipelineOrchestrator), every bbox coming out of
# OCRResult and InvoiceLayout is already in the same coordinate space -- the
# geometrically corrected display image. No coordinate remapping happens here,
# only assignment.
#
# Two things changed from the version that only knew about table cells:
#
# * It takes an :class:`~core.domain.layout.InvoiceLayout`, not a
#   `TableExtractionResult`, so it maps the **whole page**: the invoice number,
#   city, date and merchant boxes above the table and the totals strip below it
#   are regions like any other. Leaving them out was why header fields had to be
#   re-derived downstream from reconstructed text lines, which cannot see that a
#   value sits inside a ruled box under a printed caption.
#
# * Assignment is by **identity** when the fragment carries a `source_id`. In the
#   layout-driven flow the crop was cut from a specific region, so which region it
#   belongs to is known exactly -- re-deriving it from geometry would be throwing
#   away a fact in order to guess at it.
#
# Fragments with no `source_id` fall back to centroid containment: a fragment
# whose box clips a printed cell divider still belongs to the cell its ink sits
# in, and centroid containment says so without needing an overlap threshold nobody
# can tune.
# """
#
# from __future__ import annotations
#
# from core.domain.document import (
#     DocumentElement,
#     FreeFieldElement,
#     StructuredDocument,
#     TableCellElement,
# )
# from core.domain.geometry import BoundingBox
# from core.domain.layout import InvoiceLayout, LayoutRegion
# from core.domain.ocr import OCRFragment, OCRResult
# from core.domain.roles import CellRole, Zone
#
#
# def _centroid(bbox: BoundingBox) -> tuple[float, float]:
#     return bbox.x + bbox.w / 2.0, bbox.y + bbox.h / 2.0
#
#
# def _contains(box: BoundingBox, x: float, y: float) -> bool:
#     return box.x <= x <= box.x + box.w and box.y <= y <= box.y + box.h
#
#
# def _merge_text(fragments: list[OCRFragment]) -> str:
#     """Join a region's fragments in Arabic reading order: top to bottom, right to left.
#
#     Reading order matters even inside one cell: a two-line description merged
#     left-to-right would come back with its halves swapped, and the string
#     matcher would then score it against the catalog as a different product.
#     """
#     if not fragments:
#         return ""
#
#     ordered = sorted(
#         fragments,
#         key=lambda f: (round(f.bbox.y + f.bbox.h / 2.0), -(f.bbox.x + f.bbox.w)),
#     )
#     return " ".join(f.text.strip() for f in ordered if f.text and f.text.strip()).strip()
#
#
# def _hull(fragments: list[OCRFragment]) -> BoundingBox:
#     """Smallest box covering every fragment, used when a region has no box of its own."""
#     x1 = min(f.bbox.x for f in fragments)
#     y1 = min(f.bbox.y for f in fragments)
#     x2 = max(f.bbox.x + f.bbox.w for f in fragments)
#     y2 = max(f.bbox.y + f.bbox.h for f in fragments)
#     return BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
#
#
# class CellMapper:
#     """Turns (OCR fragments, page layout) into one flat list of UI elements."""
#
#     def map(self, ocr_result: OCRResult, layout: InvoiceLayout) -> StructuredDocument:
#         regions = list(layout.regions)
#
#         # Smallest region first. A layout can carry a whole-table box alongside
#         # its cells, and a fragment inside both belongs to the cell.
#         order = sorted(range(len(regions)), key=lambda i: regions[i].area)
#         by_id = {region.id: i for i, region in enumerate(regions)}
#
#         per_region: dict[int, list[OCRFragment]] = {}
#         unassigned: list[OCRFragment] = []
#
#         for fragment in ocr_result.fragments:
#             if not (fragment.text or "").strip():
#                 continue
#
#             index = self._assign(fragment, regions, order, by_id)
#             if index is None:
#                 unassigned.append(fragment)
#             else:
#                 per_region.setdefault(index, []).append(fragment)
#
#         elements: list[DocumentElement] = []
#
#         # Every region becomes an element, including the ones nothing was read
#         # in: the verification grid needs a box to click on for a quantity the
#         # engine failed to read, and a missing element would silently shift the
#         # row.
#         for index, region in enumerate(regions):
#             elements.append(self._element(region, per_region.get(index, [])))
#
#         for index, fragment in enumerate(unassigned):
#             elements.append(
#                 FreeFieldElement(
#                     id=f"free:{index}",
#                     bbox=fragment.bbox,
#                     fragments=[fragment],
#                     merged_text=fragment.text.strip(),
#                     role=CellRole.UNKNOWN,
#                     zone=Zone.UNKNOWN,
#                 )
#             )
#
#         return StructuredDocument(elements=elements)
#
#     # ------------------------------------------------------------------
#
#     @staticmethod
#     def _assign(
#         fragment: OCRFragment,
#         regions: list[LayoutRegion],
#         order: list[int],
#         by_id: dict[str, int],
#     ) -> int | None:
#         """Which region this fragment belongs to, by identity where known."""
#         if fragment.source_id is not None:
#             # The crop was cut from this region. If the id is unknown the layout
#             # and the fragments came from different runs, which is a bug, not
#             # something to paper over with a geometric guess.
#             return by_id.get(fragment.source_id)
#
#         x, y = _centroid(fragment.bbox)
#         return next(
#             (i for i in order if _contains(regions[i].bbox, x, y)),
#             None,
#         )
#
#     @staticmethod
#     def _element(region: LayoutRegion, fragments: list[OCRFragment]) -> DocumentElement:
#         merged = _merge_text(fragments)
#
#         if region.row is None or region.col is None or region.table_id is None:
#             return FreeFieldElement(
#                 id=region.id,
#                 bbox=region.bbox,
#                 fragments=fragments,
#                 merged_text=merged,
#                 role=region.role,
#                 zone=region.zone,
#             )
#
#         return TableCellElement(
#             id=region.id,
#             table_id=region.table_id,
#             row=region.row,
#             col=region.col,
#             bbox=region.bbox,
#             fragments=fragments,
#             merged_text=merged,
#             role=region.role,
#             zone=region.zone,
#             row_span=region.row_span,
#             col_span=region.col_span,
#         )
#
#     @staticmethod
#     def bounds_of(fragments: list[OCRFragment]) -> BoundingBox | None:
#         """Public helper for callers assembling their own elements from fragments."""
#         return _hull(fragments) if fragments else None
"""Maps raw OCR fragments onto the page's labelled regions.

Because geometric correction runs once, upstream of both the OCR and table
branches (see orchestration.PipelineOrchestrator), every bbox coming out of
OCRResult and InvoiceLayout is already in the same coordinate space -- the
geometrically corrected display image. No coordinate remapping happens here,
only assignment.

This version keeps the external contract identical to the version already
wired into the pipeline -- ``CellMapper().map(ocr_result, layout)`` where
``layout`` is an :class:`~core.domain.layout.InvoiceLayout`. What changed is
*how* a fragment without a known `source_id` gets assigned to a region:

* Assignment is still by **identity** first, when the fragment carries a
  `source_id`. In the layout-driven flow the crop was cut from a specific
  region, so which region it belongs to is known exactly -- re-deriving it
  from geometry would be throwing away a fact in order to guess at it. If the
  `source_id` doesn't resolve to a known region, that's a bug upstream (OCR
  and layout came from different runs) and is NOT silently papered over with
  a geometric guess.

* Fragments with no `source_id` are no longer matched by simple centroid
  containment. They go through Heuristic Scoring (IOA / IOU / center-distance
  closeness) against every region on the page, the same scoring used in the
  standalone table-cell mapper. This is strictly more robust than containment:
  it still picks the containing region in the common case (IOA -> 1.0 when a
  region fully covers a fragment) but also degrades gracefully when a
  fragment's box clips a divider or slightly overshoots its cell.

* A very verbose, visually-structured debug trace is printed at every stage
  (input dump, per-table envelopes, per-fragment scoring, assembly/merge) to
  make the whole pipeline step inspectable from the terminal.
"""

from __future__ import annotations

from typing import Any

from core.domain.document import (
    DocumentElement,
    FreeFieldElement,
    StructuredDocument,
    TableCellElement,
)
from core.domain.geometry import BoundingBox
from core.domain.layout import InvoiceLayout, LayoutRegion
from core.domain.ocr import OCRFragment, OCRResult
from core.domain.roles import CellRole, Zone


def _centroid(bbox: BoundingBox) -> tuple[float, float]:
    return bbox.x + bbox.w / 2.0, bbox.y + bbox.h / 2.0


def _contains(box: BoundingBox, x: float, y: float) -> bool:
    return box.x <= x <= box.x + box.w and box.y <= y <= box.y + box.h


def _hull(fragments: list[OCRFragment]) -> BoundingBox:
    x1 = min(f.bbox.x for f in fragments)
    y1 = min(f.bbox.y for f in fragments)
    x2 = max(f.bbox.x + f.bbox.w for f in fragments)
    y2 = max(f.bbox.y + f.bbox.h for f in fragments)
    return BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)


def _is_arabic_char(ch: str) -> bool:
    return ("\u0600" <= ch <= "\u06FF") or ("\u0750" <= ch <= "\u077F") or ("\u08A0" <= ch <= "\u08FF")


def _is_latin_char(ch: str) -> bool:
    return ch.isascii() and ch.isalpha()


def _detect_text_direction(text: str) -> str:
    if not text:
        return "ltr"
    arabic_count = sum(1 for ch in text if _is_arabic_char(ch))
    latin_count = sum(1 for ch in text if _is_latin_char(ch))
    if arabic_count == 0 and latin_count == 0:
        return "ltr"
    return "rtl" if arabic_count >= latin_count else "ltr"


def merge_fragments_in_cell(fragments: list[OCRFragment], *, region_label: str = "") -> str:
    """يدمج شظايا OCR المتعددة داخل نفس المنطقة إلى نص واحد مرتب مكانيًا."""
    fragments = [f for f in fragments if f.text and f.text.strip()]
    if not fragments:
        return ""
    if len(fragments) == 1:
        return fragments[0].text.strip()

    heights = [f.bbox.h for f in fragments if f.bbox.h > 0]
    avg_h = (sum(heights) / len(heights)) if heights else 1.0
    line_threshold = max(avg_h * 0.6, 1.0)

    frags_sorted = sorted(fragments, key=lambda f: f.bbox.y + f.bbox.h / 2.0)
    lines: list[list[OCRFragment]] = []

    for f in frags_sorted:
        fy = f.bbox.y + f.bbox.h / 2.0
        placed = False
        for line in lines:
            line_y = sum(x.bbox.y + x.bbox.h / 2.0 for x in line) / len(line)
            if abs(fy - line_y) <= line_threshold:
                line.append(f)
                placed = True
                break
        if not placed:
            lines.append([f])

    lines.sort(key=lambda line: sum(x.bbox.y + x.bbox.h / 2.0 for x in line) / len(line))

    out_lines = []
    for line_idx, line in enumerate(lines, start=1):
        combined_text = " ".join(x.text for x in line)
        direction = _detect_text_direction(combined_text)
        rtl = direction == "rtl"
        line_sorted = sorted(line, key=lambda x: x.bbox.x + x.bbox.w / 2.0, reverse=rtl)
        line_text = " ".join(x.text.strip() for x in line_sorted)
        print(f"      ↳ سطر {line_idx}{f' [{region_label}]' if region_label else ''}: "
              f"اتجاه={'RTL 🔁' if rtl else 'LTR ➡️'} ---> '{line_text}'")
        out_lines.append(line_text)

    return "\n".join(out_lines)


class CellMapper:
    """Turns (OCR fragments, page layout) into one flat list of UI elements."""

    def __init__(self, w_ioa: float = 0.3, w_iou: float = 0.7, w_dist: float = 0.0, min_score: float = 0.25):
        self.w_ioa = w_ioa
        self.w_iou = w_iou
        self.w_dist = w_dist
        self.min_score = min_score

    def _score(self, frag_bbox: BoundingBox, region_bbox: BoundingBox) -> tuple[float, float, float, float]:
        ioa = frag_bbox.ioa(region_bbox)
        iou = frag_bbox.iou(region_bbox)
        dist = frag_bbox.center_distance(region_bbox)
        norm_dist = dist / region_bbox.diagonal if region_bbox.diagonal > 0 else 1.0
        closeness = max(0.0, 1.0 - min(norm_dist, 1.0))
        score = self.w_ioa * ioa + self.w_iou * iou + self.w_dist * closeness
        return ioa, iou, dist, score

    @staticmethod
    def _dump_inputs(regions: list[LayoutRegion], fragments: list[OCRFragment]) -> None:
        print("\n" + "=" * 70)
        print("📥 [الخطوة 1: طباعة المدخلات (Input Dump)]")
        print("=" * 70)
        print(f"📊 عدد المناطق (Regions) القادمة من InvoiceLayout: {len(regions)}")
        for r in regions:
            b = r.bbox
            print(
                f"   • region id={r.id!r} | bbox=(x={b.x}, y={b.y}, w={b.w}, h={b.h}) "
                f"| table_id={r.table_id!r} row={r.row!r} col={r.col!r} "
                f"| role={getattr(r, 'role', None)!r} zone={getattr(r, 'zone', None)!r}"
            )
        print(f"\n📊 عدد شظايا OCR (Fragments) المستلمة: {len(fragments)}")
        for i, f in enumerate(fragments):
            b = f.bbox
            print(
                f"   • fragment[{i}] text={f.text!r} | bbox=(x={b.x}, y={b.y}, w={b.w}, h={b.h}) "
                f"| source_id={getattr(f, 'source_id', None)!r}"
            )

    @staticmethod
    def _dump_envelopes(envelopes: dict[str, tuple[BoundingBox, BoundingBox]]) -> None:
        print("\n" + "-" * 70)
        print("📦 [الخطوة 2: أغلفة الجداول (Table Envelopes & Padding)]")
        print("-" * 70)
        if not envelopes:
            print("   (لا توجد مناطق مرتبطة بأي جدول — لا أغلفة لطباعتها)")
            return
        for table_id, (orig, padded) in envelopes.items():
            print(f"   🗂️ جدول '{table_id}':")
            print(f"      - الغلاف الأصلي  : {orig.as_tuple()}")
            print(f"      - الغلاف الممدد   : {padded.as_tuple()}")

    def map(self, ocr_result: OCRResult, layout: InvoiceLayout) -> StructuredDocument:
        regions = list(layout.regions)
        fragments_in = list(ocr_result.fragments)

        self._dump_inputs(regions, fragments_in)

        order = sorted(range(len(regions)), key=lambda i: regions[i].area)
        by_id = {region.id: i for i, region in enumerate(regions)}

        envelopes = self._build_table_envelopes(regions)
        self._dump_envelopes(envelopes)

        per_region: dict[int, list[OCRFragment]] = {}
        unassigned: list[OCRFragment] = []

        print("\n" + "-" * 70)
        print("🎯 [الخطوة 3: المنافسة والتخصيص (Scoring & Assignment)]")
        print("-" * 70)

        for fragment in fragments_in:
            text = (fragment.text or "").strip()
            if not text:
                continue

            index = self._assign(fragment, regions, order, by_id)
            if index is None:
                print(f"   ❌ رُفض '{text}' — لم يطابق أي منطقة بثقة كافية، سيتحول لحقل حر.")
                unassigned.append(fragment)
            else:
                per_region.setdefault(index, []).append(fragment)

        elements: list[DocumentElement] = []

        print("\n" + "-" * 70)
        print("🧩 [الخطوة 4: التجميع والدمج (Assembly & RTL/LTR Merge)]")
        print("-" * 70)

        for index, region in enumerate(regions):
            region_fragments = per_region.get(index, [])
            if len(region_fragments) > 1:
                print(f"   🟩 منطقة '{region.id}': دمج {len(region_fragments)} شظية...")
            elements.append(self._element(region, region_fragments))

        for index, fragment in enumerate(unassigned):
            elements.append(
                FreeFieldElement(
                    id=f"free:{index}",
                    bbox=fragment.bbox,
                    fragments=[fragment],
                    merged_text=fragment.text.strip(),
                    role=CellRole.UNKNOWN,
                    zone=Zone.UNKNOWN,
                )
            )

        print("\n" + "=" * 70)
        print(f"✅ النتيجة النهائية: {len(regions)} منطقة، و {len(unassigned)} حقل حر غير مُسند.")
        print("=" * 70 + "\n")

        return StructuredDocument(elements=elements)

    def _assign(
        self,
        fragment: OCRFragment,
        regions: list[LayoutRegion],
        order: list[int],
        by_id: dict[str, int],
    ) -> int | None:
        source_id = getattr(fragment, "source_id", None)
        if source_id is not None:
            index = by_id.get(source_id)
            if index is not None:
                print(f"   🆔 تخصيص بالهوية: '{fragment.text}' ---> region id={regions[index].id!r} (source_id مطابق)")
            else:
                print(f"   ⚠️ '{fragment.text}' يحمل source_id={source_id!r} غير معروف في الـ layout — لن يُخمَّن جغرافيًا.")
            return index

        best_index, best_score = None, -1.0
        best_stats: tuple[float, float, float] | None = None

        for i in order:
            region = regions[i]
            ioa, iou, dist, score = self._score(fragment.bbox, region.bbox)
            if ioa > 0 or iou > 0:
                print(
                    f"      · مقابل region id={region.id!r}: "
                    f"IOA={ioa:.3f} IOU={iou:.3f} Dist={dist:.1f} ---> Score={score:.3f}"
                )
            if score > best_score:
                best_score, best_index = score, i
                best_stats = (ioa, iou, dist)

        if best_index is not None and best_score >= self.min_score:
            region = regions[best_index]
            ioa, iou, dist = best_stats
            print(
                f"   ✅ تم تخصيص '{fragment.text}' ---> region id={region.id!r} "
                f"(أفضل تقييم: IOA={ioa:.3f} IOU={iou:.3f} Dist={dist:.1f} Score={best_score:.3f})"
            )
            return best_index

        print(f"   ❌ '{fragment.text}': أفضل تقييم كان {best_score:.3f} < الحد الأدنى {self.min_score} — سيصبح حقلًا حرًا.")
        return None

    @staticmethod
    def _build_table_envelopes(regions: list[LayoutRegion]) -> dict[str, tuple[BoundingBox, BoundingBox]]:
        """Hull (+2% padding) of every region sharing a table_id.

        Purely informational: header/footer regions outside any table must
        still be assignable, so this is never used to filter candidates during
        scoring -- only printed for visibility into the table shape.
        """
        groups: dict[str, list[BoundingBox]] = {}
        for region in regions:
            if region.table_id is not None:
                groups.setdefault(region.table_id, []).append(region.bbox)

        envelopes: dict[str, tuple[BoundingBox, BoundingBox]] = {}
        for table_id, boxes in groups.items():
            x1 = min(b.x for b in boxes)
            y1 = min(b.y for b in boxes)
            x2 = max(b.x + b.w for b in boxes)
            y2 = max(b.y + b.h for b in boxes)
            orig = BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
            pad = 0.02 * max(orig.w, orig.h)
            padded = BoundingBox(
                x=int(orig.x - pad), y=int(orig.y - pad),
                w=int(orig.w + 2 * pad), h=int(orig.h + 2 * pad),
            )
            envelopes[table_id] = (orig, padded)
        return envelopes

    def _element(self, region: LayoutRegion, fragments: list[OCRFragment]) -> DocumentElement:
        merged = merge_fragments_in_cell(fragments, region_label=region.id)
        if merged:
            print(f"      = نص منطقة '{region.id}' النهائي ---> '{merged}'")

        if region.row is None or region.col is None or region.table_id is None:
            return FreeFieldElement(
                id=region.id,
                bbox=region.bbox,
                fragments=fragments,
                merged_text=merged,
                role=region.role,
                zone=region.zone,
            )

        return TableCellElement(
            id=region.id,
            table_id=region.table_id,
            row=region.row,
            col=region.col,
            bbox=region.bbox,
            fragments=fragments,
            merged_text=merged,
            role=region.role,
            zone=region.zone,
            row_span=region.row_span,
            col_span=region.col_span,
        )

    @staticmethod
    def bounds_of(fragments: list[OCRFragment]) -> BoundingBox | None:
        """Public helper for callers assembling their own elements from fragments."""
        return _hull(fragments) if fragments else None