#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
postprocess.py — طبقة ما بعد المعالجة لأنبوب استخراج الفواتير المكتوبة بخط اليد

تعمل على `results.json` الناتج عن الأنبوب الحالي، بلا إعادة تشغيل OCR.

المشاكل التي تعالجها، بالترتيب:

  1. تطبيع الترميز
     المخرجات تخلط نطاقين في السلسلة الواحدة: "١۶" = U+0661 (عربي-هندي)
     + U+06F6 (فارسي). أي مقارنة نصية أو تحقق حسابي يفشل بصمت قبل هذا.

  2. إزالة الفواصل المُقحمة
     الموديل يقحم فواصل وشرطات بين الأرقام حين تلتقط القصاصة خطوط التنقيط
     المطبوعة. مثال حقيقي: "٥-١-٧,٦,٥" تسلسلها 51765 = ٥١٧٫٦٥ صحيحة تماماً.

  3. عكس التحويل اللاتيني المتبقّي
     ٥←O ، ٧←V ، ٨←A (موثّق في §٦٫١ من سجل المشروع).

  4. رفض الخانات الرقمية الخالية من الأرقام
     "سَيرٌ" و"ثلاثون" و"سائلا" في أعمدة رقمية ليست قيماً — تُفرَّغ وتُعلَّم.

  5. التحقق الحسابي بالبحث عن موضع الفاصلة العشرية
     بدل افتراض موضع الفاصلة، نجرّب المواضع الممكنة ونقبل التركيبة الوحيدة
     التي تحقق: السعر × العدد = الإجمالي. هذا يصلح الفواصل ويتحقق في خطوة
     واحدة، ويستنتج القيمة الغائبة من قيمتين معروفتين.

  6. التحقق على مستوى الفاتورة
     مجموع إجماليات الصفوف = المجموع الكلي.

الاستخدام:
    python postprocess.py results.json
    python postprocess.py results.json --page-width 1578 --out clean.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# ١. تطبيع الترميز
# ─────────────────────────────────────────────────────────────────────────────

AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"   # U+0660..U+0669  عربي-هندي
FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"   # U+06F0..U+06F9  عربي-هندي ممتد (فارسي)

# إلى ASCII للحساب
_TO_ASCII = {ord(c): str(i) for i, c in enumerate(AR_DIGITS)}
_TO_ASCII.update({ord(c): str(i) for i, c in enumerate(FA_DIGITS)})

# إلى عربي-هندي موحّد للعرض
_TO_ARABIC = {ord(str(i)): AR_DIGITS[i] for i in range(10)}
_TO_ARABIC.update({ord(c): AR_DIGITS[i] for i, c in enumerate(FA_DIGITS)})

# §٦٫١ — الموديل يترجم شكل الرقم إلى الحرف اللاتيني المشابه بصرياً.
# مقتصرة على الثلاثة الموثّقة فقط؛ لا تُوسَّع بلا قياس.
LOOKALIKES = {"O": "5", "o": "5", "V": "7", "v": "7", "A": "8"}

# الفواصل التي يقحمها الموديل عند التقاط خطوط التنقيط
NOISE_CHARS = ",.-_/\\|'\"·،؍٫٬ \t\u200f\u200e"


def to_ascii_digits(text: str) -> str:
    return text.translate(_TO_ASCII)


def to_arabic_digits(text: str) -> str:
    return text.translate(_TO_ARABIC)


def fix_lookalikes(text: str) -> str:
    return "".join(LOOKALIKES.get(ch, ch) for ch in text)


def digit_sequence(text: str) -> str:
    """تسلسل الأرقام الخام بعد التطبيع وإزالة كل ما عداه."""
    return re.sub(r"\D", "", fix_lookalikes(to_ascii_digits(text or "")))


def noise_ratio(text: str) -> float:
    """نسبة محارف الضجيج إلى الطول — مؤشر على قصاصة التقطت خطوط تنقيط."""
    if not text:
        return 0.0
    noisy = sum(1 for ch in text if ch in NOISE_CHARS)
    return noisy / len(text)


# ─────────────────────────────────────────────────────────────────────────────
# ٢. نموذج البيانات
# ─────────────────────────────────────────────────────────────────────────────

OK          = "ok"           # قيمة مقروءة وصالحة الشكل
NO_DIGITS   = "no_digits"    # عمود رقمي بلا أي رقم → مرفوضة
DERIVED     = "derived"      # استُنتجت حسابياً
CONFIRMED   = "confirmed"    # قُرئت وأكّدها الحساب
REPAIRED    = "repaired"     # قُرئ تسلسلها صحيحاً وأُصلح موضع الفاصلة
CONFLICT    = "conflict"     # قُرئت لكن الحساب يكذّبها
UNRESOLVED  = "unresolved"   # مجهولان في الصف فتعذّر الحل


@dataclass
class Cell:
    raw: str = ""
    digits: str = ""
    value: Optional[float] = None
    status: str = OK
    synthesised: bool = False
    confidence: Optional[float] = None
    note: str = ""

    def display(self) -> str:
        if self.value is None:
            return ""
        s = f"{self.value:.2f}".rstrip("0").rstrip(".")
        return to_arabic_digits(s).replace(".", "٫")


@dataclass
class Row:
    index: int
    y: int
    price: Cell = field(default_factory=Cell)
    qty: Cell = field(default_factory=Cell)
    desc: Cell = field(default_factory=Cell)
    total: Cell = field(default_factory=Cell)
    verdict: str = ""

    def numeric_cells(self):
        return {"price": self.price, "qty": self.qty, "total": self.total}


# ─────────────────────────────────────────────────────────────────────────────
# ٣. التجميع في صفوف وإسناد الأعمدة
# ─────────────────────────────────────────────────────────────────────────────

# خرائط الأعمدة لكل قالب — نسب من عرض الصفحة.
#
# القالبان معكوسان تماماً. تشغيل القالب الخطأ لا يُنتج خطأً: يسند كل خانة
# إلى عمود آخر ويخرج تقريراً يبدو سليماً وهو هراء. لذلك الاختيار صريح
# وإجباري عند تعارض، ولا يُخمَّن.
#
# يجب أن تطابق column_kinds في ملف qwen_config المقابل. أي تعديل هناك
# يُنقل إلى هنا.
TEMPLATES = {
    # الفاتورة الأصلية: Price / Quantity / Description / Total من اليسار
    "old": [
        ("price", 0.000, 0.163),
        ("qty",   0.163, 0.285),
        ("desc",  0.285, 0.775),
        ("total", 0.775, 0.940),
    ],
    # القالب المسطَّر: ملاحظات / الإجمالي / الإفرادي / العدد / اسم المنتج
    "new": [
        ("notes", 0.000, 0.270),
        ("total", 0.270, 0.430),
        ("price", 0.430, 0.610),
        ("qty",   0.610, 0.710),
        ("desc",  0.710, 0.930),
    ],
}

COLUMNS = TEMPLATES["old"]        # يُستبدل من سطر الأوامر


def columns_from_config(column_kinds) -> list | None:
    """
    يبني خريطة الأعمدة من column_kinds في ملف الإعداد.

    هذا هو المسار الصحيح داخل الأنبوب: ملف الإعداد يعرف القالب يقيناً،
    فالتخمين من الصناديق (detect_template أدناه) احتياطٌ لسطر الأوامر فقط.
    ووجود الحدود في مكان واحد يمنع انحراف خريطة reconcile عن خريطة المحرّك —
    وهو انحراف لا يُنتج خطأً بل تقريراً يبدو سليماً وهو خاطئ بالكامل.

    يتطلب حقل role في كل عمود. يعود None إن غاب، فيسقط المستدعي إلى التخمين.
    """
    if not column_kinds:
        return None
    cols = []
    for c in column_kinds:
        role = c.get("role") if isinstance(c, dict) else getattr(c, "role", None)
        if not role:
            return None
        frm = c["from"] if isinstance(c, dict) else c.from_
        to = c["to"] if isinstance(c, dict) else c.to
        cols.append((role, float(frm), float(to)))
    return cols


def detect_template(boxes: list[dict], page_width: int) -> str:
    """
    تخمين القالب من موضع أعرض صندوق نصي.

    الوصف هو أعرض عمود في الحالتين، لكنه في القديم وسط الصفحة (~0.53)
    وفي الجديد يمينها (~0.80). الفجوة واسعة فالتمييز موثوق — لكنه يبقى
    تخميناً، فيُطبع دائماً ليراه المستخدم ويُصحّحه بـ --template عند الحاجة.
    """
    widest = max((b for b in boxes if b.get("w")), key=lambda b: b["w"], default=None)
    if not widest:
        return "old"
    centre = (widest["x"] + widest["w"] / 2) / page_width
    return "new" if centre > 0.68 else "old"


def assign_column(x: int, w: int, page_width: int) -> Optional[str]:
    centre = (x + w / 2) / page_width
    for name, lo, hi in COLUMNS:
        if lo <= centre < hi:
            return name
    return None


def group_rows(boxes: list[dict], page_width: int, tolerance: float = 0.35) -> list[Row]:
    """تجميع الصناديق في صفوف بالتقارب الرأسي، ثم إسناد كل صندوق لعموده."""
    if not boxes:
        return []

    ordered = sorted(boxes, key=lambda b: b.get("y", 0))
    heights = [b.get("h", 0) for b in ordered if b.get("h", 0) > 0]
    median_h = sorted(heights)[len(heights) // 2] if heights else 100
    threshold = median_h * tolerance * 2

    clusters: list[list[dict]] = [[ordered[0]]]
    for box in ordered[1:]:
        prev_y = clusters[-1][-1].get("y", 0)
        if abs(box.get("y", 0) - prev_y) <= threshold:
            clusters[-1].append(box)
        else:
            clusters.append([box])

    rows: list[Row] = []
    for i, cluster in enumerate(clusters):
        row = Row(index=i, y=min(b.get("y", 0) for b in cluster))
        for box in cluster:
            col = assign_column(box.get("x", 0), box.get("w", 0), page_width)
            if col is None:
                continue
            cell = Cell(
                raw=box.get("text", "") or "",
                synthesised=bool(box.get("synthesised", False)),
                confidence=box.get("detect_confidence"),
            )
            existing = getattr(row, col)
            # عند التزاحم، فضّل الصندوق المكتشف على المولّد، والأطول نصاً
            if existing.raw or existing.synthesised:
                if existing.synthesised and not cell.synthesised:
                    setattr(row, col, cell)
                elif len(cell.raw) > len(existing.raw) and not cell.synthesised:
                    setattr(row, col, cell)
            else:
                setattr(row, col, cell)
        rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# ٤. التطبيع على مستوى الخانة
# ─────────────────────────────────────────────────────────────────────────────

def normalise_numeric(cell: Cell, max_decimals: int = 2) -> None:
    """يستخرج تسلسل الأرقام ويقترح قيمة أولية. لا يبتّ في موضع الفاصلة بعد."""
    cell.digits = digit_sequence(cell.raw)

    if not cell.digits:
        cell.status = NO_DIGITS
        cell.value = None
        if cell.raw.strip():
            cell.note = "عمود رقمي أخرج نصاً عربياً — مرفوض"
        return

    # قيمة أولية: احترم الفاصلة الوحيدة إن وُجدت وبدت عشرية معقولة
    ascii_text = fix_lookalikes(to_ascii_digits(cell.raw))
    single_sep = re.fullmatch(r"(\d+)[,.٫](\d{1,2})", ascii_text.strip())
    if single_sep:
        cell.value = float(f"{single_sep.group(1)}.{single_sep.group(2)}")
    else:
        cell.value = float(cell.digits)

    if noise_ratio(cell.raw) > 0.30:
        cell.note = "كثافة فواصل عالية — تسلسل الأرقام أوثق من التنسيق"


def dot_run_candidates(raw: str) -> list[str]:
    """
    قراءات بديلة حين تنتهي الخانة بسلسلة نقاط.

    هذه النماذج تكتب الأصفار المتتابعة نقاطاً: «212...» تعني ٢١٢٠٠٠، و«١٢..»
    تعني ١٢٠٠. عدد النقاط لا يطابق عدد الأصفار دائماً، فنولّد بدائل ونترك
    المعادلة (السعر × العدد = الإجمالي) تحسم أيّها الصحيح — بدل التخمين.

    يعود بقائمة تسلسلات أرقام مرتّبة: الأصل أولاً ثم الأطول فالأطول.
    """
    digits = digit_sequence(raw)
    if not digits:
        return []
    tail = re.search(r"[.\u2026]+\s*$", (raw or "").strip())
    if not tail:
        return [digits]
    n_dots = len(tail.group(0).replace(" ", ""))
    out = [digits]
    # جرّب من نقطة واحدة حتى عدد النقاط، وواحدة زائدة: «212...» قد تُكتب
    # بثلاث نقاط لثلاثة أصفار أو بنقطتين اختصاراً.
    for k in range(1, n_dots + 2):
        cand = digits + "0" * k
        if cand not in out:
            out.append(cand)
    return out


def decimal_candidates(digits: str, max_decimals: int = 2) -> list[float]:
    """كل القيم الممكنة بوضع الفاصلة في المواضع المسموحة."""
    if not digits:
        return []
    out = []
    for d in range(0, max_decimals + 1):
        if d == 0:
            out.append(float(digits))
        elif len(digits) > d:
            out.append(float(digits[:-d] + "." + digits[-d:]))
    # أزل التكرار مع الحفاظ على الترتيب
    seen, unique = set(), []
    for v in out:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# ٥. التحقق الحسابي على مستوى الصف
# ─────────────────────────────────────────────────────────────────────────────

TOL = 0.02          # تسامح القسمة العشرية
MAX_ERR_RATE = 0.50  # أقصى نسبة أرقام مختلفة ليُقبل الاستنتاج كتفسير للقراءة


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def purity(cell: Cell) -> float:
    """نسبة الأرقام إلى طول النص الخام — قصاصة نظيفة تقترب من ١٫٠."""
    raw = (cell.raw or "").strip()
    return len(cell.digits) / len(raw) if raw else 0.0


def explains_reading(derived: float, observed_digits: str) -> bool:
    """
    هل يفسّر الاستنتاج القراءة الملاحَظة؟

    شرط أمان جوهري: لا نستبدل قراءةً بقيمة محسوبة إلا إذا كانت القيمة المحسوبة
    قريبة مما قرأه الموديل فعلاً. تقارُبها يعني أن الموديل أخطأ في رقم أو رقمين
    (خطأ تمييز مألوف)، لا أنه قرأ خانة أخرى تماماً. بلا هذا الشرط تولّد الطبقة
    أرقاماً واثقة وخاطئة — أسوأ من ترك الخانة فارغة.
    """
    if not observed_digits:
        return False
    d = re.sub(r"\D", "", f"{derived:.2f}".rstrip("0").rstrip("."))
    span = max(len(d), len(observed_digits))
    return levenshtein(d, observed_digits) / span <= MAX_ERR_RATE


def observed_decimals(raw: str) -> Optional[int]:
    """
    عدد الأرقام بعد آخر فاصل في القراءة الخام.

    الموديل كثيراً ما يخطئ في *قيمة* الرقم ويصيب في *موضع* الفاصلة، لأن الفاصلة
    علامة بصرية أوضح من شكل الرقم. نستخدم هذا الموضع لفضّ الالتباس بين قيم
    تتشارك التسلسل نفسه (٥١٧٫٦٥ و٥١٧٦٫٥ و٥١٧٦٥).

    الإشارة تُقبل من فاصل واحد فقط. القراءة المتعددة الفواصل («٥-١-٧,٦,٥»)
    ضجيجُ خطوط تنقيط لا تنسيقُ عدد، فتُهمَل إشارتها بدل أن تضلّل.
    """
    ascii_text = fix_lookalikes(to_ascii_digits(raw or ""))
    parts = [p for p in re.split(r"[,.٫،\-_/\\|]", ascii_text) if p.strip()]
    if len(parts) != 2:
        return None
    tail = re.sub(r"\D", "", parts[-1])
    return len(tail) if 1 <= len(tail) <= 2 else None


def pick_by_separator(options, raw: str) -> Optional[float]:
    """يختار من بين قيم متساوية التسلسل تلك التي يطابق كسرُها موضعَ الفاصلة."""
    options = sorted(options)
    if len(options) == 1:
        return options[0]
    want = observed_decimals(raw)
    if want is None:
        return None
    matched = [
        v for v in options
        if len(f"{v:.2f}".rstrip("0").rstrip(".").partition(".")[2]) == want
    ]
    return matched[0] if len(matched) == 1 else None


def _solve_missing(missing: str, p_cands, q_cands, t_cands) -> set[float]:
    if missing == "total":
        return {round(pv * qv, 2) for pv in p_cands for qv in q_cands}
    if missing == "price":
        return {round(tv / qv, 2) for qv in q_cands for tv in t_cands if qv}
    if missing == "qty":
        return {round(tv / pv, 2) for pv in p_cands for tv in t_cands if pv}
    return set()


def reconcile_row(row: Row) -> None:
    """
    السعر × العدد = الإجمالي.

    يبحث في مواضع الفاصلة الممكنة ويقبل التركيبة الوحيدة المتّسقة. إن تعذّر،
    يُسقط أقلّ الخانات موثوقية ويعيد الاستنتاج من الباقيتين.

    ترتيب الموثوقية مستمدّ من قياس السجل لا من الحدس: الأرقام الطويلة تفشل
    والقصيرة تنجح (§٩)، فالخانة ذات أطول تسلسل أرقام هي أولى المشتبهات،
    ويُرجَّح بينها بنقاء القصاصة.
    """
    cells = row.numeric_cells()
    known = {k: c for k, c in cells.items() if c.digits}

    def cands(name: str, cell: Cell):
        # وسّع بقراءات «النقاط = أصفار» قبل توليد مواضع الفاصلة، فالمعادلة
        # هي التي تحسم كم صفراً تمثّله سلسلة النقاط.
        seqs = dot_run_candidates(cell.raw) or ([cell.digits] if cell.digits else [])
        if name == "qty":
            return [float(s) for s in seqs if s]
        out = []
        for s in seqs:
            for v in decimal_candidates(s):
                if v not in out:
                    out.append(v)
        return out

    pools = {k: cands(k, c) for k, c in cells.items()}

    # ── الحالة الأولى: الثلاثة مقروءة ───────────────────────────────────────
    if len(known) == 3:
        matches = [
            (pv, qv, tv)
            for pv in pools["price"] for qv in pools["qty"] for tv in pools["total"]
            if abs(pv * qv - tv) <= TOL
        ]
        if len(matches) == 1:
            pv, qv, tv = matches[0]
            # اعتمد الحاصل المحسوب لا المقروء: التطابق قُبل ضمن تسامح، فقد
            # يحمل المقروء رقماً دخيلاً (١٠٩٤٫٤١ بدل ١٠٩٤٫٤٠). الضرب أدقّ.
            tv = round(pv * qv, 2)
            for name, val in (("price", pv), ("qty", qv), ("total", tv)):
                cell = cells[name]
                moved = cell.value is not None and abs(cell.value - val) > TOL
                cell.value = val
                cell.status = REPAIRED if moved else CONFIRMED
                if moved:
                    cell.note = "التسلسل صحيح؛ أُصلح موضع الفاصلة فقط"
            row.verdict = "متّسق حسابياً"
            return
        if len(matches) > 1:
            # المعادلة لا تحدّد المقياس: ٥٫٩٥×٨٧ و٥٩٫٥×٨٧ و٥٩٥×٨٧ كلها متّسقة.
            # نرجّح بأنظف إشارة فاصلة في الصف — خانة ذات فاصل واحد فقط.
            scored = []
            for pv, qv, tv in matches:
                hits = 0
                for name, val in (("price", pv), ("total", tv)):
                    want = observed_decimals(cells[name].raw)
                    if want is None:
                        continue
                    got = len(f"{val:.2f}".rstrip("0").rstrip(".").partition(".")[2])
                    hits += 1 if got == want else -1
                # ورجّح التركيبة التي تُبقي أكبر عدد من الخانات على قراءتها
                # الحرفية. خانة قُرئت نظيفةً بلا نقاط أوثق من خانة أُعيد
                # تفسيرها، فـ«١٣٢٥٠ × ١٦ = ٢١٢٠٠٠» أرجح من «١٣٢٥ × ١٦ = ٢١٢٠٠»
                # رغم أن كلتيهما تحقق المعادلة.
                for name, val in (("price", pv), ("qty", qv), ("total", tv)):
                    d = cells[name].digits
                    if d and abs(val - float(d)) <= TOL:
                        hits += 1
                scored.append((hits, (pv, qv, tv)))
            best = max(s for s, _ in scored)
            winners = [combo for s, combo in scored if s == best]
            if best > 0 and len(winners) == 1:
                pv, qv, tv = winners[0]
                for name, val in (("price", pv), ("qty", qv), ("total", tv)):
                    cell = cells[name]
                    moved = cell.value is not None and abs(cell.value - val) > TOL
                    cell.value = val
                    cell.status = REPAIRED if moved else CONFIRMED
                    if moved:
                        cell.note = "التسلسل صحيح؛ أُصلح موضع الفاصلة فقط"
                row.verdict = "متّسق حسابياً (رُجِّح المقياس بموضع الفاصلة)"
                return
            row.verdict = "أكثر من تركيبة متّسقة — يلزم ترجيح بشري"
            return

        # لا تتحقق المعادلة → أسقط أقلّ الخانات موثوقية وأعد الاستنتاج
        suspect = max(known, key=lambda k: (len(known[k].digits), -purity(known[k])))
        trimmed = {k: v for k, v in pools.items() if k != suspect}
        options = _solve_missing(
            suspect,
            trimmed.get("price", []), trimmed.get("qty", []), trimmed.get("total", []),
        )
        viable = [v for v in options if explains_reading(v, known[suspect].digits)]
        chosen = pick_by_separator(viable, known[suspect].raw) if viable else None

        if chosen is not None:
            target = cells[suspect]
            target.value = chosen
            target.status = REPAIRED
            target.note = (
                f"القراءة «{target.raw}» تناقض المعادلة؛ صُحّحت من الخانتين "
                f"الأخريين، والفارق في رقم أو رقمين فقط"
            )
            for k, c in known.items():
                if k != suspect and c.status == OK:
                    c.status = CONFIRMED
            row.verdict = f"صُحّح «{suspect}» حسابياً"
            return

        row.verdict = "الأرقام الثلاثة مقروءة والمعادلة لا تتحقق → مراجعة بشرية"
        for c in known.values():
            if c.status == OK:
                c.status = CONFLICT
        return

    # ── الحالة الثانية: خانتان معروفتان ─────────────────────────────────────
    if len(known) == 2:
        missing = next(k for k in ("price", "qty", "total") if k not in known)
        options = _solve_missing(
            missing, pools["price"], pools["qty"], pools["total"]
        )
        source_raw = " ".join(c.raw for c in known.values())
        chosen = pick_by_separator(sorted(options), source_raw) if options else None

        if chosen is not None:
            target = cells[missing]
            target.value = chosen
            target.status = DERIVED
            # لا شاهد مستقل هنا: الخانتان المصدر لم تتأكّدا بشيء، فالاستنتاج
            # صحيح رياضياً وغير مؤكَّد واقعياً. يُعرَض ولا يُعتمد بلا مراجعة.
            target.note = "مستنتجة من خانتين غير مؤكَّدتين — تحتاج تأكيداً بصرياً"
            row.verdict = f"استُنتج «{missing}» حسابياً (بلا شاهد مستقل)"
        else:
            row.verdict = "خانتان معروفتان لكن موضع الفاصلة ملتبس — يلزم تدخل"
        return

    # ── الحالة الثالثة: مجهولان أو أكثر ─────────────────────────────────────
    row.verdict = "مجهولان أو أكثر — تعذّر الحل حسابياً"
    for c in cells.values():
        if c.status == OK and not c.digits:
            c.status = UNRESOLVED


# ─────────────────────────────────────────────────────────────────────────────
# ٦. الأنبوب
# ─────────────────────────────────────────────────────────────────────────────

def process(boxes: list[dict], page_width: int, row_tolerance: float = 0.35):
    rows = group_rows(boxes, page_width, row_tolerance)

    for row in rows:
        for cell in row.numeric_cells().values():
            if cell.raw or cell.synthesised:
                normalise_numeric(cell)
        row.desc.raw = (row.desc.raw or "").strip()
        reconcile_row(row)

    return rows


def report(rows: list[Row]) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("تقرير ما بعد المعالجة")
    lines.append("=" * 78)

    needs_review = 0
    for row in rows:
        has_content = any(
            c.raw or c.value is not None for c in row.numeric_cells().values()
        ) or row.desc.raw
        if not has_content:
            continue

        lines.append("")
        lines.append(f"— صف {row.index}  (y={row.y})")
        if row.desc.raw:
            lines.append(f"    البيان   : {row.desc.raw}")
        for label, key in (("السعر", "price"), ("العدد", "qty"), ("الإجمالي", "total")):
            cell = getattr(row, key)
            if not (cell.raw or cell.value is not None):
                continue
            shown = cell.display() or "—"
            flag = {
                CONFIRMED: "✓ مؤكّد حسابياً",
                DERIVED:   "◆ مستنتج",
                REPAIRED:  "⟳ أُصلح موضع الفاصلة",
                CONFLICT:  "✗ يناقض المعادلة",
                NO_DIGITS: "✗ مرفوض (بلا أرقام)",
                UNRESOLVED:"? غير محسوم",
                OK:        "· كما قُرئ",
            }.get(cell.status, cell.status)
            raw_note = f"   [خام: {cell.raw!r}]" if cell.raw and cell.display() != cell.raw else ""
            lines.append(f"    {label:8}: {shown:12} {flag}{raw_note}")
            if cell.note:
                lines.append(f"              ↳ {cell.note}")
        if row.verdict:
            lines.append(f"    الحكم    : {row.verdict}")
            if "تعذّر" in row.verdict or "لا تتحقق" in row.verdict or "يلزم" in row.verdict:
                needs_review += 1

    lines.append("")
    lines.append("-" * 78)
    lines.append(f"صفوف تحتاج مراجعة بشرية موجّهة: {needs_review}")
    lines.append("-" * 78)
    return "\n".join(lines)


def to_json(rows: list[Row]) -> list[dict]:
    out = []
    for row in rows:
        entry = {"row": row.index, "y": row.y, "verdict": row.verdict,
                 "description": row.desc.raw}
        for key in ("price", "qty", "total"):
            cell = getattr(row, key)
            entry[key] = {
                "value": cell.value,
                "display": cell.display(),
                "digits": cell.digits,
                "raw": cell.raw,
                "status": cell.status,
                "synthesised": cell.synthesised,
                "note": cell.note,
            }
        out.append(entry)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="طبقة ما بعد المعالجة لنتائج OCR")
    ap.add_argument("results", help="مسار results.json")
    ap.add_argument("--page-width", type=int, default=1578)
    ap.add_argument("--row-tolerance", type=float, default=0.35)
    ap.add_argument("--out", default=None, help="مسار حفظ الناتج النظيف")
    ap.add_argument("--template", choices=sorted(TEMPLATES), default=None,
                    help="خريطة الأعمدة. يُخمَّن من البيانات إن أُهمل.")
    args = ap.parse_args()

    with open(args.results, encoding="utf-8") as fh:
        data = json.load(fh)
    boxes = data if isinstance(data, list) else data.get("boxes", [])

    global COLUMNS
    template = args.template or detect_template(boxes, args.page_width)
    COLUMNS = TEMPLATES[template]
    source = "محدَّد يدوياً" if args.template else "مُخمَّن — راجعه"
    print(f"القالب: {template}  ({source})\n")

    rows = process(boxes, args.page_width, args.row_tolerance)
    print(report(rows))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(to_json(rows), fh, ensure_ascii=False, indent=2)
        print(f"\nحُفظ الناتج النظيف في: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())