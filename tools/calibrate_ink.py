#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""يقيس كثافة الحبر في قصاصات مجلد det/ ليُختار min_ink_ratio من بيانات حقيقية.

العتبة لا تُخمَّن. شغّل هذا على مجلد أنتجه تشغيل سابق، وسيرتّب القصاصات
بكثافة الحبر ويقترح عتبةً تفصل ما قُرئ عمّا خرج فارغاً أو هراءً.

    python tools/calibrate_ink.py det/20260818_120000_page01

ثم ضع الرقم المقترح في qwen_config.yaml تحت ocr.cropper.params.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

INK_DELTA = 30  # يجب أن يطابق ink_delta في padded_cropper


def ink_ratio(path: Path, delta: int = INK_DELTA) -> float:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    paper = np.percentile(arr, 85)
    return float((arr < paper - delta).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("det_dir", help="مجلد det/<timestamp>_pageNN")
    ap.add_argument("--delta", type=int, default=INK_DELTA)
    args = ap.parse_args()

    det = Path(args.det_dir)
    results_file = det / "results.json"
    if not results_file.exists():
        sys.exit(f"results.json غير موجود في {det}")

    rows = {r["file"]: r for r in json.loads(results_file.read_text(encoding="utf-8"))}

    measured = []
    for name, row in rows.items():
        crop = det / name
        if not crop.exists():
            continue
        measured.append((ink_ratio(crop, args.delta), name, row.get("text", "")))

    if not measured:
        sys.exit("لم تُعثر على قصاصات.")

    measured.sort()

    print(f"{'الحبر':>8}  {'الملف':<10}  النص")
    print("-" * 70)
    for ratio, name, text in measured:
        shown = (text[:44] + "…") if len(text) > 45 else text
        print(f"{ratio:>7.3%}  {name:<10}  {shown!r}")

    # القصاصات التي أخرجت نصاً هي ما يجب أن ينجو؛ الفارغة هي ما يجب أن يُحذف.
    with_text = [r for r, _, t in measured if t.strip()]
    empty = [r for r, _, t in measured if not t.strip()]

    print()
    print("-" * 70)
    if with_text:
        print(f"أدنى كثافة لقصاصة أخرجت نصاً : {min(with_text):.3%}")
    if empty:
        print(f"أعلى كثافة لقصاصة خرجت فارغة : {max(empty):.3%}")

    if with_text:
        floor = min(with_text)
        # نصف أدنى قصاصة ناجحة: هامش أمان يقصّ الضجيج بلا خسارة خانة حقيقية.
        # لا تقترب من الحدّ الأدنى نفسه؛ الصورة التالية ستختلف قليلاً.
        suggested = round(floor * 0.5, 4)
        print()
        print(f"عتبة مقترحة: min_ink_ratio: {suggested}")
        print()
        print("راجع القائمة أعلاه أولاً. إن كانت خانة حقيقية قد خرجت فارغة")
        print("لأن النموذج أخطأ فيها لا لأنها خالية، فهي ليست ضجيجاً وستحذفها")
        print("هذه العتبة. القياس يقترح، وأنت تقرّر.")
    return 0


if __name__ == "__main__":
    sys.exit(main())