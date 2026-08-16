# ocr/engines/hybrid_surya_qwen_engine.py
"""Surya locates the fields, Qwen reads each one.

Rationale, measured on this project's sample invoice:

  * Surya's own recognition on the full page produced Chinese glyphs and LaTeX
    for handwritten Arabic. Its *detector*, however, placed boxes correctly.
  * Qwen read the same page poorly too, but read individual field crops far
    better, and the per-column prompt suppressed the digit->Latin confusion
    (٥->O, ٧->V, ٨->A) entirely.

So: keep Surya's detector, throw away its recognizer, and hand each crop to
Qwen. This class implements OCREngine, so PipelineOrchestrator needs no change
-- it is selected purely from config.

Known limit worth stating plainly: the detector misses short isolated digits.
On the sample, quantities of one or two characters were never boxed, so they
never reach Qwen at all. Lowering DETECTOR_TEXT_THRESHOLD in local.env helps;
verify per-form rather than assuming.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from surya.detection import DetectionPredictor

from core.domain.geometry import BoundingBox
from core.domain.image_payload import ImagePayload
from core.domain.ocr import OCRFragment, OCRResult
from ocr.engines.qwen_vlm_engine import PROMPTS, QwenVLMEngine, to_pil
from ocr.registry import engine_registry


@engine_registry.register("surya_qwen")
class HybridSuryaQwenEngine:
    def __init__(
        self,
        pad: int = 12,
        upscale: int = 2,
        min_side: int = 12,
        column_kinds: list | None = None,
        debug_dir: str | None = "det",
        **engine_params,
    ):
        self.det_predictor = DetectionPredictor()
        self.reader = QwenVLMEngine(**engine_params)
        # Where to dump each crop plus a per-page JSON/CSV of what was read.
        # Set to null in config to turn off. Invaluable while tuning: it is the
        # only way to see WHICH crop produced a given string.
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self._page_counter = 0
        self.pad = pad              # Surya crops tight; give the glyphs room
        self.upscale = upscale      # small crops read better enlarged
        self.min_side = min_side    # ignore specks
        # Optional list of {kind, from, to} with x as a fraction of image
        # width, so numeric columns get the digits-only prompt.
        #
        # A LIST, not a dict keyed by kind: an invoice has several numeric
        # columns (price, quantity, total) that all want kind "number", and
        # dict keys cannot repeat. Keying by column name instead silently fell
        # back to the generic prompt, because "price" is not a prompt kind.
        self.column_kinds = list(column_kinds or [])
        for c in self.column_kinds:
            if c["kind"] not in PROMPTS:
                raise ValueError(
                    f"column_kinds kind={c['kind']!r} is not a known prompt. "
                    f"Valid kinds: {sorted(PROMPTS)}")

    def _kind_for(self, x_center: float, width: int) -> str:
        if not self.column_kinds:
            return self.reader.kind
        frac = x_center / width
        for c in self.column_kinds:
            if c["from"] <= frac < c["to"]:
                return c["kind"]
        return self.reader.kind

    def recognize(self, image: ImagePayload) -> OCRResult:
        arr = image.image
        pil = to_pil(arr)
        w, h = pil.size

        det = self.det_predictor([pil])[0]

        out_dir = None
        if self.debug_dir is not None:
            self._page_counter += 1
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = self.debug_dir / f"{stamp}_page{self._page_counter:02d}"
            out_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        fragments = []
        for box in det.bboxes:
            x1, y1, x2, y2 = (int(v) for v in box.bbox)
            if x2 - x1 < self.min_side or y2 - y1 < self.min_side:
                continue
            cx1, cy1 = max(0, x1 - self.pad), max(0, y1 - self.pad)
            cx2, cy2 = min(w, x2 + self.pad), min(h, y2 + self.pad)
            crop = pil.crop((cx1, cy1, cx2, cy2))
            if crop.size[0] == 0 or crop.size[1] == 0:
                continue
            if self.upscale > 1:
                crop = crop.resize(
                    (crop.width * self.upscale, crop.height * self.upscale),
                    Image.Resampling.LANCZOS)

            kind = self._kind_for((x1 + x2) / 2, w)
            try:
                text = self.reader.read(crop, kind=kind)
            except Exception as exc:                      # noqa: BLE001
                # One unreadable field must not abort a whole invoice.
                text = ""
                print(f"[surya_qwen] crop at {(x1, y1, x2, y2)} failed: {exc}")

            if text.strip().upper() == "EMPTY":
                text = ""

            fragments.append(OCRFragment(
                text=text,
                bbox=BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1),
                # Detector confidence, not recognition confidence. The VLM
                # does not expose a score, so do not invent one.
                confidence=getattr(box, "confidence", None),
            ))

            if out_dir is not None:
                crop.save(out_dir / f"{len(fragments):03d}.png")
                rows.append({
                    "file": f"{len(fragments):03d}.png",
                    "text": text,
                    "kind": kind,
                    "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
                    "detect_confidence": getattr(box, "confidence", None),
                })

        if out_dir is not None:
            self._save_debug(out_dir, pil, rows)

        return OCRResult(fragments=fragments,
                         engine_name="surya_qwen",
                         raw_engine_output=det)

    def _save_overlay(self, out_dir: Path, page, rows):
        """Draw every detected box, numbered to match its crop filename.

        This replaces what `surya_detect --images` used to produce, and adds
        the numbering: without it there is no way to tell which box produced
        which string in results.csv.
        """
        from PIL import ImageDraw

        canvas = page.convert("RGB").copy()
        draw = ImageDraw.Draw(canvas)
        for r in rows:
            x, y, w, h = r["x"], r["y"], r["w"], r["h"]
            # Red = read something, orange = came back empty.
            colour = (220, 30, 30) if r["text"] else (230, 140, 20)
            draw.rectangle([x, y, x + w, y + h], outline=colour, width=3)
            label = r["file"].split(".")[0]
            # Label above the box, or inside it when the box hugs the top edge.
            ly = y - 18 if y >= 18 else y + 2
            draw.text((x + 2, ly), label, fill=colour)
        canvas.save(out_dir / "_page_boxes.png")

    def _save_debug(self, out_dir: Path, page, rows):
        page.save(out_dir / "_page.png")
        self._save_overlay(out_dir, page, rows)
        with open(out_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        # utf-8-sig so Excel on Windows renders the Arabic correctly.
        with open(out_dir / "results.csv", "w", encoding="utf-8-sig",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file", "text", "kind",
                                              "x", "y", "w", "h",
                                              "detect_confidence"])
            w.writeheader()
            w.writerows(rows)
        print(f"[surya_qwen] {len(rows)} crops + results -> {out_dir}")