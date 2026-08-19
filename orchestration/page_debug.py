"""Writes one folder per page recording exactly what was cropped and read.

This is what `HybridSuryaQwenEngine._save_debug` used to do, extracted into a
collaborator so it survives the flow refactor. It is the only place in the
pipeline that answers "which crop produced that string", and without it tuning
is guesswork: a wrong reading tells you nothing unless you can see the pixels
the model was actually handed.

It also produces `results.json`, a flat dump of the boxes for eyeballing a run. `postprocess/reconcile.py` used to consume it; the arithmetic repair it did now runs inside the pipeline as `invoice/reconciliation.py`, against the parsed invoice rather than against a file written on the side.
That file is not merely a convenience. `OCRFragment` carries text, bbox,
confidence and source_id -- but not `content_kind`, and nothing at all marks a
region the refiner synthesised. Both are needed to reconcile a row
arithmetically, and this is the only stage where both are still in scope.

Layout, unchanged from before so old tooling and the reconcile pass keep
working:

    det/<timestamp>_pageNN/
        001.png ... NNN.png    one per crop, numbered to match `file`
        _page.png              the page as the recognizer saw it
        _page_boxes.png        the same page with every box drawn and numbered
        results.json           one record per crop
        results.csv            the same, utf-8-sig so Excel renders the Arabic
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Red = read something, orange = came back empty. BGR, because cv2.
_COLOUR_READ = (30, 30, 220)
_COLOUR_EMPTY = (20, 140, 230)


class PageDebugWriter:
    """One instance per run. Not shared between threads -- see `for_run`.

    Flows are cached and reused across concurrent HTTP requests, so a writer
    must never be stored on a flow. It is created per run and passed down the
    call chain explicitly; that is why `_crop_and_read` takes it as an argument
    rather than reading it off `self`.
    """

    def __init__(self, root: str | Path = "det", page_number: int = 1):
        self.root = Path(root)
        self.page_number = page_number
        self._out_dir: Path | None = None

    @classmethod
    def for_run(cls, root: str | Path | None, page_number: int = 1):
        """Returns a writer, or None when debug output is switched off.

        Returning None rather than a no-op writer keeps the cost visible at the
        call site: `if debug is not None` reads as "we are paying for this".
        """
        return None if not root else cls(root, page_number)

    @property
    def out_dir(self) -> Path:
        if self._out_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._out_dir = self.root / f"{stamp}_page{self.page_number:02d}"
            self._out_dir.mkdir(parents=True, exist_ok=True)
        return self._out_dir

    # -- the one method flows call --------------------------------------

    def write(
        self,
        page: np.ndarray,
        crops: Sequence,
        texts: Sequence[str],
        read_times: Sequence[float] | None = None,
    ) -> Path | None:
        """Records this page. Never raises: debug output must not fail a run.

        `crops` are RegionCrop objects and `texts` the recognizer's replies,
        index-aligned -- the caller has already checked that. `read_times`,
        when given, is index-aligned the same way and recorded per row as
        `read_time_s`; omit it (or pass None) when the caller took the batched
        `read_all` path and has no per-crop timing to report.
        """
        try:
            return self._write(page, crops, texts, read_times)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Debug output failed (%s); the run itself is unaffected.", exc)
            return None

    def _write(
        self,
        page: np.ndarray,
        crops: Sequence,
        texts: Sequence[str],
        read_times: Sequence[float] | None,
    ) -> Path:
        out_dir = self.out_dir
        rows = []

        if read_times is not None and len(read_times) != len(crops):
            raise ValueError(
                f"read_times has {len(read_times)} entries for {len(crops)} crops; "
                "they must be index-aligned."
            )

        for index, (crop, text) in enumerate(zip(crops, texts), start=1):
            filename = f"{index:03d}.png"
            cv2.imwrite(str(out_dir / filename), crop.image)

            region = crop.region
            box = region.bbox
            kind = getattr(region.content_kind, "value", region.content_kind)

            rows.append({
                "file": filename,
                "text": text,
                "kind": kind,
                "x": int(box.x), "y": int(box.y),
                "w": int(box.w), "h": int(box.h),
                "detect_confidence": region.confidence,
                # The refiner marks a region it invented by leaving confidence
                # unset -- there was no detection behind it to score. That is
                # the same signal the old engine wrote as `box is None`.
                "synthesised": region.confidence is None,
                "source_id": region.source_id,
                "read_time_s": (
                    round(read_times[index - 1], 3) if read_times is not None else None
                ),
            })

        cv2.imwrite(str(out_dir / "_page.png"), page)
        self._write_overlay(out_dir, page, rows)

        (out_dir / "results.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

        # utf-8-sig: without the BOM, Excel on Windows renders the Arabic as
        # mojibake and the file looks corrupt when it is not.
        with open(out_dir / "results.csv", "w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["file"])
            writer.writeheader()
            writer.writerows(rows)

        logger.info("%d crop(s) + results -> %s", len(rows), out_dir)
        return out_dir

    @staticmethod
    def _write_overlay(out_dir: Path, page: np.ndarray, rows: list[dict]) -> None:
        """Every box drawn and numbered to match its crop filename.

        The numbering is the point. A page of unlabelled rectangles cannot tell
        you which box produced which string; with it, a wrong reading in
        results.json leads straight to the pixels that caused it.
        """
        canvas = page.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        for row in rows:
            x, y, w, h = row["x"], row["y"], row["w"], row["h"]
            colour = _COLOUR_READ if row["text"] else _COLOUR_EMPTY
            cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, 3)
            label = row["file"].split(".")[0]
            # Above the box, or inside it when the box hugs the top edge.
            label_y = y - 6 if y >= 20 else y + 18
            cv2.putText(canvas, label, (x + 2, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

        cv2.imwrite(str(out_dir / "_page_boxes.png"), canvas)