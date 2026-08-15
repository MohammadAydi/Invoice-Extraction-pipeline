# tools/qwen_worker.py
"""Long-lived Qwen worker, run inside the transformers>=5.0 venv.

Exists because Qwen3.5 needs transformers>=5.0 while surya-ocr 0.17.1 pins
<5.0. They cannot share a process, so the main pipeline spawns this script
with the other venv's python.exe and talks to it over stdin/stdout.

Protocol: one JSON object per line in, one per line out.
    in   {"image": "<base64 png>", "prompt": "..."}
    out  {"text": "..."} or {"error": "..."}
    "READY" is printed once the model is loaded.

The model is loaded ONCE and reused for every crop. Loading per crop would
add roughly a minute each on CPU.

    python tools/qwen_worker.py --model D:\\Arabic-Qwen3.5-OCR-v4
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Windows defaults stdio to cp1252/cp1256, which cannot represent Arabic:
# writing raises "charmap codec can't encode", and the parent then fails to
# decode the bytes as UTF-8. Both sides must agree on UTF-8 explicitly.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stdin.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

MAX_SIDE = 768
MULTIPLE = 64


def fit(img: Image.Image) -> Image.Image:
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    w, h = img.size
    nw = ((w + MULTIPLE - 1) // MULTIPLE) * MULTIPLE
    nh = ((h + MULTIPLE - 1) // MULTIPLE) * MULTIPLE
    return img.resize((nw, nh), Image.Resampling.LANCZOS) if (nw, nh) != (w, h) else img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    import transformers
    from transformers import AutoProcessor

    cls = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
    if cls is None:
        cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if cls is None:
        print(json.dumps({"error": f"transformers {transformers.__version__} "
                                   "cannot load Qwen3.5 (needs >=5.0)"}),
              flush=True)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = cls.from_pretrained(
        args.model, dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True, low_cpu_mem_usage=True,
    ).eval()

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        process_vision_info = None

    # The parent blocks on this line before sending any work.
    print("READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            raw = base64.b64decode(req["image"])
            img = fit(Image.open(io.BytesIO(raw)).convert("RGB"))
            prompt = req.get("prompt") or "اقرأ النص في هذه الصورة."

            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ]}]
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            images = (process_vision_info(messages)[0]
                      if process_vision_info else [img])
            inputs = processor(text=[text_input], images=images,
                               padding=True, return_tensors="pt").to(device)

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )
            n = inputs.input_ids.shape[1]
            text = processor.batch_decode(
                out[:, n:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0].strip()
            print(json.dumps({"text": text}, ensure_ascii=False), flush=True)
        except Exception as exc:                          # noqa: BLE001
            # Never die on one bad crop: the parent needs the rest of the page.
            print(json.dumps({"error": str(exc)}, ensure_ascii=False),
                  flush=True)


if __name__ == "__main__":
    main()