# tools/qwen_worker.py
"""Long-lived VLM worker, run inside the transformers>=5.0 venv.

Exists because Qwen3.5 needs transformers>=5.0 while surya-ocr 0.17.1 pins
<5.0. They cannot share a process, so the main pipeline spawns this script
with the other venv's python.exe and talks to it over stdin/stdout.

Protocol: one JSON object per line in, one per line out.
    in   {"image": "<base64 png>", "prompt": "..."}
    out  {"text": "..."} or {"error": "..."}
    "READY" is printed once the model is loaded.

The model is loaded ONCE and reused for every crop.

Two architectures, auto-detected from the model's own config.json:

  qwen3_5      Arabic-Qwen3.5-OCR-v4. Full precision, ~0.9B params.
               The model card requires both image dimensions to be
               multiples of 64.
  qwen2_5_vl   Arabic-handwritten-OCR-*-Qwen2.5-VL-3B, shipped pre-quantized.
               Its processor calls smart_resize internally on a 28px patch
               grid, so ANY resize here is a second, pointless distortion.

    python tools/qwen_worker.py --model D:\\Arabic-Qwen3.5-OCR-v4
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

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

ARCH_CLASSES = {
    "qwen3_5":    "Qwen3_5ForConditionalGeneration",
    "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration",
}
FALLBACK_CLASS = "AutoModelForImageTextToText"


def inspect_model(model_dir: str) -> dict:
    """Reads the model's own config.json.

    Two things matter and both must come from the file, never from a guess:
    the architecture (wrong class = wrong preprocessing) and whether the
    weights are ALREADY quantized (see load_model for why that decides
    everything).
    """
    info = {"arch": "qwen3_5", "quantized": False}
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.exists():
        return info
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:                                      # noqa: BLE001
        return info

    info["arch"] = cfg.get("model_type") or "qwen3_5"
    info["quantized"] = bool(cfg.get("quantization_config"))
    return info


def fit(img: Image.Image, arch: str) -> Image.Image:
    """Sizes the crop for the target architecture.

    qwen3_5 RESIZES to a multiple of 64 rather than padding to it. Padding is
    the more principled choice -- resizing changes the aspect ratio -- but it
    was measured and it lost: 5/12 cells correct with resize, 3/12 with
    padding, on the identical image. The measurement rules. Do not
    reintroduce padding without a new measurement.
    """
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)

    if arch != "qwen3_5":
        return img

    w, h = img.size
    nw = ((w + MULTIPLE - 1) // MULTIPLE) * MULTIPLE
    nh = ((h + MULTIPLE - 1) // MULTIPLE) * MULTIPLE
    return (img.resize((nw, nh), Image.Resampling.LANCZOS)
            if (nw, nh) != (w, h) else img)


def load_model(model_dir: str, arch: str, quantized: bool):
    import transformers
    from transformers import AutoProcessor

    cls_name = ARCH_CLASSES.get(arch, FALLBACK_CLASS)
    cls = getattr(transformers, cls_name, None)
    if cls is None:
        cls = getattr(transformers, FALLBACK_CLASS, None)
    if cls is None:
        raise RuntimeError(
            f"transformers {transformers.__version__} exposes neither "
            f"{cls_name} nor {FALLBACK_CLASS}. Qwen3.5 needs >=5.0.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}

    if quantized:
        # The weights are ALREADY 4-bit; their scheme lives in config.json and
        # transformers applies it automatically. Passing a fresh
        # BitsAndBytesConfig here is ignored with a warning at best, and risks
        # a scheme mismatch at worst -- so do not pass one.
        #
        # device_map={"": 0} pins the whole model to GPU 0. "auto" silently
        # spills layers to CPU when VRAM is tight, and a partially-offloaded
        # quantized model generates noise rather than raising. A clear OOM is
        # the correct failure here.
        kwargs["device_map"] = {"": 0}
        label = "4bit (from the model's own config)"
    else:
        # CRITICAL: never quantize a model that was not shipped quantized.
        #
        # Forcing load_in_4bit on full-precision Arabic-Qwen3.5-OCR-v4 was
        # measured and it degrades reading badly: ١٣٢٥٠ read as "۱۳۵۰"
        # unquantized became "12/5/6"; ٥١٧٫٦٥ went from "017,68" (two digits
        # off, and arithmetically repairable) to "517,86,42" (beyond repair).
        # Three quarters of the weight precision is thrown away for a model
        # that fits in VRAM comfortably at ~3.6 GB.
        #
        # float32, not float16: the 5/12 baseline was measured at float32 on
        # CPU. float16 is a narrower exponent range than the BF16 the weights
        # were stored in, and can saturate the large ones.
        kwargs["dtype"] = torch.float32
        if device == "cuda":
            kwargs["device_map"] = {"": 0}
        label = "float32 (unquantized -- left at full precision)"

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = cls.from_pretrained(model_dir, **kwargs).eval()

    print(f"[worker] arch={arch} class={cls.__name__} device={device} "
          f"weights={label}", file=sys.stderr, flush=True)
    return model, processor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--arch", default=None,
                    help="override auto-detection: " + ", ".join(ARCH_CLASSES))
    args = ap.parse_args()

    info = inspect_model(args.model)
    arch = args.arch or info["arch"]

    try:
        model, processor = load_model(args.model, arch, info["quantized"])
    except Exception as exc:                               # noqa: BLE001
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), flush=True)
        sys.exit(1)

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
            img = fit(Image.open(io.BytesIO(raw)).convert("RGB"), arch)
            prompt = req.get("prompt") or "اقرأ النص في هذه الصورة."

            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ]}]
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            images = (process_vision_info(messages)[0]
                      if process_vision_info else [img])
            # model.device, not a bare device string: with device_map the
            # model knows where its own first layer lives.
            inputs = processor(text=[text_input], images=images,
                               padding=True,
                               return_tensors="pt").to(model.device)

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,        # same crop must give same answer
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
        except Exception as exc:                           # noqa: BLE001
            # Never die on one bad crop: the parent needs the rest of the page.
            print(json.dumps({"error": str(exc)}, ensure_ascii=False),
                  flush=True)


if __name__ == "__main__":
    main()