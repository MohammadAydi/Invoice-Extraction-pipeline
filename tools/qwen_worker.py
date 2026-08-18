# tools/qwen_worker.py
"""Long-lived VLM worker, run inside the transformers>=5.0 venv.

Exists because Qwen3.5 needs transformers>=5.0 while surya-ocr 0.17.1 pins
<5.0. They cannot share a process, so the main pipeline spawns this script
with the other venv's python.exe and talks to it over stdin/stdout.

Protocol: one JSON object per line in, one per line out.
    in   {"image": "<base64 png>", "prompt": "...", "max_new_tokens": 16}
         max_new_tokens is optional; falls back to the --max-new-tokens flag.
    out  {"text": "...", "ms": 123} or {"error": "..."}
    "READY" is printed once the model is loaded.

The model is loaded ONCE and reused for every crop. Loading per crop would
add roughly a minute each on CPU.

Two architectures are supported and auto-detected from the model's own
config.json, because getting this wrong produces silently degraded output
rather than an error:

  qwen3_5      Arabic-Qwen3.5-OCR-v4 and friends. The model card requires
               both image dimensions to be multiples of 64.
  qwen2_5_vl   Arabic-handwritten-OCR-*-Qwen2.5-VL-3B. Its processor calls
               smart_resize internally on a 28px patch grid, so ANY resize
               here is a second, pointless distortion on top of its own.

    python tools/qwen_worker.py --model D:\\Arabic-Qwen3.5-OCR-v4
    python tools/qwen_worker.py --model D:\\Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v3
    python tools/qwen_worker.py --model D:\\some-model --arch qwen2_5_vl   # override
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
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

# model_type in config.json -> transformers class name.
# Add a row here to support another architecture; nothing else changes.
ARCH_CLASSES = {
    "qwen3_5":    "Qwen3_5ForConditionalGeneration",
    "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration",
}
FALLBACK_CLASS = "AutoModelForImageTextToText"


# ---------------------------------------------------------------- inspection

def inspect_model(model_dir: str) -> dict:
    """Reads the model's own config.json. Never guess from the folder name --
    a renamed directory would silently select the wrong preprocessing."""
    info = {"arch": None, "quantized": False, "quant_config": None}
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.exists():
        return info
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:                                     # noqa: BLE001
        return info

    info["arch"] = cfg.get("model_type")
    qc = cfg.get("quantization_config")
    if qc:
        info["quantized"] = True
        # Keep the FULL dict, not just quant_method: the loader below reuses
        # the exact saved parameters (nf4 vs fp4, double quant on/off, compute
        # dtype). Building a fresh BitsAndBytesConfig from scratch would
        # silently mismatch the scheme the weights were saved with, which
        # corrupts output without raising.
        info["quant_config"] = qc
    return info


# -------------------------------------------------------------- image sizing

def fit(img: Image.Image, arch: str) -> Image.Image:
    """Sizes the crop for the target architecture.

    This is the ONLY place crops are resized. The engine deliberately sends
    them untouched in subprocess mode -- fitting on both sides stacked two
    resamples on the same glyphs.
    """
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)

    if arch != "qwen3_5":
        return img

    w, h = img.size
    nw = ((w + MULTIPLE - 1) // MULTIPLE) * MULTIPLE
    nh = ((h + MULTIPLE - 1) // MULTIPLE) * MULTIPLE
    if (nw, nh) == (w, h):
        return img

    # Pad, do not resize: resizing changes the aspect ratio (a 220x90 crop
    # became 256x128, a 42% vertical stretch), which distorts digit shapes.
    #
    # Pad with the crop's OWN border colour, not pure white. After
    # illumination normalisation the paper is light grey, not (255,255,255),
    # so white padding draws a visible rectangle the model reads as a stroke.
    edge = list(img.crop((0, 0, w, 1)).getdata())          # top row
    edge += list(img.crop((0, h - 1, w, h)).getdata())     # bottom row
    fill = tuple(sorted(c[i] for c in edge)[len(edge) // 2] for i in range(3))

    canvas = Image.new("RGB", (nw, nh), fill)
    canvas.paste(img, ((nw - w) // 2, (nh - h) // 2))
    return canvas


# ------------------------------------------------------------------- loading

def load_model(model_dir: str, arch: str, quantized: bool,
               quant_config: dict | None):
    import transformers
    from transformers import AutoProcessor

    cls_name = ARCH_CLASSES.get(arch, FALLBACK_CLASS)
    cls = getattr(transformers, cls_name, None)
    if cls is None and cls_name != FALLBACK_CLASS:
        print(f"[worker] {cls_name} unavailable, falling back to {FALLBACK_CLASS}",
              file=sys.stderr, flush=True)
        cls = getattr(transformers, FALLBACK_CLASS, None)
    if cls is None:
        raise RuntimeError(
            f"transformers {transformers.__version__} exposes neither "
            f"{cls_name} nor {FALLBACK_CLASS}. Qwen3.5 needs >=5.0.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if quantized and device != "cuda":
        # bitsandbytes kernels are CUDA-only. Failing here with a clear
        # message beats a confusing traceback deep inside the loader.
        raise RuntimeError(
            "This model is quantized (bitsandbytes) and requires a CUDA GPU. "
            "No GPU was detected. Use the unquantized model, or install a "
            "CUDA build of torch in this venv.")

    kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=True)
    if quantized:
        from transformers import BitsAndBytesConfig
        saved = dict(quant_config or {})
        saved.pop("quant_method", None)   # not a BitsAndBytesConfig field

        # Do NOT enable fp32 CPU offload. It was tried and made things worse:
        # with part of the model 4-bit on GPU and part fp32 on CPU, generation
        # came back as pure noise (every crop returned the same multilingual
        # token soup). Offload is not a way around insufficient VRAM here.
        saved.pop("llm_int8_enable_fp32_cpu_offload", None)

        try:
            quant_cfg = BitsAndBytesConfig(**saved)
        except TypeError as exc:
            raise RuntimeError(
                f"Could not reconstruct this model's saved 4-bit config "
                f"({exc}). Check config.json's quantization_config against "
                f"the BitsAndBytesConfig fields for this transformers version."
            ) from exc
        kwargs["quantization_config"] = quant_cfg

        # Pin the WHOLE model to GPU 0 instead of device_map="auto".
        # "auto" silently spills layers to CPU/disk when VRAM is tight, and a
        # partially-offloaded quantized model produces garbage rather than an
        # error. {"": 0} means: if it does not fit, fail loudly with an OOM.
        # A clear crash is the correct outcome -- silent corruption is not.
        kwargs["device_map"] = {"": 0}
        dtype_label = "4bit (pinned to GPU 0)"
    else:
        # bfloat16, not float16: both models are stored in BF16, which has a
        # wider exponent range. Down-casting to FP16 can saturate large
        # weights, and that shows up as changed output rather than an error --
        # which would quietly invalidate accuracy comparisons.
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        kwargs["dtype"] = dtype
        if device == "cuda":
            kwargs["device_map"] = {"": 0}
        dtype_label = str(dtype).replace("torch.", "")

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = cls.from_pretrained(model_dir, **kwargs).eval()

    # Some converted checkpoints omit tie_word_embeddings from config.json and
    # then report lm_head.weight as MISSING. Without the output head the model
    # computes normally and then generates from noise -- the symptom is every
    # crop returning similar multilingual gibberish.
    if getattr(model, "lm_head", None) is not None:
        if getattr(model.lm_head, "weight", None) is None:
            embed = model.get_input_embeddings()
            if embed is not None:
                model.lm_head.weight = embed.weight
                print("[worker] lm_head was missing; tied to input embeddings",
                      file=sys.stderr, flush=True)

    print(f"[worker] arch={arch} class={cls.__name__} device={device} "
          f"dtype={dtype_label}", file=sys.stderr, flush=True)
    return model, processor, device


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--arch", default=None,
                    help="override auto-detection: " + ", ".join(ARCH_CLASSES))
    args = ap.parse_args()

    info = inspect_model(args.model)
    arch = args.arch or info["arch"] or "qwen3_5"
    if args.arch and info["arch"] and args.arch != info["arch"]:
        print(f"[worker] WARNING: --arch {args.arch} overrides config.json "
              f"model_type {info['arch']}", file=sys.stderr, flush=True)
    if arch not in ARCH_CLASSES:
        print(f"[worker] unknown arch {arch!r}; using {FALLBACK_CLASS} and "
              "skipping arch-specific image sizing", file=sys.stderr, flush=True)

    try:
        model, processor, device = load_model(
            args.model, arch, info["quantized"], info["quant_config"])
    except Exception as exc:                              # noqa: BLE001
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
        started = time.perf_counter()
        try:
            req = json.loads(line)
            raw = base64.b64decode(req["image"])
            img = fit(Image.open(io.BytesIO(raw)).convert("RGB"), arch)
            prompt = req.get("prompt") or "اقرأ النص في هذه الصورة."

            # Per-request cap, so a numeric cell can be held to a few tokens
            # while a description keeps room. A 16-token ceiling on a number
            # column cuts a runaway hallucination at its start instead of
            # waiting out 64 tokens of it. Falls back to the CLI default.
            max_tokens = int(req.get("max_new_tokens") or args.max_new_tokens)

            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ]}]
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            images = (process_vision_info(messages)[0]
                      if process_vision_info else [img])
            inputs = processor(text=[text_input], images=images,
                               padding=True, return_tensors="pt").to(model.device)

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
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

            # `ms` is additive: the parent reads resp["text"] and ignores the
            # rest, so older callers keep working. It is the only per-crop
            # timing available anywhere in the pipeline.
            elapsed = round((time.perf_counter() - started) * 1000, 1)
            print(json.dumps({"text": text, "ms": elapsed}, ensure_ascii=False),
                  flush=True)
        except Exception as exc:                          # noqa: BLE001
            # Never die on one bad crop: the parent needs the rest of the page.
            print(json.dumps({"error": str(exc)}, ensure_ascii=False),
                  flush=True)


if __name__ == "__main__":
    main()