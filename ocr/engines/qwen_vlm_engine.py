# ocr/engines/qwen_vlm_engine.py
"""Arabic-Qwen3.5-OCR-v4 as an OCREngine.

This is a *recognition-only* model: it reads whatever is in the image it is
handed and has no notion of layout. On a full invoice page it hallucinates
(measured: Chinese glyphs and LaTeX on this project's sample). Feed it a single
field crop instead — see HybridSuryaQwenEngine, which pairs Surya's detector
with this engine.

Three backends, chosen by `backend` in engine_params:

  "subprocess"    RECOMMENDED. Spawns tools/qwen_worker.py using a second
                  venv's python.exe, which has transformers>=5.0, and talks to
                  it over stdin/stdout. Needs no server and no GGUF
                  conversion; the HF safetensors are used as downloaded. The
                  model loads once and is reused for the whole batch.

  "http"          talks to a local Ollama / llama-server. Needs the model
                  converted to GGUF and a server running on `host`.

  "transformers"  loads the safetensors in-process. Requires transformers>=5.0,
                  which CONFLICTS with surya-ocr 0.17.1, so this only works in
                  a venv where Surya is absent.

Every heavy import is deliberately lazy (inside __init__, not at module scope)
so that merely importing this module never breaks the Surya-only environment.
"""

from __future__ import annotations

import base64
import io
import json
import os

import numpy as np
from PIL import Image

from core.domain.geometry import BoundingBox
from core.domain.image_payload import ImagePayload
from core.domain.ocr import OCRFragment, OCRResult
from ocr.registry import engine_registry

# Prompts are Arabic because the model was fine-tuned on Arabic instructions.
# The "number" prompt exists because the model transliterates Arabic-Indic
# digits into look-alike Latin letters: ٥->O, ٧->V, ٨->A. Naming the mapping
# explicitly is what suppresses it.
PROMPTS = {
    "number": (
        "هذه خانة من فاتورة، تحتوي رقماً مكتوباً بخط اليد بالأرقام الهندية "
        "(٠١٢٣٤٥٦٧٨٩) وقد يحتوي فاصلة عشرية. اكتب الرقم فقط بالأرقام "
        "الإنجليزية 0-9. لا تكتب أي حرف أبداً. إن كانت الخانة فارغة أو غير "
        "مقروءة فاكتب: EMPTY"
    ),
    "arabic_text": (
        "هذه خانة من فاتورة، تحتوي وصف منتج مكتوباً بخط اليد بالعربية. "
        "اكتب النص كما هو تماماً. قد تظهر أسماء ماركات أجنبية. "
        "إن كانت الخانة فارغة أو غير مقروءة فاكتب: EMPTY"
    ),
    # A date is NOT a number. The "number" prompt above forbids every non-digit
    # character, which strips the separators out of "٢٠٢٦/٨/١٥" and leaves an
    # eight-digit run nothing downstream can parse back into a date.
    "date": (
        "هذه خانة من فاتورة، تحتوي تاريخاً مكتوباً بخط اليد. "
        "اكتب التاريخ بالأرقام الإنجليزية 0-9 مع الفواصل كما تظهر "
        "(مثال: 2026/8/15). لا تكتب أي حرف. "
        "إن كانت الخانة فارغة أو غير مقروءة فاكتب: EMPTY"
    ),
    "any": "اقرأ النص في هذه الصورة كاملاً من البداية إلى النهاية.",
}

MAX_SIDE = 768   # the model card's own prepare_image cap
MULTIPLE = 64    # the model requires both dimensions to be multiples of 64


def to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return Image.fromarray(arr).convert("RGB")


def fit_for_model(img: Image.Image) -> Image.Image:
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    w, h = img.size
    nw = ((w + MULTIPLE - 1) // MULTIPLE) * MULTIPLE
    nh = ((h + MULTIPLE - 1) // MULTIPLE) * MULTIPLE
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    return img


@engine_registry.register("qwen_vlm")
class QwenVLMEngine:
    def __init__(
        self,
        model_path: str = r"models\qwen",
        backend: str = "http",
        host: str = "http://localhost:11434",
        model_name: str = "arabic-ocr",
        kind: str = "any",
        max_new_tokens: int = 64,
        worker_python: str = r"qwen_venv\Scripts\python.exe",
        worker_script: str = "tools/qwen_worker.py",
        **engine_params,
    ):
        self.model_path = model_path
        self.backend = backend
        self.host = host.rstrip("/")
        self.model_name = model_name
        self.kind = kind
        self.max_new_tokens = max_new_tokens
        self.worker_python = worker_python
        self.worker_script = worker_script

        self._proc = None
        if backend == "transformers":
            self._init_transformers()
        elif backend == "subprocess":
            self._init_subprocess(engine_params)
        elif backend != "http":
            raise ValueError(
                f"unknown backend: {backend!r} "
                '(expected "subprocess", "http" or "transformers")')

    # -- backends ---------------------------------------------------------
    def _init_transformers(self):
        import torch
        import transformers
        from transformers import AutoProcessor

        cls = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
        if cls is None:
            raise RuntimeError(
                f"transformers {transformers.__version__} cannot load Qwen3.5 "
                "(needs >=5.0). This venv is pinned to <5.0 for surya-ocr; use "
                'backend: "http" here, or run this engine in a separate venv.'
            )
        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True)
        self._model = cls.from_pretrained(
            self.model_path,
            dtype=self._dtype,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()

    def _read_transformers(self, img: Image.Image, prompt: str) -> str:
        from qwen_vl_utils import process_vision_info

        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}]
        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self._processor(text=[text_input], images=image_inputs,
                                 padding=True,
                                 return_tensors="pt").to(self._device)
        with self._torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,            # same crop must give same answer
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=self._processor.tokenizer.pad_token_id,
                eos_token_id=self._processor.tokenizer.eos_token_id,
            )
        n = inputs.input_ids.shape[1]
        return self._processor.batch_decode(
            out[:, n:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0].strip()

    def _init_subprocess(self, engine_params):
        """Spawn the worker in the OTHER venv and wait for it to load.

        This is the backend that needs no server and no GGUF conversion: the
        HF safetensors are used directly, just in a process where
        transformers>=5.0 is installed.
        """
        import atexit
        import subprocess

        if not os.path.exists(self.worker_python):
            raise FileNotFoundError(
                f"worker_python not found: {self.worker_python}. Point it at "
                "the python.exe of a venv that has transformers>=5.0.")
        if not os.path.exists(self.worker_script):
            raise FileNotFoundError(f"worker_script not found: {self.worker_script}")

        cmd = [self.worker_python, self.worker_script,
               "--model", self.model_path,
               "--max-new-tokens", str(self.max_new_tokens)]
        # PYTHONIOENCODING belts-and-braces with the worker's own
        # reconfigure(): on Windows the child would otherwise inherit a cp125x
        # console encoding and mangle every Arabic reply.
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None,                 # let the worker's errors reach the console
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
        )
        atexit.register(self._shutdown)

        print(f"[qwen] loading model in worker ({self.model_path}) ...")
        # transformers prints assorted warnings to STDOUT during load (e.g.
        # "[ERROR] `min_frames` is part of Qwen3VLVideoProcessorInitKwargs..."
        # which is a harmless docstring check, not a failure). Skip every line
        # until the worker's own READY sentinel, instead of treating the first
        # line as the answer.
        noise = []
        while True:
            line = self._proc.stdout.readline()
            if not line:
                tail = "\n".join(noise[-8:]) or "(no output)"
                raise RuntimeError(
                    f"worker exited before becoming ready. Last output:\n{tail}")
            line = line.strip()
            if line == "READY":
                break
            if line:
                noise.append(line)
                print(f"[qwen worker] {line}")
        print("[qwen] worker ready.")

    def _shutdown(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=10)
            except Exception:                              # noqa: BLE001
                self._proc.kill()

    def _read_subprocess(self, img: Image.Image, prompt: str) -> str:
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("qwen worker is not running")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        req = {"image": base64.b64encode(buf.getvalue()).decode(),
               "prompt": prompt}
        self._proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        # Same defence as at startup: a stray transformers warning on stdout
        # must not be parsed as the reply. Read until a line parses as JSON.
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("qwen worker closed unexpectedly")
            line = line.strip()
            if not line:
                continue
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                print(f"[qwen worker] {line}")
                continue
            break
        if "error" in resp:
            raise RuntimeError(resp["error"])
        return resp.get("text", "").strip()

    def _read_http(self, img: Image.Image, prompt: str) -> str:
        import requests

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        r = requests.post(f"{self.host}/api/generate", json={
            "model": self.model_name,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
            "options": {"temperature": 0, "num_predict": self.max_new_tokens},
        }, timeout=600)
        r.raise_for_status()
        return r.json().get("response", "").strip()

    # -- public API -------------------------------------------------------
    def read(self, img: Image.Image, kind: str | None = None) -> str:
        prompt = PROMPTS.get(kind or self.kind, PROMPTS["any"])
        img = fit_for_model(img)
        if self.backend == "transformers":
            return self._read_transformers(img, prompt)
        if self.backend == "subprocess":
            return self._read_subprocess(img, prompt)
        return self._read_http(img, prompt)

    def recognize(self, image: ImagePayload) -> OCRResult:
        """Whole-image recognition. The model returns text with no coordinates,
        so a single fragment spanning the full frame is the honest
        representation — do not fabricate per-line boxes it never produced."""
        pil = to_pil(image.image)
        text = self.read(pil)
        h, w = image.image.shape[:2]
        fragments = [OCRFragment(
            text=text,
            bbox=BoundingBox(x=0, y=0, w=w, h=h),
            confidence=None,
        )]
        return OCRResult(fragments=fragments,
                         engine_name="qwen_vlm",
                         raw_engine_output=text)