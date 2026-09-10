"""
api/hf_model.py
----------------
Runs the Hugging Face vision-language model that answers questions about
satellite imagery.

Why the model is loaded locally rather than called over an API
--------------------------------------------------------------
`aanandmodi/satquery-qwen3vl-bigearthnet-txt-lora` is a PEFT LoRA adapter on a
Qwen3-VL base. Hugging Face's serverless Inference API does not serve arbitrary
LoRA adapters on multimodal bases, so there is no free hosted endpoint to call.
The two real options are a paid dedicated Inference Endpoint, or loading the
weights in-process — which is what this module does, because it works on a
laptop with a GPU, in Colab, and on a lab workstation without extra billing.

That choice has consequences this module handles explicitly:

* **Weights are large.** The base model is several gigabytes. Loading happens
  lazily in a background thread on first use, so the web server starts in under
  a second and the UI can show real load progress instead of hanging.
* **The base model must be discovered.** A LoRA adapter records what it was
  trained on in `adapter_config.json`. Reading that is more reliable than
  hard-coding a guess, and it keeps working if the adapter is retrained on a
  different base.
* **A GPU is not guaranteed.** CPU inference works but is slow enough to be
  worth warning about rather than silently enduring, so the status object says
  so plainly.
* **Loading can fail** — no token, gated repo, out of memory, no torch. Every
  failure becomes a status message, never an exception that takes down the
  server. SatQuery's local specialists keep answering, clearly labelled.

Honesty rules
-------------
`ModelAnswer.ok` is False whenever the model did not produce a usable answer.
This module never invents a label, never substitutes a placeholder, and never
reports a class confidence the model did not emit. It does report a *generation
confidence* — the mean token probability of the produced text — which is a real
measurement of the decoder's certainty, and is labelled as exactly that rather
than being passed off as classification confidence.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from api.hf_config import HFConfig, load_config

# Status values reported to the UI.
IDLE = "idle"
LOADING = "loading"
READY = "ready"
ERROR = "error"
DISABLED = "disabled"


@dataclass
class ModelAnswer:
    """One response from the vision-language model."""

    ok: bool = False
    text: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    generation_confidence: Optional[float] = None
    latency_ms: Optional[float] = None
    cached: bool = False
    error: Optional[str] = None
    error_kind: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def summary_line(self) -> str:
        if not self.ok:
            return f"no result ({self.error or 'unknown error'})"
        bits = []
        if self.labels:
            bits.append(f"labels={', '.join(self.labels[:4])}")
        if self.generation_confidence is not None:
            bits.append(f"gen-confidence={self.generation_confidence:.2f}")
        if self.cached:
            bits.append("cached")
        elif self.latency_ms is not None:
            bits.append(f"{self.latency_ms / 1000:.1f}s")
        if not bits:
            bits.append("text returned")
        return ", ".join(bits)

    @classmethod
    def failure(cls, error: str, kind: str = "error") -> "ModelAnswer":
        return cls(ok=False, error=error, error_kind=kind)


# BigEarthNet-19 class vocabulary. Used only to spot which of these the model
# named in its own words — never to add a label the model did not produce.
BIGEARTHNET_CLASSES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]

SYSTEM_PROMPT = (
    "You are a remote-sensing image analyst. You are shown a satellite image "
    "and asked a question about it. Answer concisely and factually, describing "
    "only what is visible. Name the land-cover types you can identify. If you "
    "are unsure, say so rather than guessing."
)


class HFModelRunner:
    """Loads and runs the vision-language model. Thread-safe, lazily loaded."""

    def __init__(self, config: Optional[HFConfig] = None):
        self.config = config or load_config()
        self._model = None
        self._processor = None
        self._state = DISABLED if not self.config.enabled else IDLE
        self._message = (
            "Model disabled via HF_ENABLED=0."
            if not self.config.enabled
            else "Not loaded yet."
        )
        self._progress: List[str] = []
        self._resolved_base: Optional[str] = None
        self._is_adapter = False
        self._load_seconds: Optional[float] = None
        self._lock = threading.Lock()          # guards load state
        self._generate_lock = threading.Lock()  # one generation at a time
        self._load_thread: Optional[threading.Thread] = None
        self._cache: Dict[str, ModelAnswer] = {}

        if self.config.enabled and self.config.model_id and self.config.eager_load:
            self.ensure_loading()

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #
    @property
    def state(self) -> str:
        return self._state

    @property
    def ready(self) -> bool:
        return self._state == READY

    def status(self) -> Dict[str, Any]:
        """Everything the UI needs to render the model panel."""
        return {
            "state": self._state,
            "message": self._message,
            "progress": list(self._progress),
            "model_id": self.config.model_id or None,
            "base_model_id": self._resolved_base,
            "is_adapter": self._is_adapter,
            "device": self.config.device,
            "load_seconds": self._load_seconds,
            "config": self.config.redacted(),
            "warnings": self._warnings(),
        }

    def _warnings(self) -> List[str]:
        warnings: List[str] = []
        if self.config.device == "cpu" and self._state in (LOADING, READY):
            warnings.append(
                "Running on CPU. A single answer can take several minutes on a "
                "multi-billion-parameter vision model. A CUDA GPU, or "
                "HF_LOAD_IN_4BIT=1 on a smaller GPU, makes this usable."
            )
        if not self.config.token and self.config.model_id:
            warnings.append(
                "No Hugging Face token is set. This is fine for public repos and "
                "will fail with a 401 on a private or gated one."
            )
        return warnings

    def _note(self, message: str) -> None:
        self._progress.append(message)
        self._message = message

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def ensure_loading(self) -> None:
        """Kick off a background load if one has not started. Returns at once."""
        with self._lock:
            if self._state in (LOADING, READY, DISABLED):
                return
            if not self.config.model_id:
                self._state = ERROR
                self._message = "HF_MODEL_ID is not set."
                return
            self._state = LOADING
            self._progress = []
            self._note("Starting model load…")
            self._load_thread = threading.Thread(
                target=self._load, name="hf-model-load", daemon=True
            )
            self._load_thread.start()

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        """Block until the model is loaded. Used by CLI tools, not by the web app."""
        self.ensure_loading()
        thread = self._load_thread
        if thread is not None:
            thread.join(timeout)
        return self._state == READY

    def _load(self) -> None:
        started = time.time()
        try:
            import torch  # noqa: F401
        except ImportError:
            self._state = ERROR
            self._message = (
                "PyTorch is not installed. Install it for your platform from "
                "pytorch.org, then restart. SatQuery keeps working on its local "
                "specialists until then."
            )
            return

        try:
            self._note("Resolving the model repository…")
            self._is_adapter, base_id = self._resolve_repo()
            self._resolved_base = base_id or self.config.model_id

            if self._is_adapter:
                self._note(f"LoRA adapter detected. Base model: {self._resolved_base}")
            else:
                self._note("Standalone model detected (no LoRA adapter to apply).")

            self._note("Loading processor…")
            self._processor = self._load_processor()

            self._note(
                f"Loading weights onto {self.config.device}. The first run "
                f"downloads several GB and can take a while."
            )
            model = self._load_base_model()

            if self._is_adapter:
                self._note("Applying LoRA adapter…")
                from peft import PeftModel

                model = PeftModel.from_pretrained(
                    model, self.config.model_id, token=self.config.token
                )
                model.eval()

            self._model = model
            self._load_seconds = time.time() - started
            self._state = READY
            self._message = f"Model ready in {self._load_seconds:.0f}s."
            self._progress.append(self._message)
        except Exception as exc:  # loading must never kill the server
            self._state = ERROR
            self._message = self._explain_load_failure(exc)
            self._progress.append(self._message)

    def _resolve_repo(self) -> Tuple[bool, Optional[str]]:
        """Detect whether the repo is a LoRA adapter and find its base model.

        The adapter records its own base in `adapter_config.json`, which is
        authoritative — reading it beats guessing, and it survives the adapter
        being retrained against a different base.
        """
        if self.config.base_model_id:
            # An explicit override still needs the adapter/standalone question
            # answered, but the base is taken as given.
            return self._has_adapter_config(), self.config.base_model_id

        from huggingface_hub import hf_hub_download

        try:
            path = hf_hub_download(
                repo_id=self.config.model_id,
                filename="adapter_config.json",
                token=self.config.token,
                local_files_only=self.config.local_files_only,
            )
        except Exception:
            return False, self.config.model_id  # not an adapter, or not readable

        with open(path, "r", encoding="utf-8") as handle:
            adapter_config = json.load(handle)
        base = adapter_config.get("base_model_name_or_path")
        if not base:
            raise RuntimeError(
                "The adapter's adapter_config.json does not name a base model. "
                "Set HF_BASE_MODEL_ID explicitly (for example "
                "Qwen/Qwen3-VL-8B-Instruct)."
            )
        return True, str(base)

    def _has_adapter_config(self) -> bool:
        from huggingface_hub import hf_hub_download

        try:
            hf_hub_download(
                repo_id=self.config.model_id,
                filename="adapter_config.json",
                token=self.config.token,
                local_files_only=self.config.local_files_only,
            )
            return True
        except Exception:
            return False

    def _load_processor(self):
        """Load the processor, preferring the adapter's own copy if it ships one."""
        from transformers import AutoProcessor

        candidates = [self.config.model_id]
        if self._resolved_base and self._resolved_base != self.config.model_id:
            candidates.append(self._resolved_base)

        last_error: Optional[Exception] = None
        for repo in candidates:
            try:
                return AutoProcessor.from_pretrained(
                    repo,
                    token=self.config.token,
                    trust_remote_code=True,
                    local_files_only=self.config.local_files_only,
                )
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Could not load a processor for {candidates}: {last_error}")

    def _torch_dtype(self):
        import torch

        if self.config.dtype == "float32":
            return torch.float32
        if self.config.dtype == "float16":
            return torch.float16
        if self.config.dtype == "bfloat16":
            return torch.bfloat16
        # auto: bf16 on modern GPUs, fp32 on CPU (fp16 on CPU is slow and
        # frequently unsupported, which surprises people).
        if self.config.device == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if self.config.device == "mps":
            return torch.float16
        return torch.float32

    def _load_base_model(self):
        """Load the base weights, trying the multimodal classes in turn.

        Transformers has renamed the vision-language auto class more than once
        (AutoModelForImageTextToText, AutoModelForVision2Seq,
        AutoModelForMultimodalLM), so the correct one depends on the installed
        version. Trying them in order beats pinning a single version.
        """
        import transformers

        kwargs: Dict[str, Any] = {
            "token": self.config.token,
            "trust_remote_code": True,
            "local_files_only": self.config.local_files_only,
            "dtype": self._torch_dtype(),
        }

        if self.config.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                import torch

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                kwargs["device_map"] = "auto"
            except ImportError:
                self._note(
                    "HF_LOAD_IN_4BIT is set but bitsandbytes is not installed; "
                    "loading at full precision instead."
                )
        elif self.config.device == "cuda":
            kwargs["device_map"] = "auto"

        class_names = [
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForMultimodalLM",
            "AutoModelForCausalLM",
        ]
        last_error: Optional[Exception] = None
        for name in class_names:
            auto_class = getattr(transformers, name, None)
            if auto_class is None:
                continue
            try:
                model = auto_class.from_pretrained(self._resolved_base, **kwargs)
            except TypeError as exc:
                # Older transformers used torch_dtype rather than dtype.
                if "dtype" not in str(exc):
                    last_error = exc
                    continue
                retry = dict(kwargs)
                retry["torch_dtype"] = retry.pop("dtype")
                try:
                    model = auto_class.from_pretrained(self._resolved_base, **retry)
                except Exception as inner:
                    last_error = inner
                    continue
            except Exception as exc:
                last_error = exc
                continue

            if "device_map" not in kwargs and self.config.device != "cpu":
                model = model.to(self.config.device)
            model.eval()
            return model

        raise RuntimeError(
            f"Could not load {self._resolved_base} with any known auto class. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def _explain_load_failure(exc: Exception) -> str:
        """Turn a loading traceback into something actionable."""
        text = str(exc)
        lowered = text.lower()
        if "401" in text or "unauthorized" in lowered or "gated" in lowered:
            return (
                "Hugging Face rejected the request (401). The repository is "
                "private or gated: set a valid HF_TOKEN in .env, and make sure "
                "that token's account has been granted access to the repo."
            )
        if "404" in text or "not found" in lowered or "repositorynotfound" in lowered:
            return (
                "The repository was not found (404). Check HF_MODEL_ID for typos, "
                "and confirm the token can see it if the repo is private."
            )
        if "out of memory" in lowered or "cuda oom" in lowered:
            return (
                "Ran out of GPU memory. Set HF_LOAD_IN_4BIT=1 in .env to quantise "
                "the model, or switch to a smaller base with HF_BASE_MODEL_ID."
            )
        if "connectionerror" in lowered or "max retries" in lowered or "network" in lowered:
            return (
                "Could not reach huggingface.co. Check the network, or set "
                "HF_LOCAL_ONLY=1 if the weights are already in the local cache."
            )
        if "peft" in lowered and "no module" in lowered:
            return "The 'peft' package is required to apply a LoRA adapter: pip install peft"
        return f"Model load failed: {type(exc).__name__}: {text[:400]}"

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def analyze(
        self,
        image: np.ndarray,
        query: str,
        use_cache: bool = True,
        max_new_tokens: Optional[int] = None,
    ) -> ModelAnswer:
        """Ask the model about one image. Never raises for operational problems."""
        if not self.config.enabled:
            return ModelAnswer.failure(
                "The vision-language model is disabled (HF_ENABLED=0).", kind="disabled"
            )
        if not self.config.model_id:
            return ModelAnswer.failure("HF_MODEL_ID is not set.", kind="not_configured")

        if self._state != READY:
            self.ensure_loading()
            if self._state == ERROR:
                return ModelAnswer.failure(self._message, kind="load_error")
            return ModelAnswer.failure(
                f"The model is still loading. {self._message}", kind="loading"
            )

        try:
            pil_image = self._prepare_image(image)
        except Exception as exc:
            return ModelAnswer.failure(
                f"Unsupported or invalid image: {exc}", kind="invalid_image"
            )

        cache_key = None
        if use_cache:
            digest = _image_digest(pil_image)
            cache_key = f"{digest}|{query.strip().lower()}"
            hit = self._cache.get(cache_key)
            if hit is not None:
                clone = ModelAnswer(**dict(hit.__dict__))
                clone.cached = True
                return clone

        started = time.time()
        try:
            # One generation at a time: concurrent requests on a single GPU
            # produce OOM rather than speedup.
            with self._generate_lock:
                text, confidence = self._generate(
                    pil_image, query, max_new_tokens or self.config.max_new_tokens
                )
        except Exception as exc:
            lowered = str(exc).lower()
            kind = "oom" if "out of memory" in lowered else "inference_error"
            message = (
                "Ran out of memory during generation. Try HF_LOAD_IN_4BIT=1 or a "
                "lower HF_MAX_IMAGE_PX."
                if kind == "oom"
                else f"Generation failed: {type(exc).__name__}: {exc}"
            )
            return ModelAnswer.failure(message, kind=kind)

        answer = ModelAnswer(
            ok=bool(text and text.strip()),
            text=(text or "").strip() or None,
            generation_confidence=confidence,
            latency_ms=(time.time() - started) * 1000.0,
        )
        if not answer.ok:
            answer.error = "The model returned an empty response."
            answer.error_kind = "empty"
            return answer

        answer.labels = extract_labels(answer.text)
        parsed = _parse_embedded_json(answer.text)
        if parsed:
            answer.raw = parsed
            answer.scores = {
                str(k): float(v)
                for k, v in parsed.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }

        if cache_key:
            self._cache[cache_key] = answer
            if len(self._cache) > 32:
                self._cache.pop(next(iter(self._cache)))
        return answer

    def analyze_many(
        self, items: List[Tuple[np.ndarray, str]], max_new_tokens: Optional[int] = None
    ) -> List[ModelAnswer]:
        """Analyse several images in sequence.

        Deliberately sequential, not threaded: these calls share one GPU, so
        running them in parallel would contend for memory rather than overlap.
        """
        return [self.analyze(image, query, max_new_tokens=max_new_tokens)
                for image, query in items]

    def _prepare_image(self, array: np.ndarray):
        """Convert a SatQuery array into a PIL image sized for the vision tower."""
        from PIL import Image

        if array is None:
            raise ValueError("No image data supplied.")
        arr = np.asarray(array)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.ndim != 3:
            raise ValueError(f"Expected a 2D or 3D image array, got shape {arr.shape}.")

        # Multi-band imagery is reduced to its first three bands because the
        # vision tower expects RGB; single-band SAR is replicated to three.
        arr = arr[..., :3] if arr.shape[-1] >= 3 else np.repeat(arr[..., :1], 3, axis=-1)

        if arr.dtype != np.uint8:
            finite = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
            if finite.max() > 1.5:
                finite = finite / 255.0
            arr = (np.clip(finite, 0.0, 1.0) * 255.0).astype(np.uint8)

        image = Image.fromarray(arr, mode="RGB")
        limit = max(64, int(self.config.max_image_px))
        longest = max(image.width, image.height)
        if longest > limit:
            scale = limit / float(longest)
            image = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                Image.BILINEAR,
            )
        return image

    def _generate(self, image, query: str, max_new_tokens: int) -> Tuple[str, Optional[float]]:
        """Run one generation and measure the decoder's confidence in it."""
        import torch

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": query or "Describe this satellite image."},
                ],
            },
        ]

        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[prompt], images=[image], return_tensors="pt")

        device = getattr(self._model, "device", None) or self.config.device
        inputs = {
            key: (value.to(device) if hasattr(value, "to") else value)
            for key, value in inputs.items()
        }

        do_sample = self.config.temperature > 0.0
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=self.config.temperature if do_sample else None,
                output_scores=True,
                return_dict_in_generate=True,
            )

        sequences = outputs.sequences
        prompt_length = inputs["input_ids"].shape[1]
        generated = sequences[0][prompt_length:]
        text = self._processor.decode(generated, skip_special_tokens=True)

        # Generation confidence: the mean probability the decoder assigned to
        # the tokens it actually emitted. This is a genuine measurement of
        # decoder certainty — it is NOT a classification confidence, and the
        # UI labels it accordingly.
        confidence = None
        try:
            transition = self._model.compute_transition_scores(
                sequences, outputs.scores, normalize_logits=True
            )
            values = transition[0]
            values = values[torch.isfinite(values)]
            if values.numel():
                confidence = float(torch.exp(values.mean()).item())
        except Exception:
            confidence = None

        return text.strip(), confidence


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def extract_labels(text: str) -> List[str]:
    """Find which BigEarthNet classes the model named, in its own words.

    This only *recognises* vocabulary the model produced. It never adds a class
    the model did not mention, so the label list can never be richer than the
    model's actual output.
    """
    if not text:
        return []
    lowered = text.lower()
    found: List[str] = []
    for label in BIGEARTHNET_CLASSES:
        needle = label.lower()
        # Match the full class name, or its distinctive head for the long ones.
        head = needle.split(",")[0].split(" with ")[0].strip()
        if needle in lowered or (len(head) > 8 and head in lowered):
            found.append(label)
    return found


def _parse_embedded_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull a JSON object out of the model's reply, if it emitted one."""
    if not text or "{" not in text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        return None


def _image_digest(image) -> str:
    import hashlib
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return hashlib.sha1(buffer.getvalue()).hexdigest()


# --------------------------------------------------------------------------- #
# Process-wide singleton
# --------------------------------------------------------------------------- #
_runner_lock = threading.Lock()
_runner: Optional[HFModelRunner] = None


def get_runner(config: Optional[HFConfig] = None) -> HFModelRunner:
    global _runner
    with _runner_lock:
        if _runner is None or config is not None:
            _runner = HFModelRunner(config)
        return _runner


def reset_runner() -> None:
    global _runner
    with _runner_lock:
        _runner = None
