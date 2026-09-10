"""
api/hf_config.py
-----------------
Environment-driven configuration for the Hugging Face vision-language model.

Nothing here is hard-coded. The token in particular is read only from the
environment (optionally seeded from a git-ignored `.env`), never from source,
and never sent anywhere except huggingface.co.

Recognised variables
--------------------
HF_TOKEN            Hugging Face access token. Required for private/gated repos.
                    Also accepted as HUGGING_FACE_HUB_TOKEN or HUGGINGFACEHUB_API_TOKEN.
HF_MODEL_ID         Adapter or model repo id.
                    e.g. aanandmodi/satquery-qwen3vl-bigearthnet-txt-lora
HF_BASE_MODEL_ID    Base model for a LoRA adapter. Leave unset to read it from
                    the adapter's own adapter_config.json, which is the reliable
                    source since the adapter records what it was trained on.
HF_DEVICE           auto | cuda | mps | cpu     (default: auto)
HF_DTYPE            auto | bfloat16 | float16 | float32
HF_LOAD_IN_4BIT     1 to quantise to 4-bit (needs bitsandbytes + CUDA).
                    Cuts an 8B model from ~16 GB to ~6 GB of VRAM.
HF_MAX_NEW_TOKENS   Generation length cap (default 256).
HF_TEMPERATURE      Sampling temperature (default 0.2 — this is an analysis
                    tool, so near-greedy decoding is the right default).
HF_MAX_IMAGE_PX     Longest edge fed to the vision tower (default 768).
HF_LOCAL_ONLY       1 to forbid network access and use only the local cache.
HF_ENABLED          0 to skip the model entirely and run on local specialists.
HF_EAGER_LOAD       1 to load weights at startup instead of on first request.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

_TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN")


def _as_bool(value: Optional[str], default: bool) -> bool:
    if value is None or not str(value).strip():
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on", "y")


def _as_int(value: Optional[str], default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _as_float(value: Optional[str], default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def load_dotenv(path: str = ".env", override: bool = False) -> Dict[str, str]:
    """Minimal `.env` reader.

    Dependency-free on purpose: parsing `KEY=value` does not justify a package,
    and the app must start in a bare Colab cell. Real environment variables win
    unless `override=True`, so `HF_TOKEN=... python app.py` behaves as expected.
    """
    loaded: Dict[str, str] = {}
    if not os.path.exists(path):
        return loaded
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if not key:
                    continue
                if override or key not in os.environ:
                    os.environ[key] = value
                loaded[key] = value
    except OSError:
        return loaded
    return loaded


def resolve_device(requested: str = "auto") -> str:
    """Pick the best available device, honouring an explicit request."""
    requested = (requested or "auto").lower()
    if requested in ("cuda", "mps", "cpu"):
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


@dataclass
class HFConfig:
    """Resolved configuration for one Hugging Face model."""

    model_id: str = ""
    base_model_id: Optional[str] = None
    token: Optional[str] = None
    device: str = "auto"
    dtype: str = "auto"
    load_in_4bit: bool = False
    max_new_tokens: int = 256
    temperature: float = 0.2
    max_image_px: int = 768
    local_files_only: bool = False
    enabled: bool = True
    eager_load: bool = False
    cache_dir: Optional[str] = None
    extra: Dict[str, str] = field(default_factory=dict)

    @property
    def configured(self) -> bool:
        return bool(self.model_id) and self.enabled

    def redacted(self) -> Dict[str, object]:
        """Config summary safe for logs, the API, and the UI.

        The token is reported as present/absent only. It is never echoed, not
        even partially — a prefix is enough to correlate against a leak.
        """
        return {
            "model_id": self.model_id or "(not configured)",
            "base_model_id": self.base_model_id or "(read from adapter_config.json)",
            "token": "set" if self.token else "none",
            "device": self.device,
            "dtype": self.dtype,
            "load_in_4bit": self.load_in_4bit,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "max_image_px": self.max_image_px,
            "local_files_only": self.local_files_only,
            "enabled": self.enabled,
        }


def load_config(env_file: str = ".env") -> HFConfig:
    """Build an HFConfig from the environment (+ optional .env file)."""
    load_dotenv(env_file)

    token = None
    for name in _TOKEN_VARS:
        candidate = (os.environ.get(name) or "").strip()
        if candidate:
            token = candidate
            break

    requested_device = (os.environ.get("HF_DEVICE") or "auto").strip().lower()

    return HFConfig(
        model_id=(os.environ.get("HF_MODEL_ID") or "").strip(),
        base_model_id=(os.environ.get("HF_BASE_MODEL_ID") or "").strip() or None,
        token=token,
        device=resolve_device(requested_device),
        dtype=(os.environ.get("HF_DTYPE") or "auto").strip().lower(),
        load_in_4bit=_as_bool(os.environ.get("HF_LOAD_IN_4BIT"), False),
        max_new_tokens=_as_int(os.environ.get("HF_MAX_NEW_TOKENS"), 256),
        temperature=_as_float(os.environ.get("HF_TEMPERATURE"), 0.2),
        max_image_px=_as_int(os.environ.get("HF_MAX_IMAGE_PX"), 768),
        local_files_only=_as_bool(os.environ.get("HF_LOCAL_ONLY"), False),
        enabled=_as_bool(os.environ.get("HF_ENABLED"), True),
        eager_load=_as_bool(os.environ.get("HF_EAGER_LOAD"), False),
        cache_dir=(os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE") or None),
    )
