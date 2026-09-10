"""Dataset-backed image captioning with a deterministic CPU fallback.

The optional caption model is intentionally small: it stores normalized image
features and captions, then retrieves the nearest training image at inference
time. This makes the path reproducible and useful on CPU without requiring a
large downloaded VLM. The existing heuristic ``caption(evidence)`` API remains
fully compatible.
"""
from __future__ import annotations

import ast
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from models.base_vlm import ImageEvidence

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class CaptionRecord:
    image_path: str
    captions: Tuple[str, ...]


def _parse_caption_cell(value: str) -> List[str]:
    """Parse CSV cells containing a Python/NumPy-style list of captions."""
    text = str(value or "").strip()
    if not text:
        return []
    # NumPy's array string representation omits commas between quoted items.
    # Restore those separators before literal evaluation; this also preserves
    # apostrophes inside a double-quoted caption.
    normalized = re.sub(r"""(['"])\s+(?=['"])""", r"\1, ", text)
    try:
        parsed = ast.literal_eval(normalized)
        if isinstance(parsed, (list, tuple)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (SyntaxError, ValueError):
        pass
    # Handles cells such as: ['caption one' 'caption two'] where commas were
    # omitted by NumPy's array string representation.
    parts = re.findall(r"""['"]((?:\\.|[^'"])*)['"]""", text)
    if parts:
        return [bytes(p, "utf-8").decode("unicode_escape").strip() for p in parts if p.strip()]
    return [text.strip("\"'")] if text.strip("\"'") else []


def _resolve_image(root: Path, relative_path: str) -> Optional[str]:
    candidate = Path(relative_path)
    paths = [candidate] if candidate.is_absolute() else [root / candidate, root / candidate.name]
    for path in paths:
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return str(path)
    return None


def discover_caption_csvs(dataset_dir: str) -> List[str]:
    """Return caption CSVs in deterministic train/valid/test order."""
    root = Path(dataset_dir)
    preferred = [root / name for name in ("train.csv", "valid.csv", "val.csv", "test.csv")]
    found = [str(path) for path in preferred if path.is_file()]
    if found:
        return found
    return sorted(str(path) for path in root.rglob("*.csv"))


def load_caption_records(dataset_dir: str, split: Optional[str] = None,
                         limit: Optional[int] = None) -> List[CaptionRecord]:
    """Load rows from caption CSVs, skipping missing images with a clear count."""
    root = Path(dataset_dir)
    csv_paths = discover_caption_csvs(dataset_dir)
    if split:
        wanted = split.lower()
        csv_paths = [path for path in csv_paths if Path(path).stem.lower() == wanted]
    records: List[CaptionRecord] = []
    missing = 0
    for csv_path in csv_paths:
        with open(csv_path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "filepath" not in reader.fieldnames:
                raise ValueError(f"{csv_path} must contain a 'filepath' column")
            caption_key = "captions" if "captions" in reader.fieldnames else "caption"
            if caption_key not in reader.fieldnames:
                raise ValueError(f"{csv_path} must contain 'captions' or 'caption'")
            for row in reader:
                captions = tuple(_parse_caption_cell(row.get(caption_key, "")))
                image_path = _resolve_image(root, row.get("filepath", ""))
                if not captions:
                    continue
                if image_path is None:
                    missing += 1
                    continue
                records.append(CaptionRecord(image_path, captions))
                if limit is not None and len(records) >= limit:
                    break
        if limit is not None and len(records) >= limit:
            break
    if not records:
        detail = f"; {missing} image paths were missing" if missing else ""
        raise ValueError(f"No usable caption records found in {dataset_dir}{detail}")
    return records


def preprocess_image(image: Any, size: int = 32) -> np.ndarray:
    """Resize an image and return a normalized RGB float array."""
    from PIL import Image

    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        image = Image.fromarray(np.clip(array[..., :3] * (255 if array.max() <= 1 else 1), 0, 255).astype(np.uint8))
    image = image.convert("RGB").resize((size, size))
    return np.asarray(image, dtype=np.float32) / 255.0


def image_features(image: Any) -> np.ndarray:
    """Extract compact color/spatial features suitable for CPU retrieval."""
    arr = preprocess_image(image)
    pooled = arr.reshape(8, 4, 8, 4, 3).mean(axis=(1, 3))
    stats = np.concatenate([arr.mean(axis=(0, 1)), arr.std(axis=(0, 1))])
    hist = np.concatenate([np.histogram(arr[..., i], bins=8, range=(0, 1), density=True)[0] / 8
                           for i in range(3)])
    feature = np.concatenate([pooled.ravel(), stats, hist]).astype(np.float32)
    norm = np.linalg.norm(feature)
    return feature / norm if norm else feature


class CaptionModel:
    """Nearest-neighbor caption model trained from CSV/image records."""

    def __init__(self, features: np.ndarray, captions: Sequence[str], metadata: Optional[dict] = None):
        if len(features) != len(captions) or not len(features):
            raise ValueError("CaptionModel requires at least one feature/caption pair")
        self.features = np.asarray(features, dtype=np.float32)
        self.captions = list(captions)
        self.metadata = metadata or {}

    @classmethod
    def train(cls, records: Sequence[CaptionRecord], seed: int = 0) -> "CaptionModel":
        rng = np.random.RandomState(seed)
        features, captions = [], []
        for record in records:
            features.append(image_features(record.image_path))
            # One deterministic caption per image keeps checkpoints compact.
            captions.append(record.captions[int(rng.randint(len(record.captions)))])
        return cls(np.stack(features), captions, {"seed": seed, "records": len(records)})

    def predict(self, image: Any) -> Dict[str, Any]:
        feature = image_features(image)
        distances = np.linalg.norm(self.features - feature[None, :], axis=1)
        index = int(np.argmin(distances))
        confidence = float(np.exp(-float(distances[index]) * 3.0))
        return {"caption": self.captions[index], "confidence": round(max(0.05, min(0.98, confidence)), 2),
                "match_index": index, "distance": float(distances[index]), "backend": "dataset-nearest-neighbor"}

    def save(self, path: str) -> None:
        payload = {"features": self.features.tolist(), "captions": self.captions, "metadata": self.metadata}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    @classmethod
    def load(cls, path: str) -> "CaptionModel":
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(np.asarray(payload["features"], dtype=np.float32), payload["captions"], payload.get("metadata"))


def caption(evidence: ImageEvidence, image: Any = None,
            model: Optional[CaptionModel] = None) -> Dict[str, Any]:
    if model is not None and image is not None:
        result = model.predict(image)
        result["caption"] = f"This image appears to show: {result['caption'].strip()}"
        result["proportions"] = evidence.proportions
        return result

    proportions = evidence.proportions
    ranked = [(c, f) for c, f in sorted(proportions.items(), key=lambda kv: kv[1], reverse=True) if f > 0.03]
    if not ranked:
        text = "The image looks mostly uniform, with no clearly visible major feature."
        return {"caption": text, "confidence": 0.35, "proportions": proportions, "backend": "heuristic"}
    parts = [f"{cls.replace('_', ' ')} (~{frac*100:.0f}%)" for cls, frac in ranked[:3]]
    body = parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])}, and {parts[-1]}"
    note = ""
    if proportions.get("water", 0.0) > 0.15:
        note = " Water is only an RGB-based estimate; infrared data is needed to confirm it."
    return {"caption": f"Most of the image shows {body}. These are the main types of land or objects visible.{note}",
            "confidence": round(min(0.9, 0.5 + ranked[0][1]), 2), "proportions": proportions, "backend": "heuristic"}
