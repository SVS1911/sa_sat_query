"""
models/base_vlm.py
--------------------
The "remote-sensing-adapted vision-language component" required by the spec.

RemoteSensingVLM.encode() is the single seam where domain adaptation happens.
Three backends can be active, in order of precedence:
  1. `segmentation_backend` — a trained pixel-level CNN (models/cnn_segmentation.py)
     if you've trained one on labeled masks. Fully replaces the heuristic.
  2. `backend` (scene classifier) — a BigEarthNet-fine-tuned scene classifier
     (training/finetune_bigearthnet.py) whose predictions are blended into the
     heuristic's proportions.
  3. Heuristic spectral classifier (utils/spectral_indices.py) — the default,
     runs with zero external downloads.

`band_roles` (utils.spectral_indices.BandRoles) tells the heuristic and the
segmentation CNN which band index is Red/Green/Blue/NIR/SWIR1/SWIR2 — this
matters a lot for accuracy on real multi-band imagery (see spectral_indices.py
docstring for why guessing this was the root cause of false water detections
on Planetary Computer scenes) and is threaded through every call rather than
inferred silently.
"""
from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional

from utils.spectral_indices import classify_landcover, LAND_COVER_CLASSES, BandRoles


@dataclass
class ImageEvidence:
    class_map: np.ndarray
    proportions: Dict[str, float]
    embedding: Optional[np.ndarray] = None
    backend_name: str = "heuristic-spectral-v0"
    notes: list = field(default_factory=list)


class RemoteSensingVLM:
    """
    Swappable RS-adapted encoder. `weights_path`, if given and present on disk,
    is expected to be a checkpoint produced by training/finetune_bigearthnet.py
    (a real fine-tuned model); if absent, falls back to the heuristic backend
    so the app remains runnable out of the box.
    """

    def __init__(self, weights_path: Optional[str] = None, backend=None,
                 segmentation_backend=None):
        self.weights_path = weights_path
        self.backend = backend
        self.segmentation_backend = segmentation_backend
        self.using_finetuned = False

        if backend is None and weights_path and os.path.exists(weights_path):
            try:
                self.backend = self._load_finetuned(weights_path)
                self.using_finetuned = True
            except Exception as e:
                print(f"[RemoteSensingVLM] Could not load fine-tuned weights ({e}); "
                      f"falling back to heuristic spectral backend.")
                self.backend = None

    def _load_finetuned(self, path: str):
        """
        Loads a real checkpoint produced by training/finetune_bigearthnet.py: a small
        CNN trained on actual BigEarthNet imagery + BigEarthNet-19 multi-label
        annotations. Its scene-level (whole-patch) predictions are blended into the
        heuristic per-pixel class map's proportions (see FineTunedSceneClassifier
        below) — this is what upgrades VQA/captioning from pure heuristics to a
        real trained signal, while grounding/change still use the per-pixel
        heuristic map (BigEarthNet-19 labels are scene-level, not per-pixel).
        """
        import torch
        ckpt = torch.load(path, map_location="cpu")
        return FineTunedSceneClassifier(ckpt)

    def encode(self, arr: np.ndarray, band_roles: Optional[BandRoles] = None) -> ImageEvidence:
        if self.segmentation_backend is not None:
            class_map = self.segmentation_backend.predict_class_map(arr, band_roles)
            total = class_map.size
            proportions = {cls: float(np.sum(class_map == i)) / total
                           for i, cls in enumerate(LAND_COVER_CLASSES)}
            notes = [f"Using trained CNN segmentation backend "
                     f"({self.segmentation_backend.__class__.__name__}) for pixel-level classification."]
            return ImageEvidence(class_map=class_map, proportions=proportions, embedding=None,
                                  backend_name="cnn-segmentation", notes=notes)

        class_map, proportions, method_notes = classify_landcover(arr, band_roles)

        if self.backend is not None:
            # Real fine-tuned scene classifier available: blend its scene-level,
            # trained predictions into the heuristic per-pixel proportions.
            scene_scores = self.backend.predict_scene_scores(arr)
            proportions = _blend_proportions(proportions, scene_scores)
            notes = method_notes + [
                f"Blended heuristic per-pixel map with a BigEarthNet-fine-tuned "
                f"scene classifier (trained on {self.backend.n_source_classes} "
                f"BigEarthNet-19 classes, mapped to SatQuery's 5-class scheme)."
            ]
            return ImageEvidence(
                class_map=class_map, proportions=proportions, embedding=None,
                backend_name=f"finetuned-bigearthnet({self.backend.n_source_classes}-class)",
                notes=notes,
            )

        return ImageEvidence(
            class_map=class_map,
            proportions=proportions,
            embedding=None,
            backend_name="heuristic-spectral-v0",
            notes=method_notes + [
                "No fine-tuned weights loaded. See training/download_bigearthnet.py + "
                "training/finetune_bigearthnet.py (scene classifier) or "
                "training/train_segmentation_cnn.py (pixel-level, once you have masks) "
                "to adapt a real encoder."
            ],
        )

    def class_names(self):
        return LAND_COVER_CLASSES


class FineTunedSceneClassifier:
    """Wraps a training/finetune_bigearthnet.py checkpoint: runs the trained CNN
    on a resized copy of the input and maps its BigEarthNet-19 multi-label
    predictions into SatQuery's 5-class scheme via the checkpoint's saved
    class_to_satquery mapping."""

    def __init__(self, ckpt: dict):
        import torch
        from training.finetune_bigearthnet import SmallLandCoverCNN  # local import: torch-only path

        self.class_names = ckpt["class_names"]
        self.class_to_satquery = ckpt["class_to_satquery"]
        self.n_source_classes = len(self.class_names)

        self.model = SmallLandCoverCNN(n_classes=self.n_source_classes)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self._torch = torch

    def predict_scene_scores(self, arr: np.ndarray) -> Dict[str, float]:
        """Returns SatQuery-class -> aggregated predicted probability, aggregated
        over source BigEarthNet-19 classes that map to each SatQuery class."""
        torch = self._torch
        rgb = arr[..., :3] if arr.shape[-1] >= 3 else np.repeat(arr[..., :1], 3, -1)

        # naive resize to a fixed size via simple striding (avoids a torchvision/PIL
        # resize dependency here; fine-tuning script trains on native patch size)
        h, w, _ = rgb.shape
        target = 120
        ys = np.linspace(0, h - 1, target).astype(int)
        xs = np.linspace(0, w - 1, target).astype(int)
        resized = rgb[ys][:, xs]

        tensor = torch.tensor(resized.transpose(2, 0, 1)[None], dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits)[0].numpy()

        scores: Dict[str, float] = {c: 0.0 for c in LAND_COVER_CLASSES}
        for cls_name, prob in zip(self.class_names, probs):
            target_cls = self.class_to_satquery.get(cls_name, "other")
            scores[target_cls] = max(scores.get(target_cls, 0.0), float(prob))
        return scores


def _blend_proportions(heuristic_proportions: Dict[str, float], scene_scores: Dict[str, float],
                        scene_weight: float = 0.5) -> Dict[str, float]:
    """Combine the heuristic per-pixel area proportions with the trained scene
    classifier's confidence scores (renormalized), so a real trained signal
    influences the final answer without discarding pixel-level spatial info
    the fine-tuned model (scene-level only) can't provide."""
    total_scene = sum(scene_scores.values()) or 1.0
    scene_norm = {k: v / total_scene for k, v in scene_scores.items()}
    blended = {
        cls: (1 - scene_weight) * heuristic_proportions.get(cls, 0.0) + scene_weight * scene_norm.get(cls, 0.0)
        for cls in LAND_COVER_CLASSES
    }
    total = sum(blended.values()) or 1.0
    return {k: v / total for k, v in blended.items()}
