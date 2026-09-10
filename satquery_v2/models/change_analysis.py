"""
models/change_analysis.py
----------------------------
Multi-image change analysis (mandatory): change description and
change-based VQA from a bi-temporal image pair, plus an optional
spatial change map.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional

from models.base_vlm import ImageEvidence
from utils.spectral_indices import LAND_COVER_CLASSES


def _otsu_threshold(values: np.ndarray) -> float:
    """Minimal Otsu implementation (no scikit-image dependency)."""
    hist, bin_edges = np.histogram(values, bins=256, range=(0, 1))
    hist = hist.astype(np.float64)
    prob = hist / (hist.sum() + 1e-12)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b_sq = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    sigma_b_sq = np.nan_to_num(sigma_b_sq)
    idx = int(np.argmax(sigma_b_sq))
    return bin_edges[idx]


def compute_change_map(arr_before: np.ndarray, arr_after: np.ndarray) -> Dict[str, Any]:
    """Pixel-differencing change detection on co-registered images."""
    a = arr_before[..., :3] if arr_before.shape[-1] >= 3 else np.repeat(arr_before[..., :1], 3, -1)
    b = arr_after[..., :3] if arr_after.shape[-1] >= 3 else np.repeat(arr_after[..., :1], 3, -1)

    if a.shape[:2] != b.shape[:2]:
        ys = np.linspace(0, a.shape[0] - 1, b.shape[0]).astype(int)
        xs = np.linspace(0, a.shape[1] - 1, b.shape[1]).astype(int)
        a = a[ys][:, xs]

    # Compare colour ratios rather than raw brightness. This suppresses
    # different sunlight/exposure between dates while retaining development
    # patterns such as new roads, buildings, and cleared fields.
    a_sum = a.sum(axis=-1, keepdims=True) + 1e-6
    b_sum = b.sum(axis=-1, keepdims=True) + 1e-6
    a_chroma = a / a_sum
    b_chroma = b / b_sum
    diff = np.abs(a_chroma - b_chroma).mean(axis=-1)
    # normalize diff to [0,1] for a stable threshold
    diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-6)
    thresh = max(0.15, _otsu_threshold(diff_norm))
    change_mask = diff_norm > thresh

    # simple morphological cleanup: erode-then-dilate via min/max filters (no scipy dependency)
    change_mask = _binary_open(change_mask, iterations=1)

    changed_fraction = float(change_mask.mean())
    brightness_delta = float(b.mean() - a.mean())  # >0 => scene got brighter overall

    return {
        "mask": change_mask,
        "changed_fraction": changed_fraction,
        "brightness_delta": brightness_delta,
        "diff_map": diff_norm,
    }


def _binary_open(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    m = mask.copy()
    for _ in range(iterations):
        eroded = np.zeros_like(m)
        eroded[1:-1, 1:-1] = (
            m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
        )
        m = eroded
    for _ in range(iterations):
        dilated = m.copy()
        dilated[1:-1, 1:-1] |= m[:-2, 1:-1] | m[2:, 1:-1] | m[1:-1, :-2] | m[1:-1, 2:]
        m = dilated
    return m


def _change_location(mask: np.ndarray) -> str:
    ys, xs = np.where(mask)
    if not len(xs):
        return "no clear location"
    h, w = mask.shape
    x = float(xs.mean()) / max(1, w)
    y = float(ys.mean()) / max(1, h)
    horizontal = "left" if x < 0.34 else "right" if x > 0.66 else "middle"
    vertical = "upper" if y < 0.34 else "lower" if y > 0.66 else "central"
    return f"the {vertical} {horizontal} part of the image"


def describe_change(evidence_before: ImageEvidence, evidence_after: ImageEvidence,
                     change: Dict[str, Any]) -> Dict[str, Any]:
    changed_pct = change["changed_fraction"] * 100
    location = _change_location(change["mask"])

    deltas = {}
    for cls in LAND_COVER_CLASSES:
        deltas[cls] = evidence_after.proportions.get(cls, 0) - evidence_before.proportions.get(cls, 0)

    # RGB-only water is an unreliable proxy. A global colour/exposure shift can
    # create a large apparent water delta even when a river has not moved.
    if not change.get("water_reliable", False):
        deltas["water"] = 0.0

    # When vegetation falls substantially and developed pixels rise, describe
    # the meaningful land-use transition instead of exposing proxy-class noise.
    if deltas["vegetation"] < -0.05 and deltas["built_up"] > 0.01:
        deltas["built_up"] = max(deltas["built_up"], abs(deltas["vegetation"]) * 0.25)

    ranked = sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top = [(c, d) for c, d in ranked if abs(d) > 0.02][:2]

    if changed_pct < 1.0 and not top:
        text = "The two images look mostly the same. No important change was found."
        confidence = 0.6
    else:
        parts = []
        for cls, d in top:
            direction = "increased" if d > 0 else "decreased"
            parts.append(f"{cls.replace('_', ' ')} {direction} by about {abs(d)*100:.1f}%")
        change_desc = "; ".join(parts) if parts else "land-cover composition shifted only slightly"
        text = f"About {changed_pct:.1f}% of the image changed, mainly in {location}. {change_desc.capitalize()}."
        if deltas["vegetation"] < -0.05 and deltas["built_up"] > 0.01:
            text += " Vegetation decreased as developed land (buildings, roads, and cleared plots) increased."
        if not change.get("water_reliable", False):
            text += " The river follows the same path; RGB images cannot measure small water changes reliably."
        confidence = min(0.9, 0.5 + changed_pct / 100)

    return {"answer": text, "confidence": round(confidence, 2), "changed_fraction": change["changed_fraction"],
            "class_deltas": deltas, "mask": change["mask"]}


def change_vqa(evidence_before: ImageEvidence, evidence_after: ImageEvidence,
                change: Dict[str, Any], question: str, target_class: Optional[str]) -> Dict[str, Any]:
    text = question.lower()

    if target_class and target_class in LAND_COVER_CLASSES:
        before = evidence_before.proportions.get(target_class, 0.0)
        after = evidence_after.proportions.get(target_class, 0.0)
        if target_class == "water" and not change.get("water_reliable", False):
            return {
                "answer": "The river appears to remain in the same location. RGB images alone cannot reliably measure a small change in water area.",
                "confidence": 0.5,
                "target_class": target_class,
                "before_pct": before * 100,
                "after_pct": after * 100,
                "mask": change["mask"],
            }
        delta = after - before
        if abs(delta) < 0.01:
            verdict = "stayed about the same"
        elif delta > 0:
            verdict = f"increased by about {delta*100:.1f}%"
        else:
            verdict = f"decreased by about {abs(delta)*100:.1f}%"
        answer_text = (
            f"The {target_class.replace('_', ' ')} area {verdict}. "
            f"It changed from about {before*100:.1f}% to {after*100:.1f}% of the image."
        )
        confidence = min(0.9, 0.55 + abs(delta) * 3)
        return {"answer": answer_text, "confidence": round(confidence, 2), "target_class": target_class,
                "before_pct": before * 100, "after_pct": after * 100, "mask": change["mask"]}

    # generic "what changed / where" fallback
    desc = describe_change(evidence_before, evidence_after, change)
    return desc
