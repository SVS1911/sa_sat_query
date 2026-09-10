"""
models/optical_sar_fusion.py
-------------------------------
Cross-modal pair analysis: extract complementary information from a
co-registered optical/multispectral + SAR image pair.

Approach: optical gives spectral land-cover classification (via base_vlm);
SAR gives structural/edge information (via a Canny-style gradient detector,
no OpenCV dependency) that is especially informative for built-up detection
(strong double-bounce backscatter -> high local gradient/brightness) and
water detection (specular reflection -> very low backscatter/dark, smooth).
Fusing the two typically resolves cases where optical alone is ambiguous
(e.g. shadow vs. water, or cloud-obscured regions).
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any

from models.base_vlm import ImageEvidence
from utils.spectral_indices import LAND_COVER_CLASSES


def _sobel_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return mag / (mag.max() + 1e-6)


def fuse(optical_evidence: ImageEvidence, sar_arr: np.ndarray, target_class: str = None) -> Dict[str, Any]:
    sar_gray = sar_arr[..., :1].mean(axis=-1) if sar_arr.shape[-1] >= 1 else sar_arr.mean(axis=-1)
    sar_edges = _sobel_gradient_magnitude(sar_gray)

    class_map = optical_evidence.class_map.copy()
    opt_h, opt_w = class_map.shape
    if sar_gray.shape != (opt_h, opt_w):
        # nearest-neighbor resample SAR to match optical grid (images should already
        # be co-registered per the input_validator footprint check, but guard anyway)
        ys = (np.linspace(0, sar_gray.shape[0] - 1, opt_h)).astype(int)
        xs = (np.linspace(0, sar_gray.shape[1] - 1, opt_w)).astype(int)
        sar_gray = sar_gray[ys][:, xs]
        sar_edges = sar_edges[ys][:, xs]

    water_idx = LAND_COVER_CLASSES.index("water")
    built_idx = LAND_COVER_CLASSES.index("built_up")

    # Refine built-up: optical says built_up AND SAR shows strong structural edges
    # (double-bounce/urban texture) -> high-confidence built-up
    optical_built = class_map == built_idx
    sar_confirms_built = sar_edges > np.percentile(sar_edges, 70)
    refined_built = optical_built & sar_confirms_built

    # Refine water: optical says water AND SAR is dark/smooth (very low backscatter,
    # low local gradient) -> high-confidence water (helps rule out optical shadow
    # false-positives, since shadows don't behave like water in SAR)
    optical_water = class_map == water_idx
    sar_dark_smooth = (sar_gray < np.percentile(sar_gray, 25)) & (sar_edges < np.percentile(sar_edges, 40))
    refined_water = optical_water & sar_dark_smooth

    refined_built_frac = float(refined_built.mean())
    refined_water_frac = float(refined_water.mean())
    optical_built_frac = optical_evidence.proportions.get("built_up", 0.0)
    optical_water_frac = optical_evidence.proportions.get("water", 0.0)

    false_positive_water = max(0.0, optical_water_frac - refined_water_frac)

    summary = (
        f"Using the colour image together with the radar image, I estimate that buildings and roads "
        f"cover {refined_built_frac*100:.1f}% of the image and water covers {refined_water_frac*100:.1f}%. "
        f"The radar image helps separate real water from dark shadows."
    )

    if target_class == "built_up":
        answer_text = (f"Using both images, buildings and roads cover about "
                        f"{refined_built_frac*100:.1f}% of the image.")
    elif target_class == "water":
        answer_text = (f"Using both images, water covers about {refined_water_frac*100:.1f}% of the image.")
    else:
        answer_text = summary

    confidence = 0.7  # cross-modal agreement generally raises confidence vs single modality
    return {
        "answer": answer_text,
        "confidence": confidence,
        "refined_built_up_mask": refined_built,
        "refined_water_mask": refined_water,
        "refined_built_up_fraction": refined_built_frac,
        "refined_water_fraction": refined_water_frac,
        "optical_only_built_up_fraction": optical_built_frac,
        "optical_only_water_fraction": optical_water_frac,
        "sar_edge_map": sar_edges,
    }
