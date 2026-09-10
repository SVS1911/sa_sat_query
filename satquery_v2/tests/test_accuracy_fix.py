"""
tests/test_accuracy_fix.py
------------------------------
Demonstrates, with a synthetic-but-realistic scene, exactly the failure mode
reported on real Planetary Computer imagery: dark shadow getting misread as
water by the old RGB-only proxy — and shows the NIR-based fix resolving it,
scored with the new evaluation harness (utils/evaluation.py) rather than by
eyeballing it.

Synthetic scene layout (128x128, 4-band R,G,B,NIR):
  - true water:   low NIR reflectance (water absorbs NIR strongly)
  - true shadow:  uniformly dark across ALL bands including NIR (this is what
                  fools a color/brightness-only proxy — it looks "dark and
                  slightly blue" just like water does in plain RGB)
  - true vegetation: high NIR, high NDVI
  - true built-up: bright, flat spectral response
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.spectral_indices import classify_landcover, LAND_COVER_CLASSES, BAND_PRESETS
from utils.evaluation import evaluate_landcover


def build_synthetic_scene(h=128, w=128, seed=0):
    rng = np.random.RandomState(seed)
    arr = np.zeros((h, w, 4), dtype=np.float32)  # R, G, B, NIR
    gt = np.full((h, w), LAND_COVER_CLASSES.index("other"), dtype=np.int32)

    def region(y0, y1, x0, x1, rgba, gt_class, noise=0.02):
        arr[y0:y1, x0:x1] = np.array(rgba) + rng.uniform(-noise, noise, (y1 - y0, x1 - x0, 4))
        gt[y0:y1, x0:x1] = LAND_COVER_CLASSES.index(gt_class)

    # true open water: low across R/G/B, VERY low NIR (water absorbs NIR strongly)
    region(0, 40, 0, 60, (0.15, 0.20, 0.30, 0.03), "water")

    # true shadow (e.g. building/cloud shadow over bare ground): uniformly dark
    # across ALL bands INCLUDING NIR — this is what an RGB-only proxy confuses
    # with water, since it's dark and slightly cool-toned just like water is.
    region(40, 80, 0, 60, (0.12, 0.14, 0.16, 0.13), "bare_soil")

    # true vegetation: moderate visible, high NIR -> high NDVI
    region(0, 64, 60, 128, (0.10, 0.30, 0.08, 0.55), "vegetation")

    # true built-up: bright, flat spectral response across all bands
    region(64, 128, 60, 128, (0.75, 0.73, 0.70, 0.72), "built_up")

    return np.clip(arr, 0, 1), gt


def build_rgb_false_water_scene(h=64, w=64):
    """Regression scene for dark green vegetation being mistaken for water."""
    arr = np.zeros((h, w, 3), dtype=np.float32)
    arr[:h // 2] = (0.06, 0.20, 0.07)  # dark green forest
    arr[h // 2:] = (0.42, 0.18, 0.32)  # pink/brown road and buildings
    return arr


def main():
    rgb_scene = build_rgb_false_water_scene()
    rgb_prediction, rgb_proportions, _ = classify_landcover(rgb_scene)
    assert rgb_proportions["water"] < 0.05, (
        "Dark green vegetation should not be classified as mostly water."
    )

    arr, gt = build_synthetic_scene()

    print("=" * 70)
    print("OLD BEHAVIOR: RGB-only proxy (no band_roles -> no NIR/SWIR available)")
    print("=" * 70)
    pred_proxy, _, notes_proxy = classify_landcover(arr[..., :3], band_roles=None)
    for n in notes_proxy:
        print(" ", n)
    report_proxy = evaluate_landcover(pred_proxy, gt, LAND_COVER_CLASSES)
    print()
    print(report_proxy.pretty_print())

    shadow_gt_mask = (gt == LAND_COVER_CLASSES.index("bare_soil"))
    shadow_pred_as_water = (pred_proxy == LAND_COVER_CLASSES.index("water")) & shadow_gt_mask
    print(f"\n  -> Shadow pixels misclassified as water: "
          f"{shadow_pred_as_water.sum()} / {shadow_gt_mask.sum()} "
          f"({100*shadow_pred_as_water.sum()/max(1,shadow_gt_mask.sum()):.1f}%)")

    print()
    print("=" * 70)
    print("FIXED BEHAVIOR: true NDVI/NDWI from a real NIR band (band_roles supplied)")
    print("=" * 70)
    roles = BAND_PRESETS["rgb_nir (R,G,B,NIR)"]
    pred_fixed, _, notes_fixed = classify_landcover(arr, band_roles=roles)
    for n in notes_fixed:
        print(" ", n)
    report_fixed = evaluate_landcover(pred_fixed, gt, LAND_COVER_CLASSES)
    print()
    print(report_fixed.pretty_print())

    shadow_pred_as_water_fixed = (pred_fixed == LAND_COVER_CLASSES.index("water")) & shadow_gt_mask
    print(f"\n  -> Shadow pixels misclassified as water: "
          f"{shadow_pred_as_water_fixed.sum()} / {shadow_gt_mask.sum()} "
          f"({100*shadow_pred_as_water_fixed.sum()/max(1,shadow_gt_mask.sum()):.1f}%)")

    print()
    print("=" * 70)
    print(f"Overall accuracy: proxy={report_proxy.overall_accuracy*100:.1f}%  "
          f"-> fixed={report_fixed.overall_accuracy*100:.1f}%")
    print(f"Mean IoU:         proxy={report_proxy.mean_iou*100:.1f}%  "
          f"-> fixed={report_fixed.mean_iou*100:.1f}%")
    print("=" * 70)

    assert report_fixed.overall_accuracy > report_proxy.overall_accuracy, \
        "Expected the NIR-based fix to outperform the RGB-only proxy on this scene."
    assert shadow_pred_as_water_fixed.sum() < shadow_pred_as_water.sum(), \
        "Expected fewer shadow->water false positives with the NIR-based fix."
    print("\nPASS: NIR-based classification reduces false water detections and "
          "improves overall accuracy vs the RGB-only proxy, as measured by "
          "utils.evaluation.evaluate_landcover.")


if __name__ == "__main__":
    main()
