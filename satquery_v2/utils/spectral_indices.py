"""
utils/spectral_indices.py
--------------------------
Land-cover classification from spectral bands — the default backend behind
"remote-sensing-adapted vision-language component" (models/base_vlm.py).

WHY THIS FILE WAS REWRITTEN (real bug fix, not cosmetic)
----------------------------------------------------------
The original version guessed water/vegetation/built-up from RGB color alone
(blueness, darkness, greenness). That's a proxy, and proxies break constantly
on real imagery: dark rooftops, asphalt, and shadow all get read as "water";
bright bare soil or cloud gets read as "built-up". This is almost certainly
what produced the false water-body detections on real Microsoft Planetary
Computer (Sentinel-2) scenes.

The fix: when real NIR/SWIR bands are present (which Sentinel-2 data from
Planetary Computer has), compute the *actual* remote-sensing indices these
classes are defined by:
  - NDVI  = (NIR - Red) / (NIR + Red)                  — vegetation
  - NDWI  = (Green - NIR) / (Green + NIR)   [McFeeters] — open water
  - MNDWI = (Green - SWIR1) / (Green + SWIR1)           — water, more robust
            to built-up confusion than NDWI
  - NDBI  = (SWIR1 - NIR) / (SWIR1 + NIR)                — built-up
These are far more reliable than color guessing, but only work if we know
*which band index is which*. Silently guessing that (the old bug) is worse
than not guessing at all — Planetary Computer stacks commonly ship as
Blue,Green,Red,NIR or Red,Green,Blue,NIR or the full 12/13-band L2A order,
and guessing wrong silently corrupts every downstream answer. So this module
never guesses band roles on its own for 4+ band imagery; callers pass an
explicit `BandRoles` (see BAND_PRESETS below for the common Planetary
Computer layouts, selectable in the UI) and everything downstream is
computed from real indices instead of color.

For plain 3-band RGB (PNG/JPEG, or true-color-only GeoTIFF with no NIR/SWIR),
there is no substitute for real spectral bands, so the RGB proxy remains —
but ImageEvidence.notes always says explicitly whether true indices or a
color proxy were used, so this is auditable per request rather than a silent
guess.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

LAND_COVER_CLASSES = ["water", "vegetation", "built_up", "bare_soil", "other"]


@dataclass
class BandRoles:
    """Maps semantic band roles to indices into the array's last axis.
    Any role can be None (unavailable) — classify_landcover() degrades
    gracefully, using the best index it can compute and falling back to
    RGB proxies only for whichever specific class that leaves ambiguous."""
    red: Optional[int] = None
    green: Optional[int] = None
    blue: Optional[int] = None
    nir: Optional[int] = None
    swir1: Optional[int] = None
    swir2: Optional[int] = None

    def as_dict(self):
        return asdict(self)


# Common band layouts as actually shipped by Microsoft Planetary Computer /
# standard Sentinel-2 processing pipelines. Pick the one that matches how you
# exported your COG/stack; when in doubt, check the STAC item's asset order.
BAND_PRESETS: Dict[str, BandRoles] = {
    "rgb_only": BandRoles(red=0, green=1, blue=2),
    "rgb_nir (R,G,B,NIR)": BandRoles(red=0, green=1, blue=2, nir=3),
    "bgr_nir (B,G,R,NIR)": BandRoles(blue=0, green=1, red=2, nir=3),
    # Standard Sentinel-2 L2A 12-band stack order (B01,B02,B03,B04,B05,B06,
    # B07,B08,B8A,B09,B11,B12) as commonly produced by odc-stac/stackstac
    # against Planetary Computer's `sentinel-2-l2a` collection:
    "sentinel2_l2a_12band": BandRoles(blue=1, green=2, red=3, nir=7, swir1=10, swir2=11),
    # Same but including B10 (13-band raw order some pipelines keep):
    "sentinel2_l2a_13band": BandRoles(blue=1, green=2, red=3, nir=7, swir1=11, swir2=12),
}


def _get_band(arr: np.ndarray, idx: Optional[int]) -> Optional[np.ndarray]:
    if idx is None:
        return None
    if idx < 0 or idx >= arr.shape[-1]:
        return None
    return arr[..., idx].astype(np.float32)


def compute_indices(arr: np.ndarray, roles: Optional[BandRoles]) -> Dict[str, Optional[np.ndarray]]:
    """Compute whichever real indices the available bands support. Missing
    inputs simply produce None for that index — callers must handle that."""
    roles = roles or BandRoles()
    red = _get_band(arr, roles.red)
    green = _get_band(arr, roles.green)
    blue = _get_band(arr, roles.blue)
    nir = _get_band(arr, roles.nir)
    swir1 = _get_band(arr, roles.swir1)

    out: Dict[str, Optional[np.ndarray]] = {"ndvi": None, "ndwi": None, "mndwi": None, "ndbi": None}

    if nir is not None and red is not None:
        out["ndvi"] = (nir - red) / (nir + red + 1e-6)
    if green is not None and nir is not None:
        out["ndwi"] = (green - nir) / (green + nir + 1e-6)          # McFeeters NDWI
    if green is not None and swir1 is not None:
        out["mndwi"] = (green - swir1) / (green + swir1 + 1e-6)      # modified NDWI (more robust)
    if swir1 is not None and nir is not None:
        out["ndbi"] = (swir1 - nir) / (swir1 + nir + 1e-6)           # built-up index
    return out


def _rgb_fallback(arr: np.ndarray, roles: Optional[BandRoles]) -> np.ndarray:
    """Best-effort RGB triplet for display/proxy purposes when true bands are
    known (via roles) or otherwise just the first 3 channels."""
    if roles and roles.red is not None and roles.green is not None and roles.blue is not None:
        r = _get_band(arr, roles.red)
        g = _get_band(arr, roles.green)
        b = _get_band(arr, roles.blue)
        return np.stack([r, g, b], axis=-1)
    if arr.shape[-1] >= 3:
        return arr[..., :3]
    return np.repeat(arr[..., :1], 3, axis=-1)


def greenness_index(rgb: np.ndarray) -> np.ndarray:
    """Excess-Green proxy, used ONLY when no NIR band is available."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    denom = (r + g + b) + 1e-6
    return (2 * g - r - b) / denom


def water_color_proxy(rgb: np.ndarray) -> np.ndarray:
    """Color-based water proxy, used ONLY when no NIR/SWIR band is available.
    Deliberately conservative: darkness alone is never enough because forests,
    shadows, and asphalt are often darker than water in an RGB image."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    brightness = (r + g + b) / 3.0
    blueness = (b - r) / (b + r + 1e-6)
    darkness = 1.0 - brightness
    # Very dark blue pixels are commonly forest shadow, haze, or a display
    # artifact. Do not call them water without a minimum visible brightness.
    blue_dominant = (b > r * 1.10) & (b > g * 1.02) & (brightness > 0.08)
    # Keep darkness as a weak supporting signal only after a blue/cool colour
    # check. This prevents dark green vegetation from becoming water.
    return np.where(blue_dominant, 0.75 * blueness + 0.25 * darkness, 0.0)


def classify_landcover(arr: np.ndarray, band_roles: Optional[BandRoles] = None,
                        ) -> Tuple[np.ndarray, Dict[str, float], list]:
    """
    Produce a per-pixel land-cover class map and class-proportion summary.

    Returns:
        class_map: H x W int array, indices into LAND_COVER_CLASSES
        proportions: dict class_name -> fraction of image area
        method_notes: list of strings recording which index (true spectral vs
            color proxy) was used for each class, for the auditable trace.
    """
    indices = compute_indices(arr, band_roles)
    rgb = _rgb_fallback(arr, band_roles)
    brightness = rgb.mean(axis=-1)
    h, w = brightness.shape

    notes = []
    class_map = np.full((h, w), LAND_COVER_CLASSES.index("other"), dtype=np.int32)

    # --- water ---
    if indices["mndwi"] is not None:
        water_mask = indices["mndwi"] > 0.05
        notes.append("water: true MNDWI (green,SWIR1) — most reliable, low false-positive on shadow/asphalt.")
    elif indices["ndwi"] is not None:
        water_mask = indices["ndwi"] > 0.05
        notes.append("water: true NDWI (green,NIR).")
    else:
        water_mask = water_color_proxy(rgb) > 0.16  # RGB-only water remains an estimate
        notes.append("water: RGB color proxy (no NIR/SWIR band supplied) — less reliable, "
                      "dark blue areas are treated as uncertain. Provide band_roles with "
                      "a NIR/SWIR band to confirm water accurately.")

    # --- vegetation ---
    if indices["ndvi"] is not None:
        veg_mask = (~water_mask) & (indices["ndvi"] > 0.2)
        notes.append("vegetation: true NDVI (NIR,red).")
    else:
        # A low threshold catches dark forest canopies, which are common in
        # island imagery and should not fall into the generic "other" bucket.
        veg_mask = (~water_mask) & (greenness_index(rgb) > 0.02)
        notes.append("vegetation: RGB greenness proxy (no NIR band supplied) — less reliable.")

    # --- built-up ---
    if indices["ndbi"] is not None:
        built_mask = (~water_mask) & (~veg_mask) & (indices["ndbi"] > -0.05)
        notes.append("built_up: true NDBI (SWIR1,NIR) — reliable separation from bare soil.")
    else:
        # Brightness alone missed real settlements. Concrete, metal roofing and
        # asphalt sit around 0.35-0.65 reflectance in true colour — below the
        # old 0.55 cut — so villages and low-rise towns were being labelled
        # bare soil. What actually separates built-up from bare soil in RGB is
        # spectral *flatness*: manufactured surfaces are near-neutral grey,
        # while soil and sand are distinctly warm (red channel dominant).
        # Requiring "reasonably bright AND low chroma" recovers settlements
        # without stealing pixels from genuine bare ground.
        max_channel = rgb.max(axis=-1)
        min_channel = rgb.min(axis=-1)
        chroma = (max_channel - min_channel) / (max_channel + 1e-6)
        neutral = chroma < 0.35
        built_mask = (~water_mask) & (~veg_mask) & (brightness > 0.35) & neutral
        # Very bright surfaces are built-up (or cloud) regardless of chroma.
        built_mask |= (~water_mask) & (~veg_mask) & (brightness > 0.62)
        notes.append("built_up: brightness + low-chroma proxy (no SWIR/NIR band supplied) — "
                      "separates neutral man-made surfaces from warm bare soil, but can still "
                      "confuse bright cloud or pale rock with built-up area.")

    soil_mask = (~water_mask) & (~veg_mask) & (~built_mask) & (brightness > 0.10)

    # For an RGB photograph, reserve "other" for genuinely neutral/uncertain
    # pixels. Most residual land should still receive a useful plain-language
    # label rather than making the answer say that most of an island is unknown.
    if indices["ndvi"] is None:
        residual = (~water_mask) & (~veg_mask) & (~built_mask) & (~soil_mask)
        neutral = np.abs(rgb[..., 1] - rgb[..., 0]) < 0.04
        soil_mask |= residual & neutral

    class_map[water_mask] = LAND_COVER_CLASSES.index("water")
    class_map[veg_mask] = LAND_COVER_CLASSES.index("vegetation")
    class_map[built_mask] = LAND_COVER_CLASSES.index("built_up")
    class_map[soil_mask] = LAND_COVER_CLASSES.index("bare_soil")

    total = h * w
    proportions = {
        cls: float(np.sum(class_map == i)) / total for i, cls in enumerate(LAND_COVER_CLASSES)
    }
    return class_map, proportions, notes


def dominant_classes(proportions: Dict[str, float], top_k: int = 2, min_frac: float = 0.05):
    ranked = sorted(proportions.items(), key=lambda kv: kv[1], reverse=True)
    ranked = [(c, f) for c, f in ranked if f >= min_frac]
    return ranked[:top_k]
