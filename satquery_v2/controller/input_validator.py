"""
controller/input_validator.py
-------------------------------
Checks number, modality, format, metadata, and compatibility of input images
before the agentic controller selects a workflow. This is deliberately a
separate step so its checks show up verbatim in the auditable execution trace.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from utils.image_io import LoadedImage, same_footprint


@dataclass
class ValidationResult:
    ok: bool
    scenario: str            # "single" | "cross_modal" | "bi_temporal" | "invalid"
    messages: List[str] = field(default_factory=list)


def _aspect_ratio_gap(a: LoadedImage, b: LoadedImage) -> float:
    """Relative difference between two images' aspect ratios."""
    if not a.height or not b.height or not a.width or not b.width:
        return 1.0
    ratio_a = a.width / a.height
    ratio_b = b.width / b.height
    return abs(ratio_a - ratio_b) / max(ratio_a, ratio_b)


def validate(images: List[LoadedImage], declared_pair_type: Optional[str] = None) -> ValidationResult:
    """
    declared_pair_type: optional user hint - "cross_modal" or "bi_temporal" - used
    to disambiguate two-image inputs when modality auto-detection is uncertain
    (e.g. two optical images from different sensors both being mislabeled).
    """
    msgs = []

    if len(images) == 0:
        return ValidationResult(False, "invalid", ["No images supplied."])

    if len(images) == 1:
        img = images[0]
        msgs.append(f"Single image accepted: {img.width}x{img.height}, {img.bands} band(s), "
                     f"modality={img.modality_guess}, geo_metadata={img.has_geo_metadata}.")
        if img.bands == 0:
            return ValidationResult(False, "invalid", msgs + ["Image failed to decode."])
        return ValidationResult(True, "single", msgs)

    if len(images) == 2:
        a, b = images
        ok_fp, fp_msg = same_footprint(a, b)
        if ok_fp:
            msgs.append(f"Footprint/size compatibility check: {fp_msg}")
        else:
            # A size difference is recoverable — the controller resamples the
            # pair onto a common grid (utils.image_io.align_pair) before any
            # measurement is taken. A CRS conflict is not recoverable, because
            # resampling across projections would silently misregister pixels.
            if a.has_geo_metadata and b.has_geo_metadata and a.crs != b.crs:
                msgs.append(
                    f"CRS mismatch: {a.crs} vs {b.crs}. Reproject both images to a "
                    f"common CRS before analysis."
                )
                return ValidationResult(False, "invalid", msgs)
            if _aspect_ratio_gap(a, b) > 0.15:
                msgs.append(
                    f"Aspect-ratio mismatch too large to align safely: "
                    f"{a.width}x{a.height} vs {b.width}x{b.height}. These images "
                    f"probably do not cover the same area."
                )
                return ValidationResult(False, "invalid", msgs)
            msgs.append(
                f"Size difference detected ({a.width}x{a.height} vs {b.width}x{b.height}); "
                f"the pair will be resampled onto a common grid before analysis."
            )

        modalities = {a.modality_guess, b.modality_guess}
        if declared_pair_type in ("cross_modal", "bi_temporal"):
            scenario = declared_pair_type
            msgs.append(f"Pair type set by user declaration: {declared_pair_type}.")
        elif "sar" in modalities and "optical" in modalities:
            scenario = "cross_modal"
            msgs.append("Auto-detected one SAR + one optical image -> cross-modal pair.")
        else:
            scenario = "bi_temporal"
            msgs.append("Auto-detected two same-modality images -> treating as bi-temporal pair. "
                        "If these are actually two different sensors, set pair type explicitly.")

        msgs.append(f"Image A: {a.width}x{a.height}, {a.bands}b, modality={a.modality_guess}")
        msgs.append(f"Image B: {b.width}x{b.height}, {b.bands}b, modality={b.modality_guess}")
        return ValidationResult(True, scenario, msgs)

    return ValidationResult(False, "invalid", [f"Unsupported number of images: {len(images)} "
                                                f"(expected 1 for single-image tasks or 2 for "
                                                f"cross-modal/bi-temporal pairs)."])
