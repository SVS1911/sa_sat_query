"""
utils/numerical.py
-------------------
Every number SatQuery reports comes from here.

Design rule for this whole project: measurements are computed from pixels with
NumPy, never produced by a language model. A model may *describe* a change; it
may not *quantify* one. That separation is what makes the answers auditable —
if the UI says "12.7% of the image changed", that figure is
`change_mask.sum() / change_mask.size`, and nothing else.

Physical (ground) area is reported only when the source raster actually carried
a resolution/CRS. Without it we say so explicitly rather than inventing a
plausible number of hectares.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Region counting is capped so a pathological mask can't stall the UI.
_MAX_LABEL_PIXELS = 512 * 512
_MAX_PROPAGATION_STEPS = 600


@dataclass
class ImageStats:
    """Basic, verifiable statistics for a single image."""

    width: int = 0
    height: int = 0
    bands: int = 0
    total_pixels: int = 0
    dtype: str = ""
    mean: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    std: float = 0.0
    per_band_mean: List[float] = field(default_factory=list)
    has_geo_metadata: bool = False
    crs: Optional[str] = None
    pixel_size_m: Optional[float] = None

    def as_lines(self) -> List[str]:
        lines = [
            f"Dimensions: {self.width} x {self.height} px ({self.bands} band(s))",
            f"Total pixels: {self.total_pixels:,}",
            f"Reflectance range: min {self.minimum:.3f} / mean {self.mean:.3f} / max {self.maximum:.3f}",
            f"Standard deviation: {self.std:.3f}",
        ]
        if self.has_geo_metadata and self.pixel_size_m:
            lines.append(f"Ground sample distance: {self.pixel_size_m:.2f} m/px (CRS {self.crs})")
        elif self.has_geo_metadata:
            lines.append(f"Georeferenced (CRS {self.crs}), pixel size unavailable")
        else:
            lines.append("No CRS/resolution metadata — measurements stay in pixel units")
        return lines


def image_statistics(image: Any) -> ImageStats:
    """Compute statistics from a LoadedImage (or a raw array)."""
    array = getattr(image, "array", image)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array[:, :, None]

    height, width, bands = array.shape
    finite = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)

    pixel_size = getattr(image, "pixel_size_m", None)
    return ImageStats(
        width=width,
        height=height,
        bands=bands,
        total_pixels=int(height * width),
        dtype=str(getattr(image, "dtype_original", array.dtype)),
        mean=float(finite.mean()),
        minimum=float(finite.min()),
        maximum=float(finite.max()),
        std=float(finite.std()),
        per_band_mean=[float(finite[..., b].mean()) for b in range(bands)],
        has_geo_metadata=bool(getattr(image, "has_geo_metadata", False)),
        crs=getattr(image, "crs", None),
        pixel_size_m=pixel_size,
    )


# --------------------------------------------------------------------------- #
# Connected components
# --------------------------------------------------------------------------- #
def label_regions(mask: np.ndarray, min_area: int = 12) -> Tuple[np.ndarray, int, List[int], bool]:
    """Label 4-connected regions in a boolean mask.

    Returns (labels, count, sorted_areas, exact). `exact` is False when the
    label propagation hit its iteration cap, in which case the count is a lower
    bound and the caller must present it as approximate.

    Uses scipy when it is installed (fast, exact) and otherwise falls back to a
    vectorised minimum-label propagation, which keeps the base install free of
    a scipy dependency without dropping the feature.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not mask.any():
        return np.zeros(mask.shape, dtype=np.int32), 0, [], True

    work, scale = _downsample_for_labeling(mask)

    try:  # exact and fast when available
        from scipy import ndimage

        labels, count = ndimage.label(work)
        exact = True
    except ImportError:
        labels, count, exact = _propagate_labels(work)

    if count == 0:
        return np.zeros(mask.shape, dtype=np.int32), 0, [], exact

    flat = labels.ravel()
    counts = np.bincount(flat)
    counts[0] = 0  # background
    # Scale areas back up if we labelled a downsampled copy.
    area_scale = max(1.0, scale * scale)
    areas = [int(round(c * area_scale)) for c in counts if c > 0]
    areas = sorted((a for a in areas if a >= min_area), reverse=True)
    return labels, len(areas), areas, exact


def _downsample_for_labeling(mask: np.ndarray) -> Tuple[np.ndarray, float]:
    """Shrink very large masks so region labelling stays interactive."""
    height, width = mask.shape
    if height * width <= _MAX_LABEL_PIXELS:
        return mask, 1.0
    scale = float(np.sqrt((height * width) / _MAX_LABEL_PIXELS))
    new_h = max(1, int(height / scale))
    new_w = max(1, int(width / scale))
    ys = np.linspace(0, height - 1, new_h).astype(int)
    xs = np.linspace(0, width - 1, new_w).astype(int)
    return mask[ys][:, xs], scale


def _propagate_labels(mask: np.ndarray) -> Tuple[np.ndarray, int, bool]:
    """Vectorised connected-component labelling without scipy.

    Seeds every foreground pixel with a unique id, then repeatedly replaces
    each pixel's id with the minimum id in its 4-neighbourhood. Ids converge to
    one value per connected component. Each sweep is a handful of NumPy
    operations, so this is far faster than a Python flood fill.
    """
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int64)
    labels[mask] = np.arange(1, int(mask.sum()) + 1)

    big = np.iinfo(np.int64).max
    exact = False
    for _ in range(_MAX_PROPAGATION_STEPS):
        padded = np.full((height + 2, width + 2), big, dtype=np.int64)
        inner = np.where(mask, labels, big)
        padded[1:-1, 1:-1] = inner

        neighbour_min = np.minimum.reduce([
            padded[1:-1, 1:-1],
            padded[:-2, 1:-1],
            padded[2:, 1:-1],
            padded[1:-1, :-2],
            padded[1:-1, 2:],
        ])
        updated = np.where(mask, neighbour_min, 0)
        if np.array_equal(updated, labels):
            exact = True
            break
        labels = updated

    unique = np.unique(labels[mask])
    remap = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    remap[unique] = np.arange(1, len(unique) + 1, dtype=np.int32)
    return remap[labels], len(unique), exact


# --------------------------------------------------------------------------- #
# Change measurements
# --------------------------------------------------------------------------- #
@dataclass
class ChangeStats:
    """Measured properties of a bi-temporal change mask."""

    changed_pixels: int = 0
    total_pixels: int = 0
    changed_percent: float = 0.0
    region_count: int = 0
    region_count_exact: bool = True
    largest_region_pixels: int = 0
    region_areas: List[int] = field(default_factory=list)
    area_m2: Optional[float] = None
    area_note: str = ""
    centroid_description: str = ""

    def as_lines(self) -> List[str]:
        lines = [
            f"Changed pixels: {self.changed_pixels:,} of {self.total_pixels:,}",
            f"Changed area: {self.changed_percent:.2f}% of the image",
        ]
        if self.region_count:
            qualifier = "" if self.region_count_exact else " (approximate)"
            lines.append(f"Distinct changed regions: {self.region_count}{qualifier}")
            lines.append(f"Largest region: {self.largest_region_pixels:,} px")
        if self.area_m2 is not None:
            if self.area_m2 >= 1_000_000:
                lines.append(f"Ground area changed: {self.area_m2 / 1_000_000:.3f} km²")
            else:
                lines.append(
                    f"Ground area changed: {self.area_m2:,.0f} m² "
                    f"({self.area_m2 / 10_000:.2f} ha)"
                )
        else:
            lines.append(self.area_note)
        if self.centroid_description:
            lines.append(f"Change concentrated in {self.centroid_description}")
        return lines


def describe_location(mask: np.ndarray) -> str:
    """Plain-language position of a mask's centre of mass."""
    ys, xs = np.where(mask)
    if not len(xs):
        return ""
    height, width = mask.shape
    x = float(xs.mean()) / max(1, width)
    y = float(ys.mean()) / max(1, height)
    horizontal = "left" if x < 0.34 else "right" if x > 0.66 else "central"
    vertical = "upper" if y < 0.34 else "lower" if y > 0.66 else "middle"
    if horizontal == "central" and vertical == "middle":
        return "the centre of the image"
    return f"the {vertical} {horizontal} part of the image"


def change_statistics(
    mask: np.ndarray,
    pixel_size_m: Optional[float] = None,
    has_geo_metadata: bool = False,
    min_region_area: int = 12,
) -> ChangeStats:
    """Measure a boolean change mask. Pure arithmetic — no interpretation."""
    mask = np.asarray(mask, dtype=bool)
    total = int(mask.size)
    changed = int(mask.sum())
    percent = (changed / total * 100.0) if total else 0.0

    _, region_count, areas, exact = label_regions(mask, min_area=min_region_area)

    if pixel_size_m and pixel_size_m > 0:
        area_m2 = changed * float(pixel_size_m) ** 2
        area_note = ""
    else:
        area_m2 = None
        area_note = (
            "Physical area could not be calculated because spatial "
            "resolution/CRS metadata was unavailable."
        )

    return ChangeStats(
        changed_pixels=changed,
        total_pixels=total,
        changed_percent=percent,
        region_count=region_count,
        region_count_exact=exact,
        largest_region_pixels=areas[0] if areas else 0,
        region_areas=areas[:10],
        area_m2=area_m2,
        area_note=area_note,
        centroid_description=describe_location(mask),
    )


def class_area_table(
    proportions: Dict[str, float],
    total_pixels: int,
    pixel_size_m: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Turn class proportions into a pixel/area breakdown for the evidence panel."""
    rows: List[Dict[str, Any]] = []
    for name, fraction in sorted(proportions.items(), key=lambda kv: kv[1], reverse=True):
        pixels = int(round(fraction * total_pixels))
        row: Dict[str, Any] = {
            "class": name,
            "percent": fraction * 100.0,
            "pixels": pixels,
        }
        if pixel_size_m and pixel_size_m > 0:
            row["area_m2"] = pixels * float(pixel_size_m) ** 2
        rows.append(row)
    return rows


def difference_map(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Normalised absolute difference between two images, resampled if needed."""
    a = _to_rgb(before)
    b = _to_rgb(after)
    if a.shape[:2] != b.shape[:2]:
        ys = np.linspace(0, a.shape[0] - 1, b.shape[0]).astype(int)
        xs = np.linspace(0, a.shape[1] - 1, b.shape[1]).astype(int)
        a = a[ys][:, xs]
    diff = np.abs(a - b).mean(axis=-1)
    span = float(diff.max() - diff.min())
    if span <= 1e-6:
        return np.zeros_like(diff)
    return (diff - diff.min()) / span


def _to_rgb(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.shape[-1] >= 3:
        return array[..., :3]
    return np.repeat(array[..., :1], 3, axis=-1)
