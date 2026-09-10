"""
utils/image_io.py
------------------
Loading and basic metadata extraction for remote-sensing imagery.

Supported inputs:
  - GeoTIFF / TIFF   (multi-band optical/multispectral, or single-band SAR)
  - PNG / JPEG       (only intended for the public benchmark datasets, per spec)

Design notes:
  We try, in order: rasterio (best, gives CRS/geotransform) -> tifffile -> PIL.
  Only rasterio/tifffile/PIL that are actually installed are used; missing
  optional deps degrade gracefully rather than crashing the app.
"""
from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


@dataclass
class LoadedImage:
    array: np.ndarray               # H x W x C, float32, normalized to [0, 1]
    path: str
    bands: int = 0
    height: int = 0
    width: int = 0
    dtype_original: str = ""
    has_geo_metadata: bool = False
    crs: Optional[str] = None
    modality_guess: str = "unknown"   # "optical" | "sar" | "unknown"
    pixel_size_m: Optional[float] = None   # ground sample distance, if known
    transform: Optional[tuple] = None      # affine geotransform, if known
    warnings: list = field(default_factory=list)


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize bands while preserving RGB colour relationships.

    Percentile-stretching each RGB channel independently can turn a dark green
    forest into a blue-looking image and consequently create false water
    detections. Standard 8-bit RGB uploads already have a meaningful colour
    scale, so they are divided by 255 as-is. Scientific/multiband arrays still
    use a per-band stretch for display and heuristic analysis.
    """
    arr = arr.astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 3 and np.nanmax(arr) > 1.0:
        if np.nanmax(arr) <= 255.0:
            return np.clip(arr / 255.0, 0, 1).astype(np.float32)
    out = np.zeros_like(arr)
    n_bands = arr.shape[-1] if arr.ndim == 3 else 1
    if arr.ndim == 2:
        arr = arr[:, :, None]
    for b in range(arr.shape[-1]):
        band = arr[..., b]
        lo, hi = np.percentile(band, 2), np.percentile(band, 98)
        if hi <= lo:
            lo, hi = float(band.min()), float(band.max() or 1.0)
        band = np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)
        out[..., b] = band
    return out


def guess_modality(arr: np.ndarray, filename: str = "") -> str:
    """
    Heuristic modality detection:
      - SAR: typically single-band (or 2-band, VV/VH), high dynamic range,
        speckled texture, no strong RGB color structure.
      - Optical/multispectral: 3+ bands with distinct color/spectral separation.
    Filename hints (e.g. containing 'sar', 's1', 'vv', 'vh') are used as a
    secondary signal.
    """
    fname = filename.lower()
    if any(tag in fname for tag in ("sar", "s1", "_vv", "_vh", "risat", "sentinel-1", "sentinel1")):
        return "sar"
    if any(tag in fname for tag in ("optical", "s2", "sentinel-2", "sentinel2", "cartosat", "msi")):
        return "optical"

    bands = arr.shape[-1] if arr.ndim == 3 else 1
    if bands <= 2:
        return "sar"

    # Speckle/texture heuristic: SAR has high local variance relative to mean (speckle)
    # while optical imagery (after stretch) tends to have smoother, more structured
    # color regions. Also check for color diversity (SAR is near-grayscale even
    # when duplicated across channels).
    if bands >= 3:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        color_spread = float(np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)) + np.mean(np.abs(r - b)))
        if color_spread < 0.02:
            return "sar"  # near-grayscale across "RGB" bands -> likely SAR duplicated to 3ch
        return "optical"

    return "unknown"


def load_image(path: str) -> LoadedImage:
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported: GeoTIFF/TIFF for geospatial "
            f"imagery; PNG/JPEG only for prescribed benchmark datasets."
        )

    warnings = []
    arr = None
    has_geo = False
    crs = None
    dtype_original = ""
    pixel_size_m = None
    transform = None

    if ext in (".tif", ".tiff"):
        # Try rasterio first (geo-aware), then tifffile (fast, no geo), then PIL.
        try:
            import rasterio
            with rasterio.open(path) as src:
                data = src.read()  # bands, H, W
                arr = np.transpose(data, (1, 2, 0))
                dtype_original = str(data.dtype)
                has_geo = src.crs is not None
                crs = str(src.crs) if src.crs else None
                # Ground sample distance drives every physical-area figure we
                # report. Only trust it for projected (metre-based) CRSs —
                # degrees from a geographic CRS would silently produce
                # nonsensical square-metre values.
                try:
                    res_x, res_y = src.res
                    if src.crs is not None and getattr(src.crs, "linear_units", "") not in ("", "unknown"):
                        pixel_size_m = float((abs(res_x) + abs(res_y)) / 2.0)
                    transform = tuple(src.transform)[:6]
                except Exception:
                    pixel_size_m = None
        except Exception:
            try:
                import tifffile
                data = tifffile.imread(path)
                if data.ndim == 2:
                    arr = data[:, :, None]
                elif data.ndim == 3 and data.shape[0] <= 32 and data.shape[0] < data.shape[-1]:
                    # channels-first heuristic
                    arr = np.transpose(data, (1, 2, 0))
                else:
                    arr = data
                dtype_original = str(data.dtype)
                warnings.append("Loaded via tifffile: no CRS/geotransform available.")
            except Exception as e:
                from PIL import Image
                img = Image.open(path)
                arr = np.array(img)
                if arr.ndim == 2:
                    arr = arr[:, :, None]
                dtype_original = str(arr.dtype)
                warnings.append(f"Fell back to PIL for TIFF read ({e}); geo metadata unavailable.")
    else:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        dtype_original = str(arr.dtype)
        warnings.append("PNG/JPEG input: only valid for prescribed public benchmark datasets.")

    if arr is None:
        raise IOError(f"Could not read image at {path}")

    if arr.ndim == 2:
        arr = arr[:, :, None]

    h, w, bands = arr.shape
    norm = _normalize(arr)
    modality = guess_modality(norm, filename=os.path.basename(path))

    return LoadedImage(
        array=norm,
        path=path,
        bands=bands,
        height=h,
        width=w,
        dtype_original=dtype_original,
        has_geo_metadata=has_geo,
        crs=crs,
        modality_guess=modality,
        pixel_size_m=pixel_size_m,
        transform=transform,
        warnings=warnings,
    )


def to_display_rgb(img: LoadedImage) -> np.ndarray:
    """Return an 8-bit HxWx3 array suitable for display/overlay, from any band count."""
    arr = img.array
    if arr.shape[-1] >= 3:
        rgb = arr[..., :3]
    else:
        rgb = np.repeat(arr[..., :1], 3, axis=-1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def resample_to(array: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resample an array onto a target (height, width) grid.

    Used to align a bi-temporal or cross-modal pair whose rasters differ
    slightly in size. Nearest-neighbour is deliberate: it introduces no
    invented intermediate pixel values, so downstream change statistics stay
    traceable to real observations.
    """
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[:, :, None]
    height, width = shape
    if array.shape[0] == height and array.shape[1] == width:
        return array
    ys = np.linspace(0, array.shape[0] - 1, height).astype(int)
    xs = np.linspace(0, array.shape[1] - 1, width).astype(int)
    return array[ys][:, xs]


def align_pair(a: LoadedImage, b: LoadedImage) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """Put two images on a common pixel grid.

    Returns (array_a, array_b, note). The smaller footprint wins so we never
    upsample one image into detail it does not have. `note` is None when the
    pair already matched and is a trace message otherwise.
    """
    if (a.height, a.width) == (b.height, b.width):
        return a.array, b.array, None
    height = min(a.height, b.height)
    width = min(a.width, b.width)
    note = (
        f"Resampled the pair onto a common {width}x{height} grid "
        f"(was {a.width}x{a.height} and {b.width}x{b.height})."
    )
    return resample_to(a.array, (height, width)), resample_to(b.array, (height, width)), note


def same_footprint(a: LoadedImage, b: LoadedImage, tol: float = 0.02) -> Tuple[bool, str]:
    """Cheap co-registration compatibility check based on size (real deployments
    should also check CRS + geotransform bounds when available)."""
    if a.height == 0 or b.height == 0:
        return False, "One or both images failed to load."
    dh = abs(a.height - b.height) / max(a.height, b.height)
    dw = abs(a.width - b.width) / max(a.width, b.width)
    if dh > tol or dw > tol:
        return False, (
            f"Size mismatch beyond tolerance: {a.height}x{a.width} vs "
            f"{b.height}x{b.width}. Images should be co-registered/resampled first."
        )
    if a.has_geo_metadata and b.has_geo_metadata and a.crs != b.crs:
        return False, f"CRS mismatch: {a.crs} vs {b.crs}. Reproject before analysis."
    return True, "OK"
