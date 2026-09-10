"""Generate synthetic optical / SAR / bi-temporal sample images for smoke-testing
the pipeline without needing real satellite data or internet access."""
import os
import numpy as np
from PIL import Image

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "sample")
os.makedirs(OUT, exist_ok=True)


def make_optical(seed=0, built_shift=0):
    rng = np.random.RandomState(seed)
    h, w = 256, 256
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # background: bare soil (brownish)
    img[:, :] = [150, 120, 90]

    # vegetation block (green), top-left
    img[10:110, 10:110] = [40, 140, 60]
    img[10:110, 10:110] += rng.randint(-10, 10, (100, 100, 3)).astype(np.uint8)

    # water block (blue), bottom-left
    img[150:230, 20:110] = [30, 70, 160]

    # built-up block (bright gray), grows over time to simulate urban expansion
    x0 = 140
    x1 = min(w, 140 + 90 + built_shift)
    img[40:120, x0:x1] = [200, 200, 195]
    seg_w = x1 - x0
    img[40:120, x0:x1] += rng.randint(-5, 5, (80, seg_w, 3)).astype(np.uint8)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def make_sar_from_optical(optical_rgb):
    """Fake a co-registered SAR image: built-up -> bright/high-edge, water -> dark/smooth,
    vegetation/soil -> mid gray with speckle."""
    h, w, _ = optical_rgb.shape
    gray = optical_rgb.mean(axis=-1)
    is_water = (optical_rgb[..., 2].astype(int) - optical_rgb[..., 0].astype(int)) > 30
    is_built = gray > 180

    sar = np.full((h, w), 100, dtype=np.float32)
    sar[is_water] = 20
    sar[is_built] = 220
    speckle = np.random.RandomState(1).normal(0, 15, (h, w))
    sar = np.clip(sar + speckle, 0, 255).astype(np.uint8)
    return np.repeat(sar[:, :, None], 3, axis=-1)


if __name__ == "__main__":
    optical_t1 = make_optical(seed=0, built_shift=0)
    optical_t2 = make_optical(seed=0, built_shift=40)  # built-up expands
    sar_t1 = make_sar_from_optical(optical_t1)

    Image.fromarray(optical_t1).save(os.path.join(OUT, "optical_t1.png"))
    Image.fromarray(optical_t2).save(os.path.join(OUT, "optical_t2.png"))
    Image.fromarray(sar_t1).save(os.path.join(OUT, "sar_t1.png"))
    print("Synthetic sample images written to", os.path.abspath(OUT))
