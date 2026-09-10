"""
utils/visualization.py
------------------------
Overlay helpers: class maps -> colored overlays, bounding boxes, change masks.
Pure numpy/PIL so the app has no heavyweight plotting dependency.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLASS_COLORS = {
    "water": (39, 117, 214),
    "vegetation": (46, 160, 67),
    "built_up": (214, 84, 51),
    "bare_soil": (176, 137, 84),
    "other": (140, 140, 140),
}


def overlay_class_map(rgb_uint8: np.ndarray, class_map: np.ndarray, class_names, alpha: float = 0.45) -> np.ndarray:
    overlay = rgb_uint8.copy().astype(np.float32)
    color_layer = np.zeros_like(overlay)
    for idx, name in enumerate(class_names):
        color = CLASS_COLORS.get(name, (200, 200, 200))
        mask = class_map == idx
        color_layer[mask] = color
    blended = overlay * (1 - alpha) + color_layer * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def overlay_binary_mask(rgb_uint8: np.ndarray, mask: np.ndarray, color=(255, 45, 45), alpha: float = 0.55) -> np.ndarray:
    overlay = rgb_uint8.copy().astype(np.float32)
    color_layer = np.zeros_like(overlay)
    color_layer[mask] = color
    blended = np.where(mask[..., None], overlay * (1 - alpha) + color_layer * alpha, overlay)
    return np.clip(blended, 0, 255).astype(np.uint8)


def draw_bboxes(rgb_uint8: np.ndarray, bboxes, labels=None, color=(255, 220, 0)) -> np.ndarray:
    img = Image.fromarray(rgb_uint8)
    draw = ImageDraw.Draw(img)
    labels = labels or [None] * len(bboxes)
    for (x0, y0, x1, y1), label in zip(bboxes, labels):
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        if label:
            draw.rectangle([x0, max(0, y0 - 14), x0 + 8 * len(label), y0], fill=color)
            draw.text((x0 + 2, max(0, y0 - 13)), label, fill=(0, 0, 0))
    return np.array(img)


def side_by_side(*images: np.ndarray) -> np.ndarray:
    imgs = [Image.fromarray(im) for im in images]
    h = max(im.height for im in imgs)
    imgs = [im.resize((int(im.width * h / im.height), h)) for im in imgs]
    total_w = sum(im.width for im in imgs) + 8 * (len(imgs) - 1)
    canvas = Image.new("RGB", (total_w, h), (255, 255, 255))
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.width + 8
    return np.array(canvas)
