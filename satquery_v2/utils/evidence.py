"""
utils/evidence.py
------------------
Composes the labelled visual-evidence boards the UI shows beneath every answer.

The existing overlay helpers in utils/visualization.py produce single images.
This module arranges several of them into one captioned board, because the
evidence only makes sense as a comparison: a change mask means nothing without
the T1 and T2 frames beside it.

Boards produced here:
  single image  -> Original | Land-cover classification
  bi-temporal   -> T1 | T2 | Difference | Change mask
  optical + SAR -> Optical | SAR | Fused result

Colour-blind note: the change mask uses magenta on a desaturated base rather
than the conventional red/green, so it stays legible under deuteranopia.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.visualization import CLASS_COLORS

_PANEL_BG = (10, 16, 27)
_PANEL_BORDER = (0, 148, 168)
_CAPTION_BG = (14, 23, 40)
_CAPTION_FG = (206, 232, 245)
_GAP = 10
_CAPTION_H = 26


def _font(size: int = 13):
    """Load a readable font, degrading to PIL's bitmap default if none exist."""
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
        "Arial.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def to_uint8_rgb(array: np.ndarray) -> np.ndarray:
    """Coerce any image array into displayable HxWx3 uint8."""
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.shape[-1] >= 3:
        array = array[..., :3]
    else:
        array = np.repeat(array[..., :1], 3, axis=-1)
    if array.dtype == np.uint8:
        return array
    finite = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if finite.max() > 1.5:
        finite = finite / 255.0
    return (np.clip(finite, 0.0, 1.0) * 255.0).astype(np.uint8)


def heatmap(values: np.ndarray) -> np.ndarray:
    """Render a 0..1 scalar field as an inferno-like RGB image.

    Hand-rolled rather than pulled from matplotlib: the base install stays free
    of a plotting dependency, and the ramp only needs to be perceptually
    monotonic, which a three-stop interpolation achieves.
    """
    values = np.asarray(values, dtype=np.float32)
    span = float(values.max() - values.min())
    norm = (values - values.min()) / span if span > 1e-6 else np.zeros_like(values)

    # Dark navy -> magenta -> amber. Monotonic in lightness.
    stops = np.array([[8, 12, 40], [156, 24, 132], [255, 186, 60]], dtype=np.float32)
    scaled = norm * (len(stops) - 1)
    lower = np.floor(scaled).astype(int)
    lower = np.clip(lower, 0, len(stops) - 2)
    frac = (scaled - lower)[..., None]
    rgb = stops[lower] * (1 - frac) + stops[lower + 1] * frac
    return np.clip(rgb, 0, 255).astype(np.uint8)


def mask_to_rgb(mask: np.ndarray, color: Tuple[int, int, int] = (255, 46, 196)) -> np.ndarray:
    """Binary mask as a flat two-tone image."""
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[...] = (16, 22, 36)
    out[mask] = color
    return out


def class_map_to_rgb(class_map: np.ndarray, class_names: Sequence[str]) -> np.ndarray:
    """Paint a class-index map with the project's land-cover palette."""
    class_map = np.asarray(class_map)
    out = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for index, name in enumerate(class_names):
        out[class_map == index] = CLASS_COLORS.get(name, (140, 140, 140))
    return out


def compose_board(
    panels: List[Tuple[str, np.ndarray]],
    panel_height: int = 260,
    max_columns: int = 4,
    legend: Optional[List[Tuple[str, Tuple[int, int, int]]]] = None,
) -> Optional[np.ndarray]:
    """Arrange captioned panels into a single evidence board.

    Panels are scaled to a shared height so like-for-like comparison is
    possible, then laid out left to right and wrapped into rows.
    """
    panels = [(title, image) for title, image in panels if image is not None]
    if not panels:
        return None

    prepared: List[Tuple[str, Image.Image]] = []
    for title, array in panels:
        pil = Image.fromarray(to_uint8_rgb(array))
        if pil.height != panel_height:
            width = max(1, int(pil.width * panel_height / max(1, pil.height)))
            pil = pil.resize((width, panel_height), Image.NEAREST)
        prepared.append((title, pil))

    columns = min(max_columns, len(prepared))
    rows = [prepared[i:i + columns] for i in range(0, len(prepared), columns)]

    row_widths = [
        sum(image.width for _, image in row) + _GAP * (len(row) - 1) for row in rows
    ]
    board_width = max(row_widths) + _GAP * 2
    row_height = panel_height + _CAPTION_H
    legend_height = 30 if legend else 0
    board_height = len(rows) * row_height + _GAP * (len(rows) + 1) + legend_height

    canvas = Image.new("RGB", (board_width, board_height), _PANEL_BG)
    draw = ImageDraw.Draw(canvas)
    font = _font(13)

    y = _GAP
    for row in rows:
        x = _GAP
        for title, image in row:
            canvas.paste(image, (x, y))
            draw.rectangle(
                [x, y, x + image.width - 1, y + image.height - 1],
                outline=_PANEL_BORDER, width=1,
            )
            draw.rectangle(
                [x, y + image.height, x + image.width - 1, y + image.height + _CAPTION_H - 1],
                fill=_CAPTION_BG,
            )
            draw.text((x + 7, y + image.height + 6), title, fill=_CAPTION_FG, font=font)
            x += image.width + _GAP
        y += row_height + _GAP

    if legend:
        x = _GAP
        for label, color in legend:
            draw.rectangle([x, y + 6, x + 13, y + 19], fill=color)
            draw.text((x + 19, y + 5), label, fill=_CAPTION_FG, font=font)
            x += 19 + int(draw.textlength(label, font=font)) + 20

    return np.array(canvas)


def landcover_legend(class_names: Sequence[str]) -> List[Tuple[str, Tuple[int, int, int]]]:
    """Legend entries for a land-cover classification panel."""
    return [
        (name.replace("_", " "), CLASS_COLORS.get(name, (140, 140, 140)))
        for name in class_names
    ]
