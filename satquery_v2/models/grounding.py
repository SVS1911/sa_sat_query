"""
models/grounding.py
---------------------
Text-guided region grounding: map a target land-cover class keyword to the
regions of the image that belong to it, returning a mask + bounding boxes
around the largest connected components.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Tuple

from models.base_vlm import ImageEvidence
from utils.spectral_indices import LAND_COVER_CLASSES


def _connected_components_bboxes(mask: np.ndarray, max_boxes: int = 5, min_area: int = 40) -> List[Tuple[int, int, int, int]]:
    """Simple flood-fill connected components (BFS) -> bounding boxes, sorted by area desc.
    Avoids a scikit-image dependency for the base build."""
    visited = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    boxes = []

    for y0 in range(h):
        for x0 in range(w):
            if mask[y0, x0] and not visited[y0, x0]:
                stack = [(y0, x0)]
                visited[y0, x0] = True
                min_x, max_x, min_y, max_y = x0, x0, y0, y0
                area = 0
                while stack:
                    y, x = stack.pop()
                    area += 1
                    min_x, max_x = min(min_x, x), max(max_x, x)
                    min_y, max_y = min(min_y, y), max(max_y, y)
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                if area >= min_area:
                    boxes.append((min_x, min_y, max_x, max_y, area))

    boxes.sort(key=lambda b: b[4], reverse=True)
    return [(x0, y0, x1, y1) for (x0, y0, x1, y1, _area) in boxes[:max_boxes]]


def ground(evidence: ImageEvidence, target_class: str) -> Dict[str, Any]:
    if target_class not in LAND_COVER_CLASSES:
        target_class = "built_up"

    class_idx = LAND_COVER_CLASSES.index(target_class)
    mask = evidence.class_map == class_idx
    coverage = float(mask.mean())

    # Downsample for connected-component search on large images to keep it fast,
    # then rescale boxes back up.
    h, w = mask.shape
    scale = 1
    small_mask = mask
    if max(h, w) > 256:
        scale = max(h, w) / 256
        step = int(round(scale))
        small_mask = mask[::step, ::step]

    boxes_small = _connected_components_bboxes(small_mask, max_boxes=5, min_area=max(4, int(20 / scale)))
    boxes = [(int(x0 * scale), int(y0 * scale), int(x1 * scale), int(y1 * scale)) for (x0, y0, x1, y1) in boxes_small]

    confidence = min(0.92, 0.5 + coverage)
    if not boxes:
        answer_text = (
            f"I could not find a clear {target_class.replace('_', ' ')} area. "
            f"It may cover about {coverage*100:.1f}% of the image."
        )
    else:
        answer_text = (
            f"I found {len(boxes)} {target_class.replace('_', ' ')} area(s), "
            f"covering about {coverage*100:.1f}% of the image. "
            f"The highlighted boxes show where they are."
        )

    return {
        "answer": answer_text,
        "confidence": round(confidence, 2),
        "target_class": target_class,
        "coverage_fraction": coverage,
        "mask": mask,
        "bboxes": boxes,
    }
