"""
models/vqa.py
--------------
Single-image visual question answering (mandatory baseline).

Answers are grounded in the ImageEvidence produced by RemoteSensingVLM.encode()
(land-cover proportions today; swap in a fine-tuned VQA head later and this
module's public interface, answer(), doesn't need to change).
"""
from __future__ import annotations

import re
from typing import Dict, Any

from models.base_vlm import ImageEvidence

_PRESENCE_WORDS = ["is there", "are there", "does this", "any ", "presence of"]
_PERCENT_WORDS = ["how much", "percentage", "percent", "proportion", "fraction", "how large"]
_COUNT_WORDS = ["how many"]

_CLASS_ALIASES = {
    "water": ["water", "river", "lake", "pond", "reservoir", "flood", "coast"],
    "vegetation": ["vegetation", "forest", "tree", "crop", "farmland", "green area", "plants"],
    "built_up": ["built-up", "built up", "urban", "building", "road", "settlement", "infrastructure", "city"],
    "bare_soil": ["bare soil", "barren", "sand", "soil", "bare land"],
}


def _find_target_class(text: str):
    for cls, kws in _CLASS_ALIASES.items():
        if any(kw in text for kw in kws):
            return cls
    return None


def answer(evidence: ImageEvidence, question: str) -> Dict[str, Any]:
    text = question.lower()
    target = _find_target_class(text)
    proportions = evidence.proportions

    if target:
        pct = proportions.get(target, 0.0) * 100
        is_presence_q = any(w in text for w in _PRESENCE_WORDS)
        is_percent_q = any(w in text for w in _PERCENT_WORDS)

        if is_presence_q:
            present = pct > 3.0
            answer_text = (
                f"Yes. I can see {target.replace('_', ' ')} in about {pct:.1f}% of the image."
                if present else
                f"No large {target.replace('_', ' ')} area is visible. The estimate is {pct:.1f}% of the image."
            )
            confidence = min(0.95, 0.55 + pct / 100)
        elif is_percent_q:
            answer_text = f"About {pct:.1f}% of the image appears to be {target.replace('_', ' ')}."
            confidence = 0.65
        else:
            answer_text = f"The image contains about {pct:.1f}% {target.replace('_', ' ')}."
            confidence = 0.6
        return {"answer": answer_text, "confidence": round(confidence, 2), "target_class": target,
                "proportions": proportions}

    # Open-ended scene questions ("what is present", "what kind of scene is
    # this", "identify the land-cover types") name no single class, so the old
    # code fell through to an apology. They are the most common first question
    # a user asks, and the evidence needed to answer them is already computed —
    # so describe the composition instead of declining.
    ranked = [(cls, frac) for cls, frac in
              sorted(proportions.items(), key=lambda kv: kv[1], reverse=True) if frac > 0.02]

    if not ranked:
        return {"answer": "The image could not be resolved into recognisable land-cover types.",
                "confidence": 0.3, "target_class": None, "proportions": proportions}

    primary = ranked[0]
    described = ", ".join(
        f"{cls.replace('_', ' ')} ({frac * 100:.0f}%)" for cls, frac in ranked[:4]
    )
    scene = _scene_type(dict(ranked))
    answer_text = (
        f"This looks like {scene}. The image is mostly "
        f"{primary[0].replace('_', ' ')} at about {primary[1] * 100:.0f}% of the area, "
        f"with the full breakdown being {described}."
    )
    # Confidence tracks how dominant the leading class is: a scene that is 80%
    # one class is a far safer call than an even four-way split.
    confidence = round(min(0.85, 0.45 + primary[1] * 0.5), 2)
    return {"answer": answer_text, "confidence": confidence, "target_class": None,
            "proportions": proportions}


def _scene_type(proportions: Dict[str, float]) -> str:
    """Name the overall scene from its land-cover mix, in plain words."""
    water = proportions.get("water", 0.0)
    vegetation = proportions.get("vegetation", 0.0)
    built = proportions.get("built_up", 0.0)
    soil = proportions.get("bare_soil", 0.0)

    if built > 0.35:
        return "a built-up area — a town or city with buildings and roads"
    if water > 0.4:
        return "a mostly water scene, such as a coastline, lake, or wide river"
    if vegetation > 0.5:
        return "a vegetated landscape — forest, cropland, or parkland"
    if soil > 0.5:
        return "an open, largely bare landscape such as fields, desert, or cleared ground"
    if built > 0.1 and vegetation > 0.15:
        return "a mixed rural-to-suburban scene, with settlement among vegetation"
    if water > 0.1 and vegetation > 0.2:
        return "a mixed landscape with water and vegetation"
    return "a mixed-use landscape with no single dominant land-cover type"
