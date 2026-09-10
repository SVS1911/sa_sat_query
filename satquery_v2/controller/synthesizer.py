"""
controller/synthesizer.py
--------------------------
Turns raw workflow outputs into the final answer the user reads.

The one rule this module enforces: **text and numbers come from different
places, and the answer says which is which.** Semantic content (what the scene
is, what changed) may come from the trained model API. Every quantity comes
from utils/numerical.py, measured on the actual pixels. The synthesizer never
lets a model-authored sentence carry a figure the pipeline did not compute.

Each answer therefore has a visible provenance line, e.g.

    Source: trained model API (scene label) + measured pixel statistics
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from api.hf_model import ModelAnswer

SOURCE_API = "fine-tuned vision-language model"
SOURCE_LOCAL = "local specialist"


def provenance_line(model_used: bool, local_used: bool, measured: bool,
                    detail: Optional[str] = None) -> str:
    """One line telling the user where each part of the answer came from."""
    parts: List[str] = []
    if model_used:
        parts.append(f"{SOURCE_API} ({detail})" if detail else SOURCE_API)
    if local_used:
        parts.append(f"{SOURCE_LOCAL} ({detail})" if detail and not model_used else SOURCE_LOCAL)
    if measured:
        parts.append("measured pixel statistics")
    if not parts:
        return "Source: unavailable"
    return "Source: " + " + ".join(parts)


def format_confidence(confidence: Optional[float]) -> str:
    return f"{confidence:.2f}" if confidence is not None else "not reported by the model"


def model_unavailable_notice(reason: Optional[str]) -> str:
    """The standard, non-alarming message when the model cannot answer."""
    base = ("The fine-tuned vision-language model did not answer this request, "
            "so the description below comes from SatQuery's local specialists.")
    if reason:
        # Kept on one line: this renders inside an indented NOTES block.
        return f"{base} ({' '.join(str(reason).split())})"
    return base


def capability_notice(capability: str) -> str:
    """Told to the user when the endpoint genuinely cannot do something."""
    return (
        f"This capability ({capability}) is not currently available from the "
        f"configured model. The answer below uses SatQuery's local specialist "
        f"instead, and is labelled accordingly."
    )


def describe_model_output(answer: "ModelAnswer") -> List[str]:
    """Render the model's own output verbatim — no paraphrase, no invention."""
    if not answer.ok:
        return []
    lines: List[str] = []
    if answer.text:
        lines.append(f"Model output: {answer.text}")
    if answer.labels:
        lines.append(f"Land-cover classes named by the model: {', '.join(answer.labels)}")
    if answer.generation_confidence is not None:
        # Deliberately not called "confidence": this is the decoder's mean
        # token probability for the text it produced, which says how sure the
        # model was of its *wording*, not of a classification.
        lines.append(
            f"Generation confidence: {answer.generation_confidence:.2f} "
            f"(decoder certainty in its own wording, not a class probability)"
        )
    if answer.scores:
        top = sorted(answer.scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
        lines.append("Model-reported scores: "
                     + ", ".join(f"{n} {v:.3f}" for n, v in top))
    return lines


def compose(
    headline: str,
    measurement_lines: Sequence[str] = (),
    model_lines: Sequence[str] = (),
    notes: Sequence[str] = (),
    provenance: Optional[str] = None,
) -> str:
    """Assemble the final answer block in a consistent order.

    Order matters: the direct answer first (people read the first line and
    stop), then hard measurements, then the model's raw fields, then caveats.
    """
    blocks: List[str] = [headline.strip()]

    if measurement_lines:
        blocks.append("MEASURED FROM THE IMAGE\n" + "\n".join(f"  {line}" for line in measurement_lines))
    if model_lines:
        blocks.append("TRAINED MODEL OUTPUT\n" + "\n".join(f"  {line}" for line in model_lines))
    if notes:
        blocks.append("NOTES\n" + "\n".join(f"  {note}" for note in notes))
    if provenance:
        blocks.append(provenance)

    return "\n\n".join(block for block in blocks if block.strip())


def change_narrative(
    changed_percent: float,
    location: str,
    class_shift: Optional[str],
    before_label: Optional[str],
    after_label: Optional[str],
) -> str:
    """Plain-language change summary built only from measured inputs.

    `before_label`/`after_label` are the trained model's scene labels when it
    supplied them; they are quoted, not embellished. Everything numeric in the
    sentence is passed in already measured.
    """
    if changed_percent < 1.0:
        headline = "The two images look almost identical. No substantial change was detected."
    else:
        where = f", concentrated in {location}" if location else ""
        headline = (
            f"About {changed_percent:.1f}% of the image changed between the two dates{where}."
        )

    if before_label and after_label:
        if before_label.strip().lower() == after_label.strip().lower():
            headline += (
                f" The trained model classifies both dates as '{before_label}', so the "
                f"overall scene type did not change."
            )
        else:
            headline += (
                f" The trained model classifies the earlier image as '{before_label}' "
                f"and the later image as '{after_label}'."
            )

    if class_shift:
        headline += f" {class_shift}"
    return headline
