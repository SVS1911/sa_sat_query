"""
controller/query_parser.py
----------------------------
Interprets the natural-language query and classifies the requested task,
constrained by which input scenario was detected (single / cross_modal /
bi_temporal). Keyword/rule-based by default (transparent + auditable);
`set_llm_backend()` lets you plug in an LLM-based classifier later without
touching the controller.

Tasks:
  single image  -> "vqa" | "caption" | "grounding"
  cross_modal   -> "fusion_analysis"
  bi_temporal   -> "change_vqa" | "change_description"
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable

_llm_backend: Optional[Callable[[str, str], Dict[str, Any]]] = None


def set_llm_backend(fn: Callable[[str, str], Dict[str, Any]]) -> None:
    """Optional: register an LLM-based classifier fn(query, scenario) -> {task, params}."""
    global _llm_backend
    _llm_backend = fn


@dataclass
class TaskPlan:
    task: str
    scenario: str
    query: str
    params: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


_GROUNDING_TRIGGERS = [
    r"\bhighlight\b", r"\blocate\b", r"\bwhere is\b", r"\bfind the\b",
    r"\bpoint out\b", r"\bshow (me )?the\b", r"\bmark the\b", r"\bregion (of|referred)\b",
]
_CAPTION_TRIGGERS = [
    r"\bdescribe\b", r"\bcaption\b", r"\bsummari[sz]e\b", r"\bwhat does this image show\b",
    r"\bgeneral (overview|description)\b", r"\bscene description\b",
]
_CHANGE_DESC_TRIGGERS = [
    r"\bwhat changed\b", r"\bdescribe the change\b", r"\bchanges? between\b",
    r"\bchange (map|analysis)\b",
]
_CHANGE_QUESTION_TRIGGERS = [
    r"\bincreased\b", r"\bdecreased\b", r"\bincrease or decrease\b",
    r"\bremained unchanged\b", r"\bhow much .* (increase|decrease|change)d?\b",
    r"\bby how much\b",
]
_FUSION_TRIGGERS = [
    r"\boptical and sar\b", r"\bsar and optical\b", r"\btogether\b", r"\bcombine\b",
    r"\bjoint(ly)?\b", r"\bfus(e|ion)\b",
]

_TARGET_KEYWORDS = {
    "water": ["water", "river", "lake", "pond", "reservoir", "coast", "flood"],
    "vegetation": ["vegetation", "forest", "tree", "crop", "farmland", "green"],
    "built_up": ["built-up", "built up", "urban", "building", "road", "settlement", "infrastructure"],
    "bare_soil": ["bare soil", "barren", "sand", "soil"],
}


def _match_any(patterns, text) -> bool:
    return any(re.search(p, text) for p in patterns)


def _detect_target_class(text: str) -> Optional[str]:
    for cls, kws in _TARGET_KEYWORDS.items():
        if any(kw in text for kw in kws):
            return cls
    return None


def parse(query: str, scenario: str) -> TaskPlan:
    if _llm_backend is not None:
        try:
            result = _llm_backend(query, scenario)
            return TaskPlan(task=result["task"], scenario=scenario, query=query,
                             params=result.get("params", {}), rationale="LLM backend classification.")
        except Exception:
            pass  # fall through to rule-based parsing

    text = query.lower().strip()
    target = _detect_target_class(text)

    if scenario == "single":
        if _match_any(_GROUNDING_TRIGGERS, text):
            return TaskPlan("grounding", scenario, query,
                             params={"target_class": target or "built_up"},
                             rationale="Grounding trigger phrase (highlight/locate/show the region, etc.) detected.")
        if _match_any(_CAPTION_TRIGGERS, text):
            return TaskPlan("caption", scenario, query,
                             rationale="Caption/description trigger phrase detected.")
        # default single-image behavior: VQA is mandatory baseline
        return TaskPlan("vqa", scenario, query,
                         rationale="No caption/grounding trigger matched; defaulting to VQA "
                                    "(mandatory single-image baseline).")

    if scenario == "cross_modal":
        return TaskPlan("fusion_analysis", scenario, query,
                         params={"target_class": target},
                         rationale="Cross-modal (optical+SAR) pair -> fusion/extraction workflow.")

    if scenario == "bi_temporal":
        if _match_any(_CHANGE_QUESTION_TRIGGERS, text):
            return TaskPlan("change_vqa", scenario, query,
                             params={"target_class": target},
                             rationale="Change-VQA trigger phrase (increase/decrease/changed?) detected.")
        return TaskPlan("change_description", scenario, query,
                         params={"target_class": target},
                         rationale="Bi-temporal pair with descriptive change query -> change description "
                                    "(+ change map).")

    return TaskPlan("invalid", scenario, query, rationale="Unrecognized scenario.")
