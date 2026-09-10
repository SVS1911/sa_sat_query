"""
controller/execution_trace.py
------------------------------
The observable execution trace shown to the user.

Scope note: this records *what the system did* — which validator ran, which
workflow was selected, which endpoint was called, how long it took. It is
deliberately not a reasoning log. No internal chain-of-thought is captured
here, and none should be added: the trace exists so an analyst can audit the
pipeline, not to narrate deliberation.

Each step carries a status so the UI can render it as a checklist:

    ✓ Image validation
    ✓ Workflow selected: Change detection
    ⚠ Trained model API unreachable — using local specialist
    ✓ Change map generated
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

OK = "ok"
WARN = "warn"
FAIL = "fail"
INFO = "info"

_SYMBOLS = {OK: "✓", WARN: "⚠", FAIL: "✗", INFO: "•"}


@dataclass
class TraceStep:
    label: str
    status: str = OK
    detail: Optional[str] = None
    elapsed_ms: Optional[float] = None

    def render(self, show_detail: bool = True) -> str:
        symbol = _SYMBOLS.get(self.status, "•")
        line = f"{symbol} {self.label}"
        if self.elapsed_ms is not None and self.elapsed_ms >= 1:
            line += f"  ({self.elapsed_ms:.0f} ms)"
        if show_detail and self.detail:
            line += f"\n    {self.detail}"
        return line


@dataclass
class ExecutionTrace:
    """Ordered list of observable pipeline steps."""

    steps: List[TraceStep] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    _mark: float = field(default_factory=time.time, repr=False)

    def add(self, label: str, status: str = OK, detail: Optional[str] = None,
            timed: bool = False) -> "ExecutionTrace":
        elapsed = None
        now = time.time()
        if timed:
            elapsed = (now - self._mark) * 1000.0
        self._mark = now
        self.steps.append(TraceStep(label=label, status=status, detail=detail,
                                    elapsed_ms=elapsed))
        return self

    def ok(self, label: str, detail: Optional[str] = None, timed: bool = False):
        return self.add(label, OK, detail, timed)

    def warn(self, label: str, detail: Optional[str] = None):
        return self.add(label, WARN, detail)

    def fail(self, label: str, detail: Optional[str] = None):
        return self.add(label, FAIL, detail)

    def info(self, label: str, detail: Optional[str] = None):
        return self.add(label, INFO, detail)

    def reset_timer(self) -> None:
        self._mark = time.time()

    @property
    def total_ms(self) -> float:
        return (time.time() - self.started_at) * 1000.0

    def has_failure(self) -> bool:
        return any(step.status == FAIL for step in self.steps)

    def render(self, header: str = "SATQUERY EXECUTION TRACE",
               show_detail: bool = True) -> str:
        lines = [header, "=" * len(header)]
        lines.extend(step.render(show_detail=show_detail) for step in self.steps)
        lines.append("")
        lines.append(f"Total pipeline time: {self.total_ms / 1000.0:.2f}s")
        return "\n".join(lines)

    def as_list(self) -> List[str]:
        """Flat strings, kept for backwards compatibility with the old audit log."""
        return [step.render(show_detail=True) for step in self.steps]

    def as_dicts(self) -> List[Dict[str, Any]]:
        return [
            {
                "label": step.label,
                "status": step.status,
                "detail": step.detail,
                "elapsed_ms": step.elapsed_ms,
            }
            for step in self.steps
        ]
