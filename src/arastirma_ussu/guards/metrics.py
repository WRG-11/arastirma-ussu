"""Guard pipeline metrics — simple in-memory counters."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from arastirma_ussu.guards.types import PipelineVerdict, Severity


@dataclass
class GuardMetrics:
    """In-memory counter store for guard pipeline stats."""

    total_runs: int = 0
    total_pass: int = 0
    total_warn: int = 0
    total_fail: int = 0
    guard_triggers: dict[str, int] = field(default_factory=dict)
    rouge_scores: list[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, verdict: PipelineVerdict) -> None:
        """Record a pipeline run."""
        with self._lock:
            self.total_runs += 1
            self.total_pass += verdict.passed
            self.total_warn += verdict.warned
            self.total_fail += verdict.failed
            for r in verdict.results:
                if r.severity != Severity.PASS:
                    self.guard_triggers[r.guard_name] = (
                        self.guard_triggers.get(r.guard_name, 0) + 1
                    )
                if r.guard_name == "check_rouge" and r.score is not None:
                    self.rouge_scores.append(r.score)

    def summary(self) -> dict:
        """Return a summary dict."""
        with self._lock:
            avg_rouge = (
                sum(self.rouge_scores) / len(self.rouge_scores)
                if self.rouge_scores
                else None
            )
            return {
                "total_runs": self.total_runs,
                "total_pass": self.total_pass,
                "total_warn": self.total_warn,
                "total_fail": self.total_fail,
                "guard_triggers": dict(self.guard_triggers),
                "avg_rouge": avg_rouge,
            }

    def reset(self) -> None:
        """Clear all counters."""
        with self._lock:
            self.total_runs = 0
            self.total_pass = 0
            self.total_warn = 0
            self.total_fail = 0
            self.guard_triggers.clear()
            self.rouge_scores.clear()
