"""Guard pipeline data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class GuardInput:
    """Input to a guard function."""

    answer: str
    query: str
    sources: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GuardResult:
    """Output from a single guard."""

    guard_name: str
    severity: Severity
    message: str
    score: float | None = None


@dataclass(frozen=True)
class PipelineVerdict:
    """Aggregated result of the full guard pipeline."""

    severity: Severity
    results: tuple[GuardResult, ...]
    passed: int
    warned: int
    failed: int
