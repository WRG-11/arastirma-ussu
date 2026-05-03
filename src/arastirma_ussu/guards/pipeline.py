"""Guard pipeline runner — collects all guard results into a verdict."""

from __future__ import annotations

from arastirma_ussu.config import GuardConfig
from arastirma_ussu.guards.quality import check_language, check_length, check_repetition, check_rouge
from arastirma_ussu.guards.security import (
    check_hallucination_indicators,
    check_pii_leak,
    check_prompt_injection,
)
from arastirma_ussu.guards.types import GuardInput, GuardResult, PipelineVerdict, Severity

_cfg = GuardConfig()


def run_guards(inp: GuardInput, config: GuardConfig = _cfg) -> PipelineVerdict:
    """Run all guards, return aggregated verdict.

    All guards always run (no short-circuit) so metrics capture everything.
    """
    results: list[GuardResult] = [
        check_length(inp, config),
        check_repetition(inp, config),
        check_language(inp, config),
        check_rouge(inp, config),
        check_prompt_injection(inp, config),
        check_pii_leak(inp, config),
        check_hallucination_indicators(inp),
    ]

    passed = sum(1 for r in results if r.severity == Severity.PASS)
    warned = sum(1 for r in results if r.severity == Severity.WARN)
    failed = sum(1 for r in results if r.severity == Severity.FAIL)

    if failed > 0:
        severity = Severity.FAIL
    elif warned > 0:
        severity = Severity.WARN
    else:
        severity = Severity.PASS

    return PipelineVerdict(
        severity=severity,
        results=tuple(results),
        passed=passed,
        warned=warned,
        failed=failed,
    )
