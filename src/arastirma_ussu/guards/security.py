"""Deterministic security guards — injection, PII, hallucination indicators."""

from __future__ import annotations

import re
from datetime import datetime

from arastirma_ussu.config import GuardConfig
from arastirma_ussu.guards.types import GuardInput, GuardResult, Severity

_cfg = GuardConfig()

# ---------------------------------------------------------------------------
# Prompt injection patterns
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(previous|prior|all|the\s+above)\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+(now|a)\b.{0,30}(task|instruction|role|system)", re.IGNORECASE),
    re.compile(r"^\s*system\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"disregard\s+(your|the|all)\b", re.IGNORECASE),
    re.compile(r"```\s*system", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"<<SYS>>", re.IGNORECASE),
    re.compile(r"\[INST\]", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# PII patterns
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_PHONE_TR_RE = re.compile(r"\b(?:\+90|0)?5\d{9}\b")
_API_KEY_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),           # OpenAI
    re.compile(r"AKIA[A-Z0-9]{16}"),               # AWS
    re.compile(r"ghp_[A-Za-z0-9]{36}"),            # GitHub PAT
    re.compile(r"sk-ant-[A-Za-z0-9\-]{20,}"),      # Anthropic
]

# ---------------------------------------------------------------------------
# Hallucination patterns
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_DEGENERATE_URL_RE = re.compile(
    r"https?://(?:example\.com|placeholder|xxx|test\.test|fake\.\w+)", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Guard functions
# ---------------------------------------------------------------------------

def check_prompt_injection(inp: GuardInput, config: GuardConfig = _cfg) -> GuardResult:
    """Detect prompt injection artifacts in the answer."""
    if not config.enable_injection_check:
        return GuardResult("check_prompt_injection", Severity.PASS, "Injection check devre disi")

    matches = [p.pattern for p in _INJECTION_PATTERNS if p.search(inp.answer)]

    if len(matches) >= 2:
        return GuardResult(
            "check_prompt_injection", Severity.FAIL,
            f"Birden fazla injection pattern tespit edildi ({len(matches)} match)",
        )
    if len(matches) == 1:
        return GuardResult(
            "check_prompt_injection", Severity.WARN,
            "Injection pattern tespit edildi",
        )
    return GuardResult("check_prompt_injection", Severity.PASS, "Injection tespit edilmedi")


def check_pii_leak(inp: GuardInput, config: GuardConfig = _cfg) -> GuardResult:
    """Detect PII (email, phone, API keys) in the answer."""
    if not config.enable_pii_check:
        return GuardResult("check_pii_leak", Severity.PASS, "PII check devre disi")

    # API keys → FAIL
    for pattern in _API_KEY_PATTERNS:
        if pattern.search(inp.answer):
            return GuardResult("check_pii_leak", Severity.FAIL, "API anahtari tespit edildi")

    # Email / phone → WARN
    warnings: list[str] = []
    if _EMAIL_RE.search(inp.answer):
        warnings.append("email")
    if _PHONE_TR_RE.search(inp.answer):
        warnings.append("telefon")

    if warnings:
        return GuardResult(
            "check_pii_leak", Severity.WARN,
            f"PII tespit edildi: {', '.join(warnings)}",
        )
    return GuardResult("check_pii_leak", Severity.PASS, "PII tespit edilmedi")


def check_hallucination_indicators(
    inp: GuardInput,
    current_year: int | None = None,
) -> GuardResult:
    """Surface-level hallucination signals — future years, degenerate URLs."""
    year = current_year if current_year is not None else datetime.now().year
    threshold = year + 2

    warnings: list[str] = []

    # Future year detection
    for match in _YEAR_RE.finditer(inp.answer):
        mentioned_year = int(match.group(1))
        if mentioned_year >= threshold:
            warnings.append(f"gelecek yil referansi ({mentioned_year})")
            break

    # Degenerate URL detection
    if _DEGENERATE_URL_RE.search(inp.answer):
        warnings.append("sueruekluermez URL (placeholder)")

    if warnings:
        return GuardResult(
            "check_hallucination_indicators", Severity.WARN,
            f"Haludinasyon sinyali: {'; '.join(warnings)}",
        )
    return GuardResult("check_hallucination_indicators", Severity.PASS, "Haludinasyon sinyali yok")
