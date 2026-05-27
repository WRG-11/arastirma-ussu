"""Deterministic quality guards - length, repetition, language, ROUGE."""

from __future__ import annotations

import re

from arastirma_ussu.config import GuardConfig
from arastirma_ussu.guards.types import GuardInput, GuardResult, Severity

_cfg = GuardConfig()

# Turkish stopwords for language drift detection
_TR_STOPWORDS = {"ve", "ile", "için", "bir", "bu", "da", "de", "olan", "olarak", "gibi", "daha", "en"}

# Sentence boundary pattern
_SENTENCE_RE = re.compile(r"[.!?]+\s+")


def check_length(inp: GuardInput, config: GuardConfig = _cfg) -> GuardResult:
    """Check answer length - empty/very short answers are suspicious."""
    length = len(inp.answer.strip())
    if length < config.min_answer_length:
        return GuardResult("check_length", Severity.FAIL, f"Answer is too short ({length} chars)")
    if length < config.warn_answer_length:
        return GuardResult("check_length", Severity.WARN, f"Answer is short ({length} chars)")
    return GuardResult("check_length", Severity.PASS, "Length OK")


def check_repetition(inp: GuardInput, config: GuardConfig = _cfg) -> GuardResult:
    """Detect degenerate repetition at sentence level."""
    text = inp.answer.strip()
    if not text:
        return GuardResult("check_repetition", Severity.PASS, "Empty text")

    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    if len(sentences) <= 1:
        return GuardResult("check_repetition", Severity.PASS, "Single sentence")

    unique = len(set(s.lower() for s in sentences))
    ratio = unique / len(sentences)

    if ratio < config.repetition_fail_ratio:
        return GuardResult(
            "check_repetition", Severity.FAIL,
            f"Heavy repetition (unique/total: {ratio:.2f})", score=ratio,
        )
    if ratio < config.repetition_warn_ratio:
        return GuardResult(
            "check_repetition", Severity.WARN,
            f"Repetition detected (unique/total: {ratio:.2f})", score=ratio,
        )
    return GuardResult("check_repetition", Severity.PASS, "No repetition", score=ratio)


def check_language(inp: GuardInput, config: GuardConfig = _cfg) -> GuardResult:
    """Check for garbage text and Turkish language presence."""
    text = inp.answer.strip()
    if not text:
        return GuardResult("check_language", Severity.FAIL, "Empty text")

    # Alpha ratio - catch garbage/numbers-only
    alpha_count = sum(1 for c in text if c.isalpha())
    alpha_ratio = alpha_count / len(text)

    if alpha_ratio < 0.3:
        return GuardResult("check_language", Severity.FAIL, f"Low alpha ratio ({alpha_ratio:.2f})")

    # TR stopword detection - catch English drift in the Turkish-language product surface
    words = text.lower().split()
    tr_count = sum(1 for w in words if w in _TR_STOPWORDS)

    if tr_count < config.min_tr_stopwords:
        return GuardResult(
            "check_language", Severity.WARN,
            f"Low Turkish content ({tr_count} TR stopwords); possible English drift.",
        )

    return GuardResult("check_language", Severity.PASS, "Language check passed")


def check_rouge(inp: GuardInput, config: GuardConfig = _cfg) -> GuardResult:
    """ROUGE-1 against source documents (when available)."""
    if not config.enable_rouge:
        return GuardResult("check_rouge", Severity.PASS, "ROUGE disabled")

    if not inp.sources:
        return GuardResult("check_rouge", Severity.PASS, "No sources, ROUGE skipped", score=None)

    try:
        from rouge_score.rouge_scorer import RougeScorer
    except ImportError:
        return GuardResult("check_rouge", Severity.PASS, "rouge-score not installed, skipped", score=None)

    scorer = RougeScorer(["rouge1"], use_stemmer=False)
    reference = " ".join(inp.sources)
    result = scorer.score(reference, inp.answer)
    r1_f1 = result["rouge1"].fmeasure

    if r1_f1 < config.rouge_warn_threshold:
        return GuardResult(
            "check_rouge", Severity.WARN,
            f"Low ROUGE-1 ({r1_f1:.3f}) - little overlap with sources", score=r1_f1,
        )
    return GuardResult("check_rouge", Severity.PASS, f"ROUGE-1: {r1_f1:.3f}", score=r1_f1)
