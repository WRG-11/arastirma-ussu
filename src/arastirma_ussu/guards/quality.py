"""Deterministic quality guards — length, repetition, language, ROUGE."""

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
    """Check answer length — empty/very short answers are suspicious."""
    length = len(inp.answer.strip())
    if length < config.min_answer_length:
        return GuardResult("check_length", Severity.FAIL, f"Cevap cok kisa ({length} karakter)")
    if length < config.warn_answer_length:
        return GuardResult("check_length", Severity.WARN, f"Cevap kisa ({length} karakter)")
    return GuardResult("check_length", Severity.PASS, "Uzunluk yeterli")


def check_repetition(inp: GuardInput, config: GuardConfig = _cfg) -> GuardResult:
    """Detect degenerate repetition at sentence level."""
    text = inp.answer.strip()
    if not text:
        return GuardResult("check_repetition", Severity.PASS, "Bos metin")

    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    if len(sentences) <= 1:
        return GuardResult("check_repetition", Severity.PASS, "Tek cumle")

    unique = len(set(s.lower() for s in sentences))
    ratio = unique / len(sentences)

    if ratio < config.repetition_fail_ratio:
        return GuardResult(
            "check_repetition", Severity.FAIL,
            f"Agir tekrar (benzersiz/toplam: {ratio:.2f})", score=ratio,
        )
    if ratio < config.repetition_warn_ratio:
        return GuardResult(
            "check_repetition", Severity.WARN,
            f"Tekrar tespit edildi (benzersiz/toplam: {ratio:.2f})", score=ratio,
        )
    return GuardResult("check_repetition", Severity.PASS, "Tekrar yok", score=ratio)


def check_language(inp: GuardInput, config: GuardConfig = _cfg) -> GuardResult:
    """Check for garbage text and Turkish language presence."""
    text = inp.answer.strip()
    if not text:
        return GuardResult("check_language", Severity.FAIL, "Bos metin")

    # Alpha ratio — catch garbage/numbers-only
    alpha_count = sum(1 for c in text if c.isalpha())
    alpha_ratio = alpha_count / len(text)

    if alpha_ratio < 0.3:
        return GuardResult("check_language", Severity.FAIL, f"Metin icerik orani dusuk ({alpha_ratio:.2f})")

    # TR stopword detection — catch English drift
    words = text.lower().split()
    tr_count = sum(1 for w in words if w in _TR_STOPWORDS)

    if tr_count < config.min_tr_stopwords:
        return GuardResult(
            "check_language", Severity.WARN,
            f"Turkce icerik az ({tr_count} TR stopword). Ingilizce drift olabilir.",
        )

    return GuardResult("check_language", Severity.PASS, "Dil kontrolu basarili")


def check_rouge(inp: GuardInput, config: GuardConfig = _cfg) -> GuardResult:
    """ROUGE-1 against source documents (when available)."""
    if not config.enable_rouge:
        return GuardResult("check_rouge", Severity.PASS, "ROUGE devre disi")

    if not inp.sources:
        return GuardResult("check_rouge", Severity.PASS, "Kaynak yok, ROUGE atlandi", score=None)

    try:
        from rouge_score.rouge_scorer import RougeScorer
    except ImportError:
        return GuardResult("check_rouge", Severity.PASS, "rouge-score yuklu degil, atlandi", score=None)

    scorer = RougeScorer(["rouge1"], use_stemmer=False)
    reference = " ".join(inp.sources)
    result = scorer.score(reference, inp.answer)
    r1_f1 = result["rouge1"].fmeasure

    if r1_f1 < config.rouge_warn_threshold:
        return GuardResult(
            "check_rouge", Severity.WARN,
            f"Dusuk ROUGE-1 ({r1_f1:.3f}) — kaynaklarla oertuesme az", score=r1_f1,
        )
    return GuardResult("check_rouge", Severity.PASS, f"ROUGE-1: {r1_f1:.3f}", score=r1_f1)
