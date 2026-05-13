"""Layer 5.5 — RAGAS LLM-as-Judge (deneysel).

Skeleton module wraps RAGAS metrics (faithfulness, answer_relevancy,
context_recall) behind a single ``evaluate_answer`` call. Real RAGAS
execution requires the ``[layer55]`` extra (``ragas`` + ``datasets``)
plus a configured judge LLM (Ollama recommended for lokal-first).

Tests for this layer are gated behind the ``experimental`` pytest
marker and skipped by default.
"""
from __future__ import annotations

from .types import JudgeResult
from .ragas_judge import evaluate_answer

__all__ = ["JudgeResult", "evaluate_answer"]
