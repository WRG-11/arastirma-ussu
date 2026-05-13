"""Layer 5.5 (RAGAS LLM-as-Judge) skeleton tests.

Most heavy paths are gated behind the ``experimental`` marker and
``ragas`` import availability; the pure-Python contract tests run as
part of the default smoke suite.
"""
from __future__ import annotations

import importlib
import math

import pytest

from arastirma_ussu.eval import JudgeResult, evaluate_answer


_RAGAS_AVAILABLE = importlib.util.find_spec("ragas") is not None


def test_judge_result_overall_excludes_nan() -> None:
    r = JudgeResult(faithfulness=0.8, answer_relevancy=0.6, context_recall=math.nan)
    assert r.overall == pytest.approx(0.7)


def test_judge_result_overall_all_nan_is_nan() -> None:
    r = JudgeResult(faithfulness=math.nan, answer_relevancy=math.nan, context_recall=math.nan)
    assert math.isnan(r.overall)


def test_judge_result_is_passing_default_threshold() -> None:
    high = JudgeResult(faithfulness=0.9, answer_relevancy=0.9, context_recall=0.9)
    low = JudgeResult(faithfulness=0.2, answer_relevancy=0.2, context_recall=0.2)
    assert high.is_passing()
    assert not low.is_passing()


def test_judge_result_is_passing_custom_threshold() -> None:
    r = JudgeResult(faithfulness=0.5, answer_relevancy=0.5, context_recall=0.5)
    assert r.is_passing(threshold=0.4)
    assert not r.is_passing(threshold=0.6)


def test_judge_result_is_passing_all_nan_is_false() -> None:
    r = JudgeResult(faithfulness=math.nan, answer_relevancy=math.nan, context_recall=math.nan)
    assert not r.is_passing()


def test_evaluate_answer_raises_without_ragas(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the ``[layer55]`` extras are missing, the wrapper must give
    a clear, actionable ImportError instead of a bare ModuleNotFoundError."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name in {"ragas", "datasets"} or name.startswith(("ragas.", "datasets.")):
            raise ImportError(f"mocked missing: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"\[layer55\]"):
        evaluate_answer("q", "a", ["c"])


@pytest.mark.skipif(not _RAGAS_AVAILABLE, reason="ragas not installed")
def test_evaluate_answer_no_llm_returns_nan_skeleton() -> None:
    """Contract path: with ragas installed but no judge LLM, return an
    all-nan JudgeResult instead of attempting an unconfigured evaluate()."""
    r = evaluate_answer("Soru?", "Cevap.", ["bağlam 1", "bağlam 2"])
    assert isinstance(r, JudgeResult)
    assert math.isnan(r.faithfulness)
    assert math.isnan(r.answer_relevancy)
    assert math.isnan(r.context_recall)
    assert r.raw.get("reason") == "no llm provided"


@pytest.mark.experimental
@pytest.mark.integration
@pytest.mark.skipif(not _RAGAS_AVAILABLE, reason="ragas not installed")
def test_evaluate_answer_with_ollama_judge_smoke() -> None:
    """End-to-end smoke: RAGAS with local Ollama judge.

    Heavy + slow. Gated behind ``experimental`` + ``integration`` so it
    runs only when the operator explicitly opts in (e.g.,
    ``pytest -m experimental``).
    """
    pytest.importorskip("langchain_ollama")
    from langchain_ollama import ChatOllama, OllamaEmbeddings  # type: ignore

    judge = ChatOllama(model="qwen2.5:7b", temperature=0)
    embed = OllamaEmbeddings(model="qwen2.5:7b")

    r = evaluate_answer(
        question="Türkiye'nin başkenti neresi?",
        answer="Türkiye'nin başkenti Ankara.",
        contexts=["Ankara, Türkiye'nin başkentidir."],
        ground_truth="Ankara",
        llm=judge,
        embeddings=embed,
    )
    assert isinstance(r, JudgeResult)
    # We don't assert on values — RAGAS metric drift makes that brittle.
