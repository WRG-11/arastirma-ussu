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


@pytest.mark.skipif(
    importlib.util.find_spec("langchain_ollama") is None,
    reason="langchain-ollama not installed",
)
def test_default_ollama_judge_default_model() -> None:
    """Helper returns a (ChatOllama, OllamaEmbeddings) pair wired to
    qwen2.5:7b (tek-model default) without invoking Ollama."""
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    from arastirma_ussu.eval import default_ollama_judge

    llm, embeddings = default_ollama_judge()
    assert isinstance(llm, ChatOllama)
    assert isinstance(embeddings, OllamaEmbeddings)
    assert llm.model == "qwen2.5:7b"
    assert embeddings.model == "qwen2.5:7b"


@pytest.mark.skipif(
    importlib.util.find_spec("langchain_ollama") is None,
    reason="langchain-ollama not installed",
)
def test_default_ollama_judge_custom_model_and_base_url() -> None:
    """Custom model + base_url propagate to both wrappers."""
    from arastirma_ussu.eval import default_ollama_judge

    llm, embeddings = default_ollama_judge(
        model="llama3.2:3b", base_url="http://remote:11434"
    )
    assert llm.model == "llama3.2:3b"
    assert embeddings.model == "llama3.2:3b"


def test_default_ollama_judge_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Surface a clear ImportError when langchain-ollama is missing."""
    import builtins

    from arastirma_ussu.eval import default_ollama_judge

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "langchain_ollama":
            raise ImportError("mocked: langchain_ollama missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="langchain-ollama"):
        default_ollama_judge()


@pytest.mark.experimental
@pytest.mark.integration
@pytest.mark.skipif(not _RAGAS_AVAILABLE, reason="ragas not installed")
def test_evaluate_answer_good_golden() -> None:
    """E2E good-path golden: an answer fully supported by context should
    score finite metrics across the board.

    Heavy + slow. Gated behind ``experimental`` + ``integration`` so it
    runs only when the operator explicitly opts in
    (``pytest -m experimental``).
    """
    pytest.importorskip("langchain_ollama")
    from arastirma_ussu.eval import default_ollama_judge

    llm, embeddings = default_ollama_judge()
    r = evaluate_answer(
        question="Türkiye'nin başkenti neresi?",
        answer="Türkiye'nin başkenti Ankara.",
        contexts=["Ankara, Türkiye'nin başkentidir."],
        ground_truth="Ankara",
        llm=llm,
        embeddings=embeddings,
    )
    assert isinstance(r, JudgeResult)
    # At least one metric should be finite — RAGAS + Ollama can be
    # stochastic, so we don't assert on absolute values but we do
    # assert at least one metric path completed.
    finite = [
        v for v in (r.faithfulness, r.answer_relevancy, r.context_recall)
        if not math.isnan(v)
    ]
    assert len(finite) >= 1, f"all metrics nan; raw={r.raw}"


@pytest.mark.experimental
@pytest.mark.integration
@pytest.mark.skipif(not _RAGAS_AVAILABLE, reason="ragas not installed")
def test_evaluate_answer_bad_golden() -> None:
    """E2E bad-path golden: an answer contradicting the context should
    not score 1.0 on faithfulness.

    Stochastic + heavy: a single failed call yields no signal, so we
    require both that the call completes and that faithfulness (when
    finite) is strictly below 1.0. Other metrics are not asserted
    because RAGAS LLM-as-judge can drift.
    """
    pytest.importorskip("langchain_ollama")
    from arastirma_ussu.eval import default_ollama_judge

    llm, embeddings = default_ollama_judge()
    r = evaluate_answer(
        question="Türkiye'nin başkenti neresi?",
        answer="Türkiye'nin başkenti İstanbul.",  # contradicts context
        contexts=["Ankara, Türkiye'nin başkentidir."],
        ground_truth="Ankara",
        llm=llm,
        embeddings=embeddings,
    )
    assert isinstance(r, JudgeResult)
    if not math.isnan(r.faithfulness):
        assert r.faithfulness < 1.0, (
            f"contradicting answer scored perfect faithfulness; raw={r.raw}"
        )
