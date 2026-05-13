"""Layer 5.5 — RAGAS LLM-as-Judge.

Wraps three RAGAS metrics (faithfulness, answer_relevancy,
context_recall) behind a single ``evaluate_answer`` call. Lokal-first
disipline: ``default_ollama_judge`` returns a (llm, embeddings) pair
wired to a local Ollama instance so no external API calls are made.

End-to-end proven via experimental marker tests (see test_eval.py).
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from .types import JudgeResult


def default_ollama_judge(
    model: str = "qwen2.5:7b",
    base_url: str | None = None,
) -> tuple[Any, Any]:
    """Return a ``(llm, embeddings)`` pair wired to a local Ollama instance.

    Both halves are langchain-compatible wrappers (``ChatOllama`` +
    ``OllamaEmbeddings``) so they plug straight into
    ``evaluate_answer(llm=..., embeddings=...)``.

    Parameters
    ----------
    model
        Ollama model tag. Defaults to ``qwen2.5:7b`` to match the
        project-wide tek-model disiplini.
    base_url
        Ollama HTTP endpoint. ``None`` lets langchain-ollama pick
        ``OLLAMA_HOST`` / ``http://localhost:11434``.

    Raises
    ------
    ImportError
        When ``langchain-ollama`` is not installed (it's a core
        dependency, so this normally only fires in stripped envs).
    """
    try:
        from langchain_ollama import ChatOllama, OllamaEmbeddings
    except ImportError as e:
        raise ImportError(
            "default_ollama_judge requires langchain-ollama"
        ) from e

    kwargs: dict[str, Any] = {"model": model}
    if base_url is not None:
        kwargs["base_url"] = base_url
    llm = ChatOllama(temperature=0, **kwargs)
    embeddings = OllamaEmbeddings(**kwargs)
    return llm, embeddings


def evaluate_answer(
    question: str,
    answer: str,
    contexts: Sequence[str],
    *,
    ground_truth: str | None = None,
    llm: Any = None,
    embeddings: Any = None,
) -> JudgeResult:
    """Run a 3-metric RAGAS evaluation on a single Q/A/context tuple.

    Parameters
    ----------
    question
        The user's original query.
    answer
        The system's generated answer to evaluate.
    contexts
        Retrieved passages the answer was grounded in (RAG context).
    ground_truth
        Reference answer for ``context_recall``. When ``None``, the
        answer itself is used so the metric still has a target.
    llm
        A langchain-compatible LLM (e.g., ``ChatOllama``) used as the
        RAGAS judge. Required for ``faithfulness`` and
        ``answer_relevancy`` — when ``None``, an all-nan skeleton
        ``JudgeResult`` is returned.
    embeddings
        A langchain-compatible embeddings model used by RAGAS for
        semantic similarity. Required for ``answer_relevancy``.

    Returns
    -------
    JudgeResult
        Per-metric scores plus the raw RAGAS row.

    Raises
    ------
    ImportError
        When the ``[layer55]`` extras (``ragas``, ``datasets``) are not
        installed.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        # NOTE: ragas 0.4 has two metric trees:
        #   - ``ragas.metrics`` (legacy, pre-instantiated, accepts
        #     ``LangchainLLMWrapper`` + Ollama)
        #   - ``ragas.metrics.collections`` (modern, class-based but
        #     requires ``InstructorLLM`` via OpenAI client; Ollama
        #     wiring is not first-class as of 0.4.3)
        # We stay on the legacy tree until the modern tree gets a
        # local-LLM factory; only cost is DeprecationWarnings.
        from ragas.metrics import (
            answer_relevancy,
            context_recall,
            faithfulness,
        )
    except ImportError as e:
        raise ImportError(
            "Layer 5.5 RAGAS requires `pip install -e \".[layer55]\"`"
        ) from e

    if llm is None:
        # No judge LLM → return all-nan skeleton result for callers who
        # only want to exercise the contract.
        return JudgeResult(
            faithfulness=float("nan"),
            answer_relevancy=float("nan"),
            context_recall=float("nan"),
            raw={"reason": "no llm provided"},
        )

    ds = Dataset.from_dict(
        {
            "question": [question],
            "answer": [answer],
            "contexts": [list(contexts)],
            "ground_truth": [ground_truth or answer],
        }
    )

    metrics: list[Any] = [faithfulness, context_recall]
    if embeddings is not None:
        metrics.insert(1, answer_relevancy)
    result = evaluate(ds, metrics=metrics, llm=llm, embeddings=embeddings)
    raw = result.to_pandas().to_dict(orient="records")[0]

    def _score(*keys: str) -> float:
        """Try several possible RAGAS column names — class instances
        sometimes emit CamelCase, snake_case, or hyphenated keys."""
        for key in keys:
            v = raw.get(key)
            if v is None:
                continue
            try:
                f = float(v)
                if not math.isnan(f):
                    return f
            except (TypeError, ValueError):
                continue
        return math.nan

    return JudgeResult(
        faithfulness=_score("faithfulness", "Faithfulness"),
        answer_relevancy=_score("answer_relevancy", "AnswerRelevancy"),
        context_recall=_score("context_recall", "ContextRecall"),
        raw=raw,
    )
