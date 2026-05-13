"""Layer 5.5 — RAGAS LLM-as-Judge skeleton.

Wraps three RAGAS metrics (faithfulness, answer_relevancy,
context_recall) behind a single ``evaluate_answer`` call. Lokal-first
disipline: pass a langchain-ollama ``ChatOllama`` instance as the
``llm`` argument so no external API calls are made.

This is a skeleton — the contract is fixed but caller must supply
``llm`` and ``embeddings`` to actually exercise RAGAS.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from .types import JudgeResult


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
        Reference answer for ``context_recall``. When ``None``, that
        metric returns ``nan``.
    llm
        A langchain-compatible LLM (e.g., ``ChatOllama``) used as the
        RAGAS judge. Required for ``faithfulness`` and
        ``answer_relevancy`` — when ``None``, those metrics return
        ``nan``.
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

    metrics = [faithfulness, answer_relevancy, context_recall]
    result = evaluate(ds, metrics=metrics, llm=llm, embeddings=embeddings)
    raw = result.to_pandas().to_dict(orient="records")[0]

    def _score(key: str) -> float:
        v = raw.get(key)
        try:
            return float(v) if v is not None else math.nan
        except (TypeError, ValueError):
            return math.nan

    return JudgeResult(
        faithfulness=_score("faithfulness"),
        answer_relevancy=_score("answer_relevancy"),
        context_recall=_score("context_recall"),
        raw=raw,
    )
