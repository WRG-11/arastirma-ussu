"""Targeted coverage tests for low-coverage modules — pure-Python paths
that don't need Ollama / Qdrant / ragas. Added as part of the 70 → 72
coverage ratchet (Round 14 audit)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ─── agent/prompts.build_system_prompt ────────────────────────────────────


def test_build_system_prompt_with_real_registry() -> None:
    """``build_system_prompt`` formats every registered tool's
    description + name into the ReAct template."""
    from arastirma_ussu.agent.prompts import REACT_SYSTEM_PROMPT, build_system_prompt
    from arastirma_ussu.agent.tools import TOOL_REGISTRY

    prompt = build_system_prompt(TOOL_REGISTRY)
    # Must include every tool name from the registry
    for name in TOOL_REGISTRY:
        assert name in prompt
    # And it must be a fully-formatted prompt (no unresolved braces)
    assert "{tool_names}" not in prompt
    assert "{tool_descriptions}" not in prompt
    # And it's the real template, not a fallback string
    assert "Thought:" in prompt
    assert "Final Answer:" in prompt
    # Sanity: empty registry yields a still-valid template skeleton
    empty = build_system_prompt({})
    assert "Final Answer:" in empty
    assert empty != REACT_SYSTEM_PROMPT  # was formatted


# ─── memory/tool.memory_search — happy path ────────────────────────────────


def test_memory_search_happy_path_empty_store() -> None:
    """With qdrant-client installed, ``memory_search`` returns the
    truncated formatted-result string from the memory store rather
    than the ImportError fallback."""
    pytest.importorskip("qdrant_client")
    from arastirma_ussu.memory.tool import memory_search

    # Use a stable but unique query so we don't depend on any seed data.
    result = memory_search("__coverage_probe_no_match_expected__")
    assert isinstance(result, str)
    # ImportError fallback would say "Hafiza modulu yuklu degil"; with
    # qdrant installed we should be on the happy path.
    assert "Hafiza modulu yuklu degil" not in result


# ─── memory/tool.memory_search — exception path ────────────────────────────


def test_memory_search_runtime_exception() -> None:
    """When the underlying store raises, ``memory_search`` returns the
    generic "Hafiza arama hatasi" branch."""
    from arastirma_ussu.memory import tool as memory_tool_mod

    with patch.object(
        memory_tool_mod, "memory_search", wraps=memory_tool_mod.memory_search
    ):
        # Patch the lazy-imported store function so the call raises.
        with patch(
            "arastirma_ussu.memory.store.get_memory",
            side_effect=RuntimeError("simulated qdrant down"),
        ):
            result = memory_tool_mod.memory_search("q")
    assert "Hafiza arama hatasi" in result
    assert "simulated qdrant down" in result


# ─── agent/graph._retry_turkish — fallback + happy paths ──────────────────


def test_retry_turkish_fallback_when_llm_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """If ``_create_llm`` raises, ``_retry_turkish`` falls back to the
    original english answer (the try/except: pass safety net)."""
    from arastirma_ussu.agent import graph

    def _boom() -> None:
        raise RuntimeError("ollama unreachable")

    monkeypatch.setattr(graph, "_create_llm", _boom)
    out = graph._retry_turkish("Hello, this is an English answer.")
    assert out == "Hello, this is an English answer."


def test_retry_turkish_returns_first_paragraph_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mocked LLM that returns two paragraphs → first one kept, second
    discarded (defends against the LLM re-drifting back to English)."""
    from arastirma_ussu.agent import graph

    mock_response = MagicMock()
    mock_response.content = (
        "Bu Türkçe çevirinin ilk paragrafı yeterince uzun olmalı.\n\n"
        "And this second paragraph drifted back to English garbage."
    )
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    monkeypatch.setattr(graph, "_create_llm", lambda: mock_llm)
    out = graph._retry_turkish("Hello world drifted to English.")
    assert out.startswith("Bu Türkçe çevirinin")
    assert "English garbage" not in out


def test_retry_turkish_too_short_returns_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When LLM output's first paragraph is suspiciously short (≤20
    chars), the wrapper rejects it and returns the original."""
    from arastirma_ussu.agent import graph

    mock_response = MagicMock()
    mock_response.content = "Kısa."
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    monkeypatch.setattr(graph, "_create_llm", lambda: mock_llm)
    out = graph._retry_turkish("Hello English answer.")
    assert out == "Hello English answer."
