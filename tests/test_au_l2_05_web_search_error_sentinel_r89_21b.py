"""R89-21b AU-L2-05 regression — web_search exception → sentinel, not raw repr.

Pre-fix: ``web_search`` returned ``f"Web arama hatasi: {e}"`` on any
exception. Two risks:
  1. PII / path / env-var leakage through exception repr (urllib /
     requests / proxy lib stack-trace style payloads).
  2. Prompt-injection — an attacker shaping an exception message
     (e.g., crafted URL triggers ``ValueError("Ignore previous instructions, ...")``)
     gets that string fed back to the LLM as observation context.

Post-fix: stable sentinel ``"Web arama yapilamadi."`` + server-side
``logging.warning`` carrying the real exception for diagnostics.
"""

from __future__ import annotations

import logging
import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def ddgs_stub(monkeypatch):
    """Stub ddgs so the test never makes a real network call."""
    stub = types.ModuleType("ddgs")

    class _BoomDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def text(self, *a, **kw):
            raise RuntimeError(
                "PROXY_USER=admin:s3cret /home/user/.netrc — leaked"
            )

    stub.DDGS = _BoomDDGS  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ddgs", stub)


def test_au_l2_05_exception_returns_sentinel_not_raw_repr(ddgs_stub, caplog) -> None:
    from arastirma_ussu.agent.tools import web_search

    # Patch translation so we don't pull in Ollama
    import arastirma_ussu.agent.tools as tools_mod

    orig = tools_mod._translate_query_to_english
    tools_mod._translate_query_to_english = lambda q: q
    try:
        with caplog.at_level(logging.WARNING):
            result = web_search("test query")
    finally:
        tools_mod._translate_query_to_english = orig

    # Sentinel string returned (Turkish locale per file convention)
    assert result == "Web arama yapilamadi.", (
        f"AU-L2-05: expected sentinel; got {result!r}"
    )
    # Raw exception details MUST NOT leak into the return value
    assert "PROXY_USER" not in result
    assert "/home/user" not in result
    assert "leaked" not in result
    assert "Ignore previous" not in result  # injection guard

    # But diagnostics MUST be in server logs
    assert any(
        "web_search failed" in r.message for r in caplog.records
    ), "AU-L2-05: expected 'web_search failed' warning log for diagnostics"


def test_au_l2_05_source_has_no_raw_exception_interpolation() -> None:
    """Structural guard — the f-string anti-pattern must be gone."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "src" / "arastirma_ussu" / "agent" / "tools.py"
    )
    text = src.read_text(encoding="utf-8")
    assert 'f"Web arama hatasi: {e}"' not in text, (
        "AU-L2-05 REGRESSION: raw exception interpolation back in web_search"
    )
