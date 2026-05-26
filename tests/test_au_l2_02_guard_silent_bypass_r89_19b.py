"""R89-19b AU-L2-02 regression test — guard pipeline must not silently swallow.

Pre-fix: ``except Exception: pass`` around guard pipeline in app.py
and graph.py silently swallowed ALL exceptions. An attacker who
could trigger any guard-side exception (e.g., malformed payload,
upstream flaky import) would receive the *unguarded* answer back.

Post-fix: log warning + re-raise (fail-secure). Caller may choose
to catch + degrade gracefully, but the silent bypass is closed.

Test strategy: app.py and graph.py top-level imports pull heavy
runtime deps (gradio, langchain_ollama, langgraph) that aren't in
the unit-test env. We assert the fix at *source-text* level — the
canonical "guard pipeline failed" fail-secure marker is present and
the structural ``raise`` token follows the warning in the guard
pipeline block. Other ``except Exception: pass`` patterns elsewhere
in the files (memory save, locale translation retry) are out of
scope for this fix.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PY = REPO_ROOT / "app.py"
GRAPH_PY = REPO_ROOT / "src" / "arastirma_ussu" / "agent" / "graph.py"

GUARD_FAIL_SECURE_MARKER = "guard pipeline failed"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_au_l2_02_app_py_fail_secure_log_and_raise_present() -> None:
    """app.py _apply_guards must log + re-raise (fail-secure)."""
    src = _read(APP_PY)
    assert GUARD_FAIL_SECURE_MARKER in src, (
        "AU-L2-02: 'guard pipeline failed' fail-secure log marker MISSING from app.py"
    )
    # After the marker, a ``raise`` must follow within ~5 lines (the fix
    # is `logging.warning(...); raise`).
    idx = src.index(GUARD_FAIL_SECURE_MARKER)
    tail = src[idx : idx + 400]
    assert re.search(r"\n\s+raise\b", tail), (
        "AU-L2-02: re-raise statement MISSING after fail-secure log in app.py"
    )


def test_au_l2_02_graph_py_fail_secure_log_and_raise_present() -> None:
    """graph.py REPL guard block must log + re-raise (fail-secure)."""
    src = _read(GRAPH_PY)
    assert GUARD_FAIL_SECURE_MARKER in src, (
        "AU-L2-02: 'guard pipeline failed' fail-secure log marker MISSING from graph.py"
    )
    idx = src.index(GUARD_FAIL_SECURE_MARKER)
    tail = src[idx : idx + 400]
    assert re.search(r"\n\s+raise\b", tail), (
        "AU-L2-02: re-raise statement MISSING after fail-secure log in graph.py"
    )


def test_au_l2_02_layer5_import_graceful_degradation_preserved() -> None:
    """ImportError pass (Layer 5 not installed) MUST remain — design intent."""
    src = _read(GRAPH_PY)
    assert "except ImportError:" in src, (
        "AU-L2-02: ImportError graceful degradation branch removed by mistake"
    )
    assert "Layer 5 not installed" in src, (
        "AU-L2-02: Layer-5-not-installed design comment removed by mistake"
    )
