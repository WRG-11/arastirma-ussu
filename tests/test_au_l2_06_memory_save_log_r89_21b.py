"""R89-21b AU-L2-06 regression — memory save failure must log (not silently pass).

Pre-fix: ``app.py:_save_memory`` and ``graph.py`` REPL memory-save
block both used ``except Exception: pass``. Memory write failures
were invisible to operators (degraded memory layer ran silently).

Post-fix: log warning with the underlying exception detail; do NOT
re-raise (memory is non-critical, distinct from AU-L2-02 guards).

Source-text level test — exercising the live functions requires the
gradio + langgraph runtime stack which is heavy to mock cleanly. The
structural guards below catch a revert via grep.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PY = REPO_ROOT / "app.py"
GRAPH_PY = REPO_ROOT / "src" / "arastirma_ussu" / "agent" / "graph.py"

MARKER = "memory save failed"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_au_l2_06_app_py_memory_save_logs_warning() -> None:
    """app.py _save_memory must emit a 'memory save failed' log."""
    src = _read(APP_PY)
    assert MARKER in src, (
        f"AU-L2-06: '{MARKER}' marker MISSING from app.py _save_memory"
    )
    # No bare 'except Exception: pass' immediately under get_memory().save
    bad_pattern = (
        r"get_memory\(\)\.save\([^\n]*\n\s*except Exception:\n\s+pass"
    )
    assert not re.search(bad_pattern, src), (
        "AU-L2-06 REGRESSION: bare-except-pass back around app.py memory save"
    )


def test_au_l2_06_app_py_does_not_re_raise_after_memory_save() -> None:
    """app.py memory-save must NOT re-raise (memory is non-critical)."""
    src = _read(APP_PY)
    # Find the _save_memory function body and assert no 'raise' inside
    # the except clause specifically tied to the memory save.
    idx = src.find("def _save_memory(")
    assert idx >= 0, "AU-L2-06: _save_memory function missing"
    body = src[idx : idx + 1200]
    # Inside _save_memory the only 'raise' should be either absent or
    # commented; assert no bare 'raise' line under indentation level 8+
    bare_raise = re.search(r"\n\s{4,}raise\b", body)
    assert bare_raise is None, (
        "AU-L2-06 REGRESSION: _save_memory re-raises -- memory must fail-soft"
    )


def test_au_l2_06_graph_py_memory_save_logs_warning() -> None:
    """graph.py REPL memory-save block must emit 'memory save failed'."""
    src = _read(GRAPH_PY)
    assert MARKER in src, (
        f"AU-L2-06: '{MARKER}' marker MISSING from graph.py memory save"
    )
