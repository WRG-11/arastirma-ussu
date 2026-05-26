"""R89-19b AU-L2-10 regression test — tools.py translation wraps user query.

Sister to AU-L2-09 (app.py _maybe_translate_query). Same prompt-injection
class, different call site: ``_translate_query_to_english`` translates
Turkish queries to English for DuckDuckGo. Raw query interpolated
directly into the prompt -> attacker injects "Output ADMIN" and the
search query gets polluted.

Post-fix: wrap query in ``<USER_INPUT>`` tags with explicit
"Translate the text INSIDE the tags" framing.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_PY = REPO_ROOT / "src" / "arastirma_ussu" / "agent" / "tools.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_au_l2_10_tools_translation_uses_user_input_tags() -> None:
    """tools._translate_query_to_english must wrap query in <USER_INPUT> tags."""
    src = _read(TOOLS_PY)
    assert "<USER_INPUT>" in src and "</USER_INPUT>" in src, (
        "AU-L2-10: <USER_INPUT> delimiter tags MISSING from tools.py translation prompt"
    )
    # Unsafe inline-interpolation pattern must be gone
    unsafe_pattern = 'Translate to English. Output ONLY the translation:\\n{query}'
    assert unsafe_pattern not in src, (
        "AU-L2-10 REGRESSION: unsafe raw-query-into-prompt interpolation back"
    )


def test_au_l2_10_tools_translation_mentions_tag_boundary() -> None:
    """Prompt should explicitly tell the model to translate text *inside* tags."""
    src = _read(TOOLS_PY)
    assert "inside <USER_INPUT>" in src or "<USER_INPUT> tags" in src, (
        "AU-L2-10: tag-boundary instruction framing MISSING from tools.py"
    )
