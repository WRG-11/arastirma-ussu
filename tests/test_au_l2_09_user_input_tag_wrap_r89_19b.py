"""R89-19b AU-L2-09 regression test — translation prompt wraps user query.

Pre-fix: ``_maybe_translate_query`` built the LLM prompt as
``f"Translate this to Turkish. Only output the Turkish translation:\n{query}"``.
Any malicious query was concatenated directly to the instruction —
classic prompt-injection pre-guard bypass surface.

Post-fix: query wrapped in ``<USER_INPUT>...</USER_INPUT>`` tags with
explicit "Translate the text INSIDE the tags" framing. Model still
attacks the injected text but the instruction boundary is clear.

Source-text level test (heavy top-level imports preclude in-process
exercise of _maybe_translate_query without large mock infra).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PY = REPO_ROOT / "app.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_au_l2_09_translation_prompt_uses_user_input_tags() -> None:
    """Translation prompt must wrap raw query in <USER_INPUT> tags."""
    src = _read(APP_PY)
    assert "<USER_INPUT>" in src and "</USER_INPUT>" in src, (
        "AU-L2-09: <USER_INPUT> delimiter tags MISSING from translation prompt"
    )
    # The unsafe inline-interpolation pattern must be gone
    unsafe_pattern = 'Translate this to Turkish. Only output the Turkish translation:\\n{query}'
    assert unsafe_pattern not in src, (
        "AU-L2-09 REGRESSION: unsafe raw-query-into-prompt interpolation back"
    )


def test_au_l2_09_translation_prompt_mentions_tag_boundary() -> None:
    """Prompt should explicitly tell the model to translate text *inside* tags."""
    src = _read(APP_PY)
    # Loose match — anchors on the instruction framing, not exact wording
    assert "inside <USER_INPUT>" in src or "<USER_INPUT> tags" in src, (
        "AU-L2-09: tag-boundary instruction framing MISSING from translation prompt"
    )
