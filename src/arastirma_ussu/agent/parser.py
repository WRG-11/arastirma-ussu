"""Regex-based ReAct output parser for dolphin-mistral."""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReActAction:
    """Parsed tool call from LLM output."""

    thought: str
    action: str
    action_input: str


@dataclass(frozen=True)
class ReActFinalAnswer:
    """Parsed final answer from LLM output."""

    thought: str
    answer: str


ReActResult = ReActAction | ReActFinalAnswer | None

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_THOUGHT_RE = re.compile(
    r"Thought\s*:\s*(.+?)(?=\n(?:Action|Final\s+Answer)\s*:)", re.DOTALL
)
_ACTION_RE = re.compile(r"Action\s*:\s*(.+?)(?:\n|$)")
_ACTION_INPUT_RE = re.compile(r"Action\s+Input\s*:\s*(.+?)(?:\n|$)", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"Final\s+Answer\s*:\s*(.+)", re.DOTALL)

_MIN_FINAL_ANSWER_LEN = 20  # chars — below this, prefer Action if present


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_react_output(text: str) -> ReActResult:
    """Parse raw LLM text into ReActAction, ReActFinalAnswer, or None.

    Precedence rules:
    * Final Answer present **and** len >= 20 → ReActFinalAnswer
    * Final Answer present but < 20 chars **and** Action present → ReActAction
      (guards against placeholder / empty Final Answer)
    * Action + Action Input present → ReActAction
    * Otherwise → None (parse failure)
    """
    text = text.strip()

    thought_match = _THOUGHT_RE.search(text)
    thought = thought_match.group(1).strip() if thought_match else ""

    fa_match = _FINAL_ANSWER_RE.search(text)
    action_match = _ACTION_RE.search(text)
    action_input_match = _ACTION_INPUT_RE.search(text)

    has_action = action_match is not None and action_input_match is not None

    if fa_match:
        answer = fa_match.group(1).strip()
        # Min-length guard: short FA + valid action → prefer action
        if len(answer) < _MIN_FINAL_ANSWER_LEN and has_action:
            return ReActAction(
                thought=thought,
                action=action_match.group(1).strip(),
                action_input=action_input_match.group(1).strip(),
            )
        return ReActFinalAnswer(thought=thought, answer=answer)

    if has_action:
        return ReActAction(
            thought=thought,
            action=action_match.group(1).strip(),
            action_input=action_input_match.group(1).strip(),
        )

    return None
