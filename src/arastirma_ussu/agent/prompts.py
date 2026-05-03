"""ReAct prompt templates for dolphin-mistral."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arastirma_ussu.agent.tools import ToolDef

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

REACT_SYSTEM_PROMPT = """\
You are a research assistant. Answer the user's question by reasoning step-by-step.

You have access to the following tools:

{tool_descriptions}

Use the following EXACT format for every response. You must ALWAYS start with \
Thought, then either use a tool OR give the Final Answer.

Thought: <your reasoning about what to do next>
Action: <tool name — exactly one of: {tool_names}>
Action Input: <input string for the tool>

After you receive an Observation (tool result), continue with another Thought.

When you have enough information to answer, respond with:

Thought: <your final reasoning>
Final Answer: <your complete answer to the user's question>

IMPORTANT RULES:
- Always start with "Thought:"
- Use EXACTLY one tool per turn (one Action + one Action Input)
- Tool names are case-sensitive: use exactly {tool_names}
- Do NOT emit Action and Final Answer in the same turn — pick one
- When you have the answer, use "Final Answer:" — do not call another tool
- If a tool returns an error, think about what went wrong and try differently
- crew_research is a FINAL synthesis step — call it at most ONCE per question
"""

OBSERVATION_TEMPLATE = "\nObservation: {observation}\n"

FALLBACK_ANSWER = (
    "Uzgunum, sorunuzu yeterince arastiramadam. "
    "Lutfen farkli bir sekilde sormayi deneyin."
)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_system_prompt(registry: dict[str, ToolDef]) -> str:
    """Format the system prompt with actual tool descriptions."""
    descriptions: list[str] = []
    for tool in registry.values():
        descriptions.append(f"- {tool.name}: {tool.description}")

    tool_names = ", ".join(registry.keys())
    tool_descriptions = "\n".join(descriptions)

    return REACT_SYSTEM_PROMPT.format(
        tool_descriptions=tool_descriptions,
        tool_names=tool_names,
    )
