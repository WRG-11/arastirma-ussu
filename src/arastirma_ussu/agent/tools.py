"""Tool definitions for the ReAct agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_OBS_CHARS = 2000  # ~500 tokens — context window guard


def _truncate(text: str, max_chars: int = _MAX_OBS_CHARS) -> str:
    """Truncate text to *max_chars*, appending an ellipsis if cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated)"


# ---------------------------------------------------------------------------
# Tool registry type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolDef:
    """A tool the agent can invoke."""

    name: str
    description: str
    func: Callable[[str], str]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def web_search(query: str) -> str:
    """Search the web via DuckDuckGo and return top results."""
    from duckduckgo_search import DDGS  # lazy — no network in smoke tests

    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=3)
        if not results:
            return "Arama sonucu bulunamadi."
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            lines.append(f"{i}. {title}\n   {body}\n   URL: {href}")
        return _truncate("\n\n".join(lines))
    except Exception as e:
        return f"Web arama hatasi: {e}"


def summarize(text: str) -> str:
    """Extractive summarizer — no LLM call, just first-sentence heuristic."""
    text = text.strip()
    if not text:
        return "Ozetlenecek metin bos."
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    if len(sentences) <= 3:
        return text
    parts = sentences[:3]
    if sentences[-1] not in parts:
        parts.append(sentences[-1])
    return _truncate(". ".join(parts) + ".")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def build_tool_registry() -> dict[str, ToolDef]:
    """Build and return the tool registry dict."""
    from arastirma_ussu.ingest.tool import doc_search  # lazy — Layer 2
    from arastirma_ussu.memory.tool import memory_search  # lazy — Layer 3

    return {
        "web_search": ToolDef(
            name="web_search",
            description="Search the web for current information. Input: search query string.",
            func=web_search,
        ),
        "summarize": ToolDef(
            name="summarize",
            description="Summarize a long text into key points. Input: the text to summarize.",
            func=summarize,
        ),
        "doc_search": ToolDef(
            name="doc_search",
            description=(
                "Search your local document library for relevant information. "
                "Input: search query string. Returns matching document chunks."
            ),
            func=doc_search,
        ),
        "memory_search": ToolDef(
            name="memory_search",
            description=(
                "Search your conversation memory for relevant past questions and answers. "
                "Input: search query string. Returns similar past Q&A pairs."
            ),
            func=memory_search,
        ),
    }


TOOL_REGISTRY: dict[str, ToolDef] = build_tool_registry()
