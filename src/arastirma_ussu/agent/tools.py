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

def _translate_query_to_english(query: str) -> str:
    """Translate a Turkish query to English for better DuckDuckGo results."""
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage as HMsg

        from arastirma_ussu.config import OllamaConfig

        cfg = OllamaConfig()
        llm = ChatOllama(
            model=cfg.model, base_url=cfg.base_url,
            temperature=0.1, num_predict=64, num_ctx=2048,
        )
        # R89-19b AU-L2-10: wrap raw user query in delimiters to break
        # prompt injection (sister to AU-L2-09 in app.py). User query
        # was concatenated directly into translation instruction.
        resp = llm.invoke([HMsg(
            content=(
                "Translate the text inside <USER_INPUT> tags to English.\n"
                "Output ONLY the English translation, not the tags.\n"
                "<USER_INPUT>\n"
                f"{query}\n"
                "</USER_INPUT>"
            )
        )])
        translated = resp.content.strip().split("\n")[0].strip()
        if translated and len(translated) > 3:
            return translated
    except Exception:
        pass
    return query


def web_search(query: str) -> str:
    """Search the web via DuckDuckGo and return top results.

    Translates Turkish queries to English for better search quality.
    """
    from ddgs import DDGS  # lazy — no network in smoke tests; duckduckgo-search renamed to ddgs 2026

    # Translate to English for better DuckDuckGo results
    en_query = _translate_query_to_english(query)

    try:
        with DDGS() as ddgs:
            results = ddgs.text(en_query, max_results=3)
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
    from arastirma_ussu.crew.tool import crew_research  # lazy — Layer 4
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
        "crew_research": ToolDef(
            name="crew_research",
            description=(
                "Delegate to a multi-agent crew for DEEP research ONLY. "
                "NEVER use for simple questions. Use ONLY when the user explicitly "
                "asks for detailed/deep analysis. Use at most ONCE. Very slow (~60s)."
            ),
            func=crew_research,
        ),
    }


TOOL_REGISTRY: dict[str, ToolDef] = build_tool_registry()
