"""Gradio chat UI for Arastirma Ussu."""

from __future__ import annotations

import os
import shutil
import sys
import time
from collections.abc import Generator
from pathlib import Path

# Force UTF-8 mode process-wide — must happen before any other import
# so that open(), sys.stdout, and all libraries default to UTF-8
# instead of Windows cp1254.
os.environ["PYTHONUTF8"] = "1"
for _stream_name in ("stdout", "stderr", "stdin"):
    _stream = getattr(sys, _stream_name, None)
    if _stream and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

import gradio as gr
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from arastirma_ussu.agent.graph import (
    DEFAULT_MAX_ITERATIONS,
    _extract_sources,
    _retry_turkish,
    build_graph,
)
from arastirma_ussu.agent.prompts import FALLBACK_ANSWER, build_system_prompt
from arastirma_ussu.agent.state import AgentState
from arastirma_ussu.agent.tools import build_tool_registry

# Build once at startup
_registry = build_tool_registry()
_system_prompt = build_system_prompt(_registry)
_app = build_graph()

# ---------------------------------------------------------------------------
# Status map for tool calls
# ---------------------------------------------------------------------------

_STATUS = {
    "doc_search": "Belgeler taraniyor...",
    "web_search": "Webde araniyor...",
    "memory_search": "Hafiza kontrol ediliyor...",
    "crew_research": "Detayli arastirma yapiliyor...",
    "summarize": "Ozetleniyor...",
}


# ---------------------------------------------------------------------------
# Guard pipeline helper
# ---------------------------------------------------------------------------

def _apply_guards(result: dict, query: str) -> str:
    """Run guard pipeline on result, return final answer string."""
    answer = result.get("final_answer") or result.get("error") or FALLBACK_ANSWER

    if result.get("final_answer"):
        try:
            from arastirma_ussu.guards import GuardInput, Severity, run_guards

            sources = _extract_sources(result.get("messages", []))
            verdict = run_guards(
                GuardInput(answer=result["final_answer"], query=query, sources=sources)
            )
            if verdict.severity == Severity.FAIL:
                answer = FALLBACK_ANSWER
            elif verdict.severity == Severity.WARN:
                lang_warn = any(
                    r.guard_name == "check_language" and r.severity == Severity.WARN
                    for r in verdict.results
                )
                if lang_warn:
                    answer = _retry_turkish(answer)
        except Exception:
            pass

    return answer


def _save_memory(query: str, answer: str) -> None:
    """Save Q&A to conversation memory if answer is valid."""
    if answer and answer != FALLBACK_ANSWER:
        try:
            from arastirma_ussu.memory.store import get_memory
            get_memory().save(question=query, answer=answer)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------

def _history_to_messages(history: list[dict], max_turns: int = 3) -> list[BaseMessage]:
    """Convert Gradio chat history to LangChain messages for context."""
    recent = history[-(max_turns * 2):]
    messages: list[BaseMessage] = []
    for entry in recent:
        role = entry.get("role", "")
        content = (entry.get("content") or "")[:300]
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


# ---------------------------------------------------------------------------
# Streaming agent
# ---------------------------------------------------------------------------

def _stream_agent(
    query: str, history: list[dict] | None = None,
) -> Generator[str, None, None]:
    """Run agent with streaming status + typing effect."""
    history_msgs = _history_to_messages(history or [])
    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=_system_prompt),
            *history_msgs,
            HumanMessage(content=query),
        ],
        "iteration": 0,
        "max_iterations": DEFAULT_MAX_ITERATIONS,
        "last_action": "",
        "last_action_input": "",
        "last_observation": "",
        "final_answer": "",
        "error": "",
    }

    # Stream graph events — show status per tool call
    final_state = {}
    for event in _app.stream(initial_state):
        for node_name, update in event.items():
            final_state = {**final_state, **update}
            if node_name == "reason":
                action = update.get("last_action", "")
                if action and action in _STATUS:
                    yield f"*{_STATUS[action]}*"

    # Guard pipeline (blocking, before user sees answer)
    answer = _apply_guards(final_state, query)

    # Save to memory
    _save_memory(query, answer)

    # Typing effect — yield progressively longer substrings
    for i in range(0, len(answer), 4):
        yield answer[: i + 4]
        time.sleep(0.01)
    yield answer  # ensure complete text


# ---------------------------------------------------------------------------
# File upload + index
# ---------------------------------------------------------------------------

_SUPPORTED_EXT = {".pdf", ".txt", ".md", ".docx"}
_DOC_DIR = Path("data/documents")


def _index_documents() -> str:
    """Re-index documents in data/documents/."""
    try:
        from arastirma_ussu.ingest.index import ensure_index

        idx = ensure_index(force_rebuild=True)
        if idx:
            return "Belgeler indekslendi."
        return "data/documents/ dizininde belge bulunamadi."
    except ImportError:
        return "Layer 2 yuklu degil."
    except Exception as e:
        return f"Indeksleme hatasi: {e}"


def _handle_file_upload(files: list[str]) -> str:
    """Copy uploaded files to data/documents/ and trigger re-indexing."""
    _DOC_DIR.mkdir(parents=True, exist_ok=True)
    copied, skipped = [], []

    for file_path in files:
        p = Path(file_path)
        if p.suffix.lower() not in _SUPPORTED_EXT:
            skipped.append(p.name)
            continue
        shutil.copy2(str(p), str(_DOC_DIR / p.name))
        copied.append(p.name)

    if not copied:
        exts = ", ".join(sorted(_SUPPORTED_EXT))
        return f"Desteklenmeyen dosya turu. Desteklenen: {exts}"

    result = _index_documents()
    parts = [f"Yuklenen: {', '.join(copied)}"]
    if skipped:
        parts.append(f"Atlanan: {', '.join(skipped)}")
    parts.append(result)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def respond(message: dict | str, history: list[dict]) -> Generator[str, None, None]:
    """Gradio chat callback — generator for streaming."""
    # Multimodal: message can be {"text": ..., "files": [...]}
    if isinstance(message, dict):
        text = (message.get("text") or "").strip()
        files = message.get("files") or []
    else:
        text = message.strip()
        files = []

    # Handle file uploads
    if files:
        yield _handle_file_upload(files)
        if not text:
            return

    # Index command
    if text.lower() in ("indeksle", "reindex", "index"):
        yield _index_documents()
        return

    if not text:
        return

    yield from _stream_agent(text, history)


demo = gr.ChatInterface(
    fn=respond,
    title="Arastirma Ussu",
    multimodal=True,
    description=(
        "Yerel AI arastirma asistani — dolphin-mistral:7B | LangGraph | Qdrant\n\n"
        "Belge yuklemek icin dosyayi surukle veya atacsimgesine tikla (PDF, TXT, MD, DOCX)"
    ),
    examples=[
        {"text": "Yapay zeka nedir?"},
        {"text": "Arastirma Ussu projesinde kac katman var?"},
        {"text": "indeksle"},
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
