"""Gradio chat UI for Arastirma Ussu."""

from __future__ import annotations

import gradio as gr
from langchain_core.messages import HumanMessage, SystemMessage

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


def _run_agent(query: str) -> str:
    """Run the full agent pipeline and return the answer string."""
    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=_system_prompt),
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

    result = _app.invoke(initial_state)
    answer = result.get("final_answer") or result.get("error") or FALLBACK_ANSWER

    # Guard pipeline
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

    # Save to memory
    if result.get("final_answer") and answer != FALLBACK_ANSWER:
        try:
            from arastirma_ussu.memory.store import get_memory

            get_memory().save(question=query, answer=result["final_answer"])
        except Exception:
            pass

    return answer


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


def respond(message: str, history: list[dict]) -> str:
    """Gradio chat callback."""
    if message.strip().lower() in ("indeksle", "reindex", "index"):
        return _index_documents()
    return _run_agent(message)


demo = gr.ChatInterface(
    fn=respond,
    title="Arastirma Ussu",
    description=(
        "Yerel AI arastirma asistani — dolphin-mistral:7B | LangGraph | Qdrant\n\n"
        '*\"indeksle\" yazarak `data/documents/` klasorundeki belgeleri indeksleyebilirsin.*'
    ),
    examples=[
        "Yapay zeka nedir?",
        "Arastirma Ussu projesinde kac katman var?",
        "indeksle",
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
