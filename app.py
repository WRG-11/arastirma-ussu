"""Gradio chat UI for Araştırma Üssü."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from collections.abc import Generator
from pathlib import Path

# Force UTF-8 mode process-wide — must happen before any other import
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
# Constants
# ---------------------------------------------------------------------------

_SUPPORTED_EXT = {".pdf", ".txt", ".md", ".docx"}
_DOC_DIR = Path("data/documents")

_STATUS = {
    "doc_search": "Belgeler taranıyor...",
    "web_search": "Web'de aranıyor...",
    "memory_search": "Hafıza kontrol ediliyor...",
    "crew_research": "Detaylı araştırma yapılıyor...",
    "summarize": "Özetleniyor...",
}

_WELCOME = """\
Merhaba! Ben **Araştırma Üssü**, yerel AI araştırma asistanınım.

**Neler yapabilirim:**
- Sorularınızı araştırıp Türkçe cevaplarım
- Belgelerinizi (PDF, TXT, MD, DOCX) indeksleyip içinden bilgi bulurum
- Önceki konuşmalarımızı hatırlarım
- Web'den güncel bilgi ararım

**Nasıl kullanılır:**
- Soru yazın ve gönderin
- Dosya yüklemek için ataç simgesine tıklayın
- `indeksle` yazarak belgeleri yeniden indeksleyin

*qwen2.5:3B modeli ile tamamen yerel çalışıyorum — verileriniz bilgisayarınızdan çıkmaz.*\
"""


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
        except Exception as exc:
            # R89-19b AU-L2-02: was `except Exception: pass` — silently
            # swallowed guard failures (security bypass class). Now
            # log + re-raise (fail-secure). Caller is responsible for
            # mapping to user-facing FALLBACK_ANSWER if desired.
            logging.warning("guard pipeline failed: %s", exc)
            raise

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
# English → Turkish query translation
# ---------------------------------------------------------------------------

_TR_WORDS = {"ve", "ile", "bir", "bu", "ne", "nedir", "nasil", "nasıl", "kaç",
             "kac", "neden", "kim", "hangi", "var", "yok", "icin", "için"}


def _maybe_translate_query(query: str) -> str:
    """If query looks English, translate to Turkish before agent runs."""
    words = query.lower().split()
    if not words:
        return query
    tr_count = sum(1 for w in words if w in _TR_WORDS)
    if tr_count / len(words) > 0.2:
        return query
    if len(words) < 3:
        return query
    try:
        from arastirma_ussu.agent.graph import _create_llm
        llm = _create_llm()
        resp = llm.invoke([HumanMessage(
            content=f"Translate this to Turkish. Only output the Turkish translation:\n{query}"
        )])
        translated = resp.content.strip().split("\n")[0].strip()
        if translated and len(translated) > 5:
            return translated
    except Exception:
        pass
    return query


# ---------------------------------------------------------------------------
# Streaming agent
# ---------------------------------------------------------------------------

def _stream_agent(
    query: str, history: list[dict] | None = None,
) -> Generator[str, None, None]:
    """Run agent with streaming status + typing effect.

    Yields partial answer strings. The last yield is the complete answer
    with metadata footer appended.
    """
    query = _maybe_translate_query(query)

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
    start = time.time()
    final_state = {}
    tools_used: list[str] = []

    for event in _app.stream(initial_state):
        for node_name, update in event.items():
            final_state = {**final_state, **update}
            if node_name == "reason":
                action = update.get("last_action", "")
                if action and action in _STATUS:
                    tools_used.append(action)
                    yield f"*{_STATUS[action]}*"

    elapsed = time.time() - start

    # Guard pipeline
    answer = _apply_guards(final_state, query)

    # Save to memory
    _save_memory(query, answer)

    # Build metadata footer
    tool_labels = {
        "doc_search": "Belge", "web_search": "Web",
        "memory_search": "Hafıza", "crew_research": "Ekip",
        "summarize": "Özet",
    }
    used = list(dict.fromkeys(tools_used))  # dedupe, keep order
    tool_str = ", ".join(tool_labels.get(t, t) for t in used) if used else "Doğrudan"
    meta = f"\n\n---\n*{elapsed:.1f}s · Kaynak: {tool_str}*"

    # Typing effect
    for i in range(0, len(answer), 4):
        yield answer[: i + 4]
        time.sleep(0.01)
    yield answer + meta


# ---------------------------------------------------------------------------
# File upload + index
# ---------------------------------------------------------------------------

def _index_documents() -> str:
    """Re-index documents in data/documents/."""
    try:
        from arastirma_ussu.ingest.index import ensure_index

        idx = ensure_index(force_rebuild=True)
        if idx:
            return "Belgeler indekslendi."
        return "data/documents/ dizininde belge bulunamadı."
    except ImportError:
        return "Layer 2 yüklü değil."
    except Exception as e:
        return f"İndeksleme hatası: {e}"


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
        return f"Desteklenmeyen dosya türü. Desteklenen: {exts}"

    result = _index_documents()
    parts = [f"Yüklenen: {', '.join(copied)}"]
    if skipped:
        parts.append(f"Atlanan: {', '.join(skipped)}")
    parts.append(result)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sidebar helpers
# ---------------------------------------------------------------------------

def _get_doc_list() -> str:
    """Return markdown list of indexed documents."""
    if not _DOC_DIR.exists():
        return "*Belge yok*"
    files = sorted(
        f for f in _DOC_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in _SUPPORTED_EXT
    )
    if not files:
        return "*Belge yok*"
    lines = [f"- `{f.name}` ({f.stat().st_size / 1024:.0f} KB)" for f in files]
    return "\n".join(lines)


def _get_memory_count() -> str:
    """Return conversation memory point count."""
    try:
        from arastirma_ussu.memory.store import get_memory
        count = get_memory().count()
        return f"**{count}** kayıt (max 5000)"
    except Exception:
        return "Bağlantı yok"


def _clear_memory() -> str:
    """Clear conversation memory."""
    try:
        from arastirma_ussu.memory.store import get_memory
        get_memory().clear()
        return "Hafıza temizlendi."
    except Exception:
        return "Hata oluştu."


def _refresh_sidebar():
    """Return updated sidebar values."""
    return _get_doc_list(), _get_memory_count()


# ---------------------------------------------------------------------------
# Gradio Blocks layout
# ---------------------------------------------------------------------------

def respond(message: dict | str, history: list[dict]) -> Generator[str, None, None]:
    """Gradio chat callback — generator for streaming."""
    if isinstance(message, dict):
        text = (message.get("text") or "").strip()
        files = message.get("files") or []
    else:
        text = message.strip()
        files = []

    if files:
        yield _handle_file_upload(files)
        if not text:
            return

    if text.lower() in ("indeksle", "reindex", "index"):
        yield _index_documents()
        return

    if not text:
        return

    yield from _stream_agent(text, history)


with gr.Blocks(title="Araştırma Üssü") as demo:

    # --- Header ---
    gr.Markdown("# Araştırma Üssü")
    gr.Markdown(
        "Yerel AI araştırma asistanı — qwen2.5:3B | LangGraph | Qdrant"
    )

    with gr.Row():
        # --- Main chat area (left, wider) ---
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": _WELCOME}],
                height=520,
                render_markdown=True,
                buttons=["copy"],
            )
            msg = gr.MultimodalTextbox(
                placeholder="Sorunuzu yazın veya dosya yükleyin...",
                file_count="multiple",
                show_label=False,
            )

        # --- Sidebar (right, narrower) ---
        with gr.Column(scale=1, min_width=250):
            gr.Markdown("### Belgeler")
            doc_list = gr.Markdown(_get_doc_list())
            index_btn = gr.Button("Yeniden İndeksle", size="sm")
            index_status = gr.Markdown("")

            gr.Markdown("---")
            gr.Markdown("### Konuşma Hafızası")
            mem_count = gr.Markdown(_get_memory_count())
            clear_btn = gr.Button("Hafızayı Temizle", size="sm", variant="stop")
            clear_status = gr.Markdown("")

            gr.Markdown("---")
            gr.Markdown("### Hızlı Sorular")
            ex1 = gr.Button("Yapay zeka nedir?", size="sm", variant="secondary")
            ex2 = gr.Button("Kaç katman var?", size="sm", variant="secondary")
            ex3 = gr.Button("Python nedir?", size="sm", variant="secondary")

    # --- Event handlers ---

    def user_submit(message, history):
        """Add user message to history."""
        if isinstance(message, dict):
            text = (message.get("text") or "").strip()
        else:
            text = message.strip()
        if not text:
            return history, gr.MultimodalTextbox(value=None)
        history = history + [{"role": "user", "content": text}]
        return history, gr.MultimodalTextbox(value=None)

    def bot_respond(message, history):
        """Stream bot response."""
        if not history:
            return history
        # Handle file uploads from multimodal input
        if isinstance(message, dict):
            files = message.get("files") or []
            if files:
                upload_result = _handle_file_upload(files)
                history = history + [{"role": "assistant", "content": upload_result}]
                yield history

        user_msg = history[-1]["content"] if history and history[-1]["role"] == "user" else ""
        if not user_msg:
            return history

        # Streaming response
        history = history + [{"role": "assistant", "content": ""}]
        for chunk in respond(user_msg, history[:-1]):
            history[-1]["content"] = chunk
            yield history

    def do_index():
        result = _index_documents()
        return result, _get_doc_list()

    def do_clear():
        result = _clear_memory()
        return result, _get_memory_count()

    def ask_example(question):
        return gr.MultimodalTextbox(value={"text": question})

    # Wire events
    submit_event = msg.submit(
        user_submit, [msg, chatbot], [chatbot, msg]
    ).then(
        bot_respond, [msg, chatbot], chatbot
    ).then(
        _refresh_sidebar, None, [doc_list, mem_count]
    )

    index_btn.click(do_index, None, [index_status, doc_list])
    clear_btn.click(do_clear, None, [clear_status, mem_count])
    ex1.click(ask_example, gr.State("Yapay zeka nedir?"), msg)
    ex2.click(ask_example, gr.State("Araştırma Üssü projesinde kaç katman var?"), msg)
    ex3.click(ask_example, gr.State("Python nedir?"), msg)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, i18n="tr")
