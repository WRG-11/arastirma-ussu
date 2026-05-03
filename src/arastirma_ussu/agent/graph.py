"""LangGraph ReAct agent — manual parsing, no native tool-calling."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from arastirma_ussu.agent.parser import ReActAction, ReActFinalAnswer, parse_react_output
from arastirma_ussu.agent.prompts import (
    FALLBACK_ANSWER,
    OBSERVATION_TEMPLATE,
    build_system_prompt,
)
from arastirma_ussu.agent.state import AgentState
from arastirma_ussu.agent.tools import ToolDef, build_tool_registry
from arastirma_ussu.config import OllamaConfig

DEFAULT_MAX_ITERATIONS = 6


# ---------------------------------------------------------------------------
# Routing (module-level so tests can import it)
# ---------------------------------------------------------------------------

def route(state: AgentState) -> str:
    """Decide next node after reason_node."""
    if state.get("final_answer"):
        return END
    if state.get("iteration", 0) >= state.get(
        "max_iterations", DEFAULT_MAX_ITERATIONS
    ):
        return END
    if state.get("error"):
        return END
    if state.get("last_action"):
        return "action"
    return END


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _create_llm(config: OllamaConfig | None = None) -> ChatOllama:
    cfg = config or OllamaConfig()
    return ChatOllama(
        model=cfg.model,
        base_url=cfg.base_url,
        temperature=0.3,  # low for ReAct format compliance
        num_predict=cfg.max_tokens,
        num_ctx=8192,  # dolphin-mistral default is 2048 — too small
    )


# ---------------------------------------------------------------------------
# Graph builder (closure pattern for testability)
# ---------------------------------------------------------------------------

def build_graph(
    llm: ChatOllama | None = None,
    registry: dict[str, ToolDef] | None = None,
):
    """Build and compile the ReAct StateGraph.

    Parameters let tests inject a mock LLM / custom registry.
    """
    _llm = llm or _create_llm()
    _registry = registry or build_tool_registry()

    # -- nodes ---------------------------------------------------------------

    def reason_node(state: AgentState) -> dict:
        """Call LLM, parse ReAct output, return partial state."""
        iteration = state.get("iteration", 0) + 1

        try:
            response = _llm.invoke(state["messages"])
            raw_text: str = response.content
        except Exception as e:
            return {
                "iteration": iteration,
                "final_answer": "",
                "last_action": "",
                "last_action_input": "",
                "error": f"LLM hatasi: {e}",
            }

        parsed = parse_react_output(raw_text)
        new_messages = [AIMessage(content=raw_text)]

        if isinstance(parsed, ReActFinalAnswer):
            return {
                "messages": new_messages,
                "iteration": iteration,
                "final_answer": parsed.answer,
                "last_action": "",
                "last_action_input": "",
                "error": "",
            }

        if isinstance(parsed, ReActAction):
            return {
                "messages": new_messages,
                "iteration": iteration,
                "final_answer": "",
                "last_action": parsed.action,
                "last_action_input": parsed.action_input,
                "error": "",
            }

        # Parse failure
        return {
            "messages": new_messages,
            "iteration": iteration,
            "final_answer": "",
            "last_action": "",
            "last_action_input": "",
            "error": f"LLM ciktisi parse edilemedi: {raw_text[:200]}",
        }

    def action_node(state: AgentState) -> dict:
        """Execute the selected tool, return observation as HumanMessage."""
        action = state.get("last_action", "")
        action_input = state.get("last_action_input", "")

        if action not in _registry:
            observation = (
                f"Hata: '{action}' bilinmeyen bir arac. "
                f"Mevcut araclar: {', '.join(_registry.keys())}"
            )
        else:
            try:
                observation = _registry[action].func(action_input)
            except Exception as e:
                observation = f"Tool error: {e}"

        obs_message = HumanMessage(
            content=OBSERVATION_TEMPLATE.format(observation=observation)
        )
        return {
            "messages": [obs_message],
            "last_observation": observation,
        }

    # -- wiring --------------------------------------------------------------

    graph = StateGraph(AgentState)
    graph.add_node("reason", reason_node)
    graph.add_node("action", action_node)

    graph.add_edge(START, "reason")
    graph.add_conditional_edges("reason", route, {"action": "action", END: END})
    graph.add_edge("action", "reason")

    return graph.compile()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Interactive REPL — ``arastirma`` console_script."""
    print("Arastirma Ussu — AI Arastirma Asistani")
    print("Komutlar: 'q' cikis | 'indeksle' belge indeksleme\n")

    app = build_graph()
    registry = build_tool_registry()
    system_prompt = build_system_prompt(registry)

    while True:
        try:
            query = input("Soru: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGorusmeler!")
            break

        if query.lower() in ("q", "quit", "exit", "cikis"):
            print("Gorusmeler!")
            break

        if not query:
            continue

        if query.lower() in ("indeksle", "reindex", "index"):
            print("Belgeler indeksleniyor...")
            try:
                from arastirma_ussu.ingest.index import ensure_index

                idx = ensure_index(force_rebuild=True)
                if idx:
                    print("Indeksleme tamamlandi.\n")
                else:
                    print("data/documents/ dizininde belge bulunamadi.\n")
            except ImportError:
                print("Layer 2 yuklu degil. pip install -e '.[layer2]'\n")
            except Exception as e:
                print(f"Indeksleme hatasi: {e}\n")
            continue

        initial_state: AgentState = {
            "messages": [
                SystemMessage(content=system_prompt),
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

        result = app.invoke(initial_state)

        answer = result.get("final_answer") or result.get("error") or FALLBACK_ANSWER
        print(f"\nYanit: {answer}\n")
