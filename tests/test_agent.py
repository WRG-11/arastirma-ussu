"""Layer 1 smoke + integration tests for the ReAct agent."""

import pytest
from langgraph.graph import END

from arastirma_ussu.agent.parser import (
    ReActAction,
    ReActFinalAnswer,
    parse_react_output,
)
from arastirma_ussu.agent.tools import (
    TOOL_REGISTRY,
    _truncate,
    build_tool_registry,
    summarize,
)
from arastirma_ussu.agent.graph import build_graph, route

# ═══════════════════════════════════════════════════════════════════════════
# Smoke tests — no Ollama needed
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.smoke
class TestParserAction:
    def test_simple_action(self):
        text = (
            "Thought: I need to search for info\n"
            "Action: web_search\n"
            "Action Input: python langgraph tutorial"
        )
        result = parse_react_output(text)
        assert isinstance(result, ReActAction)
        assert result.action == "web_search"
        assert result.action_input == "python langgraph tutorial"
        assert "search" in result.thought.lower()

    def test_action_extra_whitespace(self):
        text = (
            "Thought:  I should look this up \n"
            "Action:  summarize \n"
            "Action Input:  some long text here "
        )
        result = parse_react_output(text)
        assert isinstance(result, ReActAction)
        assert result.action == "summarize"
        assert result.action_input == "some long text here"


@pytest.mark.smoke
class TestParserFinalAnswer:
    def test_simple_final_answer(self):
        text = (
            "Thought: I have enough information now\n"
            "Final Answer: Python was created by Guido van Rossum in 1991."
        )
        result = parse_react_output(text)
        assert isinstance(result, ReActFinalAnswer)
        assert "Guido" in result.answer
        assert "enough" in result.thought.lower()

    def test_multiline_final_answer(self):
        text = (
            "Thought: Let me summarize\n"
            "Final Answer: Here are the key points:\n"
            "1. First point\n"
            "2. Second point"
        )
        result = parse_react_output(text)
        assert isinstance(result, ReActFinalAnswer)
        assert "First point" in result.answer
        assert "Second point" in result.answer


@pytest.mark.smoke
class TestParserEdgeCases:
    def test_malformed_returns_none(self):
        result = parse_react_output("Hello world, no format here at all")
        assert result is None

    def test_empty_string(self):
        result = parse_react_output("")
        assert result is None

    def test_multiline_thought(self):
        text = (
            "Thought: I need to think about this carefully.\n"
            "There are several factors to consider.\n"
            "Let me search for more information.\n"
            "Action: web_search\n"
            "Action Input: multi factor analysis"
        )
        result = parse_react_output(text)
        assert isinstance(result, ReActAction)
        assert "several factors" in result.thought

    def test_final_answer_precedence_long(self):
        """Long Final Answer wins even if Action is also present."""
        text = (
            "Thought: done\n"
            "Action: web_search\n"
            "Action Input: test query\n"
            "Final Answer: This is a sufficiently long answer that should take precedence."
        )
        result = parse_react_output(text)
        assert isinstance(result, ReActFinalAnswer)

    def test_short_final_answer_prefers_action(self):
        """Short (<20 char) Final Answer + valid Action → prefer Action."""
        text = (
            "Thought: let me check\n"
            "Action: web_search\n"
            "Action Input: python info\n"
            "Final Answer: ok"
        )
        result = parse_react_output(text)
        assert isinstance(result, ReActAction)
        assert result.action == "web_search"


@pytest.mark.smoke
class TestToolRegistry:
    def test_has_required_tools(self):
        assert "web_search" in TOOL_REGISTRY
        assert "summarize" in TOOL_REGISTRY

    def test_callables(self):
        for tool_def in TOOL_REGISTRY.values():
            assert callable(tool_def.func)

    def test_build_returns_fresh_dict(self):
        r1 = build_tool_registry()
        r2 = build_tool_registry()
        assert r1 is not r2
        assert set(r1.keys()) == set(r2.keys())


@pytest.mark.smoke
class TestSummarize:
    def test_short_text_unchanged(self):
        result = summarize("Short text.")
        assert result == "Short text."

    def test_empty_text(self):
        result = summarize("")
        assert "bos" in result.lower()

    def test_long_text_shortened(self):
        sentences = ". ".join(f"Sentence number {i}" for i in range(10)) + "."
        result = summarize(sentences)
        assert len(result) < len(sentences)


@pytest.mark.smoke
class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("hello", 100) == "hello"

    def test_long_text_truncated(self):
        long = "x" * 5000
        result = _truncate(long, 2000)
        assert len(result) < 2100
        assert result.endswith("(truncated)")


@pytest.mark.smoke
class TestRoute:
    def _state(self, **overrides) -> dict:
        base = {
            "final_answer": "",
            "iteration": 1,
            "max_iterations": 6,
            "error": "",
            "last_action": "",
            "last_action_input": "",
            "last_observation": "",
            "messages": [],
        }
        base.update(overrides)
        return base

    def test_final_answer_ends(self):
        assert route(self._state(final_answer="done")) == END

    def test_max_iterations_ends(self):
        assert route(self._state(iteration=6, max_iterations=6)) == END

    def test_error_ends(self):
        assert route(self._state(error="parse fail")) == END

    def test_action_continues(self):
        assert route(self._state(last_action="web_search")) == "action"

    def test_empty_state_ends(self):
        assert route(self._state()) == END


@pytest.mark.smoke
class TestGraphCompile:
    def test_build_graph_with_mock_llm(self):
        """build_graph should compile even with a dummy LLM object."""

        class _MockLLM:
            def invoke(self, messages):
                class _Resp:
                    content = "Thought: test\nFinal Answer: mock answer for testing"
                return _Resp()

        app = build_graph(llm=_MockLLM())
        assert app is not None


# ═══════════════════════════════════════════════════════════════════════════
# Integration tests — need Ollama running
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
def test_full_graph_simple(skip_no_ollama):
    """Simple question — agent should produce a final answer."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from arastirma_ussu.agent.prompts import build_system_prompt
    from arastirma_ussu.agent.tools import build_tool_registry

    registry = build_tool_registry()
    system_prompt = build_system_prompt(registry)
    app = build_graph()

    result = app.invoke({
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Python programlama dilinin yaraticisi kimdir?"),
        ],
        "iteration": 0,
        "max_iterations": 6,
        "last_action": "",
        "last_action_input": "",
        "last_observation": "",
        "final_answer": "",
        "error": "",
    })

    assert result.get("final_answer"), f"No final answer. Error: {result.get('error')}"


@pytest.mark.integration
def test_full_graph_with_search(skip_no_ollama):
    """Question needing web search — agent should use a tool."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from arastirma_ussu.agent.prompts import build_system_prompt
    from arastirma_ussu.agent.tools import build_tool_registry

    registry = build_tool_registry()
    system_prompt = build_system_prompt(registry)
    app = build_graph()

    result = app.invoke({
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content="LangGraph 2026 son surum numarasi nedir?"),
        ],
        "iteration": 0,
        "max_iterations": 6,
        "last_action": "",
        "last_action_input": "",
        "last_observation": "",
        "final_answer": "",
        "error": "",
    })

    assert result.get("iteration", 0) > 1 or result.get("final_answer")


@pytest.mark.integration
def test_full_graph_max_iterations(skip_no_ollama):
    """With max_iterations=1, agent must terminate quickly."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from arastirma_ussu.agent.prompts import build_system_prompt
    from arastirma_ussu.agent.tools import build_tool_registry

    registry = build_tool_registry()
    system_prompt = build_system_prompt(registry)
    app = build_graph()

    result = app.invoke({
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Detayli bir analiz yap: yapay zekanin gelecegi"),
        ],
        "iteration": 0,
        "max_iterations": 1,
        "last_action": "",
        "last_action_input": "",
        "last_observation": "",
        "final_answer": "",
        "error": "",
    })

    assert result.get("iteration", 0) <= 2
