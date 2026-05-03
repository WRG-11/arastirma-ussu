"""Layer 4 smoke + integration tests for CrewAI multi-agent crew."""

import time

import pytest

from arastirma_ussu.crew.tool import crew_research

# ═══════════════════════════════════════════════════════════════════════════
# Smoke tests — no CrewAI or Ollama needed
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.smoke
class TestCrewToolSmoke:
    def test_import(self):
        assert crew_research is not None

    def test_callable(self):
        assert callable(crew_research)

    def test_returns_string(self):
        result = crew_research("test query")
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.smoke
class TestCrewConfig:
    def test_defaults(self):
        from arastirma_ussu.config import CrewConfig

        cfg = CrewConfig()
        assert cfg.timeout == 600
        assert cfg.max_agent_iter == 3
        assert cfg.temperature == 0.4
        assert cfg.verbose is False
        assert cfg.llm_request_timeout == 60

    def test_in_app_config(self):
        from arastirma_ussu.config import AppConfig, CrewConfig

        app = AppConfig()
        assert isinstance(app.crew, CrewConfig)

    def test_crew_timeout_warn_removed(self):
        from arastirma_ussu.config import AppConfig

        assert not hasattr(AppConfig(), "crew_timeout_warn")


@pytest.mark.smoke
class TestToolRegistryCrewResearch:
    def test_registry_has_crew_research(self):
        from arastirma_ussu.agent.tools import build_tool_registry

        registry = build_tool_registry()
        assert "crew_research" in registry

    def test_registry_callable(self):
        from arastirma_ussu.agent.tools import build_tool_registry

        registry = build_tool_registry()
        assert callable(registry["crew_research"].func)

    def test_description_contains_once(self):
        """Tool description should discourage repeated calls."""
        from arastirma_ussu.agent.tools import build_tool_registry

        registry = build_tool_registry()
        desc = registry["crew_research"].description.lower()
        assert "once" in desc


@pytest.mark.smoke
class TestTimeout:
    def test_fast_function_succeeds(self):
        from arastirma_ussu.crew.timeout import run_with_timeout

        result = run_with_timeout(lambda: "ok", timeout=5)
        assert result == "ok"

    def test_slow_function_raises(self):
        from arastirma_ussu.crew.timeout import run_with_timeout

        with pytest.raises(TimeoutError):
            run_with_timeout(lambda: time.sleep(10), timeout=1)


@pytest.mark.smoke
class TestPromptCrewRule:
    def test_prompt_contains_crew_once_rule(self):
        from arastirma_ussu.agent.prompts import REACT_SYSTEM_PROMPT

        assert "crew_research" in REACT_SYSTEM_PROMPT
        assert "ONCE" in REACT_SYSTEM_PROMPT


# ═══════════════════════════════════════════════════════════════════════════
# Smoke tests — CrewAI installed, no Ollama
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.smoke
class TestCrewBuild:
    def test_build_agents(self, skip_no_crewai):
        from arastirma_ussu.crew.agents import build_agents

        # Mock LLM — just needs to exist, not actually call Ollama
        agents = build_agents("ollama/dolphin-mistral")
        assert "analyst" in agents
        assert "writer" in agents
        assert len(agents) == 2

    def test_build_tasks(self, skip_no_crewai):
        from arastirma_ussu.crew.agents import build_agents
        from arastirma_ussu.crew.tasks import build_tasks

        agents = build_agents("ollama/dolphin-mistral")
        tasks = build_tasks(agents, "test query")
        assert len(tasks) == 2

    def test_agents_no_delegation(self, skip_no_crewai):
        from arastirma_ussu.crew.agents import build_agents

        agents = build_agents("ollama/dolphin-mistral")
        for agent in agents.values():
            assert agent.allow_delegation is False

    def test_agents_no_tools(self, skip_no_crewai):
        from arastirma_ussu.crew.agents import build_agents

        agents = build_agents("ollama/dolphin-mistral")
        for agent in agents.values():
            assert agent.tools == []

    def test_task_context_chaining(self, skip_no_crewai):
        """report_task should have context=[analysis_task]."""
        from arastirma_ussu.crew.agents import build_agents
        from arastirma_ussu.crew.tasks import build_tasks

        agents = build_agents("ollama/dolphin-mistral")
        tasks = build_tasks(agents, "test query")
        analysis_task, report_task = tasks
        assert report_task.context == [analysis_task]


# ═══════════════════════════════════════════════════════════════════════════
# Integration tests — need Ollama running
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
def test_crew_kickoff(skip_no_ollama, skip_no_crewai):
    """Full crew execution with dolphin-mistral — may take 30-120s."""
    from arastirma_ussu.crew.crew import run_crew

    result = run_crew("Python programlama dilinin avantajlari nelerdir?")
    assert isinstance(result, str)
    assert len(result) > 20


@pytest.mark.integration
def test_crew_research_tool_e2e(skip_no_ollama, skip_no_crewai):
    """crew_research tool end-to-end."""
    result = crew_research("Yapay zeka arastirma yontemleri hakkinda kisa bir analiz yap.")
    assert isinstance(result, str)
    assert len(result) > 20
    assert "hata" not in result.lower() or "zaman asimi" not in result.lower()
