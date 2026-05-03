"""Research crew assembly and execution."""

from __future__ import annotations

import logging

from arastirma_ussu.config import CrewConfig, OllamaConfig

logger = logging.getLogger(__name__)

_ccfg = CrewConfig()
_ocfg = OllamaConfig()


def _create_crew_llm(crew_cfg: CrewConfig = _ccfg, ollama_cfg: OllamaConfig = _ocfg):
    """Create the CrewAI LLM pointing at Ollama."""
    from crewai import LLM

    return LLM(
        model=f"ollama/{ollama_cfg.model}",
        base_url=ollama_cfg.base_url,
        temperature=crew_cfg.temperature,
        timeout=crew_cfg.llm_request_timeout,
        # num_ctx is set in the Ollama Modelfile (8192), not here —
        # CrewAI's litellm passes unknown kwargs to OpenAI-compat API
        # which rejects them (Completions.create() TypeError).
    )


def build_research_crew(query: str):
    """Build a sequential research crew for the given query."""
    from crewai import Crew, Process

    from arastirma_ussu.crew.agents import build_agents
    from arastirma_ussu.crew.tasks import build_tasks

    llm = _create_crew_llm()
    agents = build_agents(llm)
    tasks = build_tasks(agents, query)

    return Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=_ccfg.verbose,
    )


def run_crew(query: str) -> str:
    """Build crew, kick off, return result string."""
    crew = build_research_crew(query)
    result = crew.kickoff()
    return result.raw if hasattr(result, "raw") else str(result)
