"""CrewAI agent definitions — tool-less, sequential analysis + writing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arastirma_ussu.config import CrewConfig

if TYPE_CHECKING:
    from crewai import Agent

_cfg = CrewConfig()


def build_agents(llm) -> dict[str, Agent]:
    """Build the research crew agents.

    Both agents are tool-less and delegation-free: all data gathering
    happens in the outer LangGraph agent. CrewAI agents focus on
    analysis and synthesis only.
    """
    from crewai import Agent

    analyst = Agent(
        role="Senior Research Analyst",
        goal=(
            "Analyze the provided research data thoroughly. "
            "Identify key themes, evaluate source quality, "
            "and extract the most important findings."
        ),
        backstory=(
            "You are an experienced research analyst who excels at "
            "synthesizing information from multiple sources into "
            "clear, structured analyses."
        ),
        llm=llm,
        tools=[],
        allow_delegation=False,
        verbose=_cfg.verbose,
        max_iter=_cfg.max_agent_iter,
    )

    writer = Agent(
        role="Research Report Writer",
        goal=(
            "Transform the analyst's findings into a clear, "
            "well-structured research report written in Turkish."
        ),
        backstory=(
            "You are a skilled technical writer who creates "
            "accessible reports from complex analyses. "
            "You always write in Turkish (Turkce)."
        ),
        llm=llm,
        tools=[],
        allow_delegation=False,
        verbose=_cfg.verbose,
        max_iter=_cfg.max_agent_iter,
    )

    return {"analyst": analyst, "writer": writer}
