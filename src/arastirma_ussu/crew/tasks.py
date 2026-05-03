"""CrewAI task definitions for the research crew."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crewai import Agent, Task


def build_tasks(agents: dict[str, Agent], query: str) -> list[Task]:
    """Build sequential tasks: analysis → report.

    ``context=[analysis_task]`` on report_task ensures the writer
    receives the analyst's output. Process.sequential alone only
    controls execution order, not data passing.
    """
    from crewai import Task

    analysis_task = Task(
        description=(
            f"Analyze the following research data and question:\n\n"
            f"{query}\n\n"
            "Identify key themes, evaluate the quality of information, "
            "and list the most important findings in a structured format."
        ),
        expected_output=(
            "A structured analysis with: "
            "1) Key findings (bullet points), "
            "2) Source quality assessment, "
            "3) Information gaps identified."
        ),
        agent=agents["analyst"],
    )

    report_task = Task(
        description=(
            "Using the analyst's findings, write a clear and concise "
            "research report in Turkish (Turkce). The report should be "
            "accessible to a general audience while maintaining accuracy."
        ),
        expected_output=(
            "A well-structured Turkish research report with: "
            "1) Summary (ozet), "
            "2) Key findings (temel bulgular), "
            "3) Conclusion (sonuc)."
        ),
        agent=agents["writer"],
        context=[analysis_task],
    )

    return [analysis_task, report_task]
