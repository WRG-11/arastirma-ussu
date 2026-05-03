"""Research crew tool for the ReAct agent."""

from __future__ import annotations


def crew_research(query: str) -> str:
    """Delegate a complex research question to the multi-agent crew.

    Lazy-imports so smoke tests work without crewai installed.
    Wrapped with timeout to prevent hung execution.
    """
    try:
        from arastirma_ussu.agent.tools import _truncate
        from arastirma_ussu.config import CrewConfig
        from arastirma_ussu.crew.crew import run_crew
        from arastirma_ussu.crew.timeout import run_with_timeout

        cfg = CrewConfig()
        result = run_with_timeout(run_crew, query, timeout=cfg.timeout)
        return _truncate(result)
    except ImportError:
        return "CrewAI modulu yuklu degil. pip install -e '.[layer4]' calistirin."
    except TimeoutError:
        return "Arastirma ekibi zaman asimina ugradi. Daha basit bir soru deneyin."
    except Exception as e:
        return f"Crew arastirma hatasi: {e}"
