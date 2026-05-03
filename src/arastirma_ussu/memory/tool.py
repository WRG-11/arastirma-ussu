"""Conversation memory search tool for the ReAct agent."""

from __future__ import annotations


def memory_search(query: str) -> str:
    """Search conversation memory for relevant past Q&A pairs.

    Lazy-imports so smoke tests work without qdrant-client installed.
    """
    try:
        from arastirma_ussu.agent.tools import _truncate
        from arastirma_ussu.memory.store import get_memory

        memory = get_memory()
        results = memory.search(query)
        formatted = memory.format_results(results)
        return _truncate(formatted)
    except ImportError:
        return "Hafiza modulu yuklu degil. pip install -e '.[layer3]' calistirin."
    except Exception as e:
        return f"Hafiza arama hatasi: {e}"
