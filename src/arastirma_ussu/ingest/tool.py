"""Document search tool for the ReAct agent."""

from __future__ import annotations


def doc_search(query: str) -> str:
    """Search indexed documents for relevant information.

    Lazy-imports LlamaIndex so smoke tests work without it installed.
    Applies truncation to keep output within context window budget.
    """
    try:
        from arastirma_ussu.agent.tools import _truncate
        from arastirma_ussu.ingest.index import query_index

        return _truncate(query_index(query))
    except ImportError:
        return "Belge arama modulu yuklu degil. pip install -e '.[layer2]' calistirin."
    except Exception as e:
        return f"Belge arama hatasi: {e}"
