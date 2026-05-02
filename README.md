# Arastirma Ussu

Kisisel AI arastirma asistani. 5 katmanli mimari, tamamen lokal (Ollama).

## Katmanlar

| # | Katman | Teknoloji | Durum |
|---|--------|-----------|-------|
| 0 | Bootstrap | pyproject, venv, CLAUDE.md | - |
| 1 | Basic Agent | LangGraph + Ollama | - |
| 2 | Data Connection | LlamaIndex | - |
| 3 | Smart Memory | Qdrant | - |
| 4 | Multi-Agent | CrewAI | - |
| 5 | Quality & Security | Deterministik metrikler + Guards | - |

## Kurulum

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows + Git Bash
pip install -e ".[dev]"
make sanity
```
