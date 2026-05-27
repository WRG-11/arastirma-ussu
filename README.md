# Araştırma Üssü (Research Base)

[![CI](https://github.com/WRG-11/arastirma-ussu/actions/workflows/ci.yml/badge.svg)](https://github.com/WRG-11/arastirma-ussu/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-72%25-brightgreen)](#)

> Personal AI research assistant. Full 5-layer stack, local-first
> (Ollama + Qdrant), production-RAG patterns. E2E live (2026-05-03).
>
> **Turkish-language by design** — the assistant answers in Turkish, ships
> with Turkish drift detection and a Turkish UI. The repo's English
> documentation is the *discovery surface*; the *product surface* speaks
> Turkish on purpose.

`arastirma-ussu` is an open-source AI assistant designed to let small-scale
researchers run their daily Q&A, document scanning and long-term
note-taking workflows from one place. Design goal: **reproduce the
3-layer-guard + semantic-cache + golden-eval patterns of large SaaS RAG
stacks at single-machine, solo-user scale**.

## Why?

| Problem | Our answer |
|---------|------------|
| Commercial RAG tools are expensive + data-extractive | All LLM calls go through local Ollama |
| Single-layer LLM answers are unsafe | 3-layer guard (input + retrieval + output) |
| Answers drift out of category over time | Qdrant + persistent memory layer |
| Tracking multiple sources is hard | LlamaIndex doc-store + DuckDuckGo web cache |

## 5-Layer Architecture (all layers implemented)

```
+------------------------------------------------------+
|  L5: Quality & Security Guards (7 guards, 41 tests)  |
+------------------------------------------------------+
|  L4: Multi-Agent (CrewAI sequential, 17 tests)       |
+------------------------------------------------------+
|  L3: Smart Memory (Qdrant 2 collections, 20 tests)   |
+------------------------------------------------------+
|  L2: Data Connection (LlamaIndex + MiniLM, 13 tests) |
+------------------------------------------------------+
|  L1: Basic Agent (LangGraph + Ollama, 26 tests)      |
+------------------------------------------------------+
|  L0: Bootstrap (toolcall sanity, manual ReAct fb)    |
+------------------------------------------------------+
```

| # | Layer | Technology | Status |
|---|-------|------------|--------|
| 0 | Bootstrap | toolcall sanity check, manual ReAct fallback | ✓ |
| 1 | Basic Agent | LangGraph + langchain-ollama, ReAct regex parser | ✓ (23 smoke + 3 integration) |
| 2 | Data Connection | LlamaIndex + MiniLM CPU embed | ✓ (13 tests) |
| 3 | Smart Memory | Qdrant 2 collections, hybrid embed, LRU 5k | ✓ (20 tests) |
| 4 | Multi-Agent | CrewAI tool-less sequential, allow_delegation=False | ✓ (17 tests) |
| 5 | Quality & Security | 7 guards + action whitelist | ✓ (41 tests) |
| 5.5 | RAGAS LLM-as-Judge | optional / experimental | ✓ end-to-end (7 contract + 2 Ollama-judge golden) |

Each layer is activated via the `pip install -e ".[layerN]"` extras; L1 is
the minimum required to run anything.

## E2E Status (2026-05-03 live test)

- **111/111 smoke tests PASSED**.
- REPL: question → `doc_search` → Turkish answer works end-to-end.
- RAG: index a document → Qdrant → `doc_search` query works.
- Memory: `memory_search` retrieves prior Q&A pairs.
- Guard: `check_language` catches English drift; `_retry_turkish` routes
  to translation.
- Encoding: `_ensure_utf8_stdio` forces Windows cp1254 → UTF-8.

## Installation

```bash
git clone https://github.com/WRG-11/arastirma-ussu.git
cd arastirma-ussu

python -m venv .venv
source .venv/Scripts/activate   # Windows + Git Bash
# or: .venv\Scripts\Activate.ps1  (PowerShell)

# Minimum (L0 + L1):
pip install -e ".[dev]"

# L2 and beyond via extras:
pip install -e ".[layer2,layer3,layer4,layer5,dev]"
```

### Ollama requirement

L1 needs a local Ollama instance:

```bash
ollama serve &
ollama pull qwen2.5:7b   # single model, multilingual, strong Turkish, ~4.5GB VRAM
```

Single-model discipline (RTX 4070 8GB VRAM): no second model load. An
external instance can be used via the `OLLAMA_HOST` env var.

### Layer 3 — Qdrant Docker

```bash
make qdrant-up    # localhost:6333
make qdrant-down  # cleanup
```

## Quick Use

```bash
make sanity      # L0 environment + toolcall sanity check
make smoke       # smoke test across all layers
make layer1      # L1 only (LangGraph + Ollama)
make layer2      # L2 only (LlamaIndex doc_search)
make layer3      # L3 only (Qdrant memory)
make layer4      # L4 only (CrewAI multi-agent)
make layer5      # L5 only (guard pipeline)
make layer55     # L5.5 RAGAS skeleton contract tests (experimental)

python app.py    # Gradio chat UI (http://127.0.0.1:7861)
```

Example prompt (Turkish is the product language): `Türkiye'nin başkenti
neresi?` — the assistant responds in Turkish, drift guard verifies it
stayed in Turkish, retry path is exercised if it did not.

## Design Principles

- **Local-first**: All LLM and embedding calls are local by default
  (Ollama, fastembed, MiniLM CPU). Web search is opt-in.
- **Layers are optional**: L1 runs without L2/L3/L4.
- **Single-model discipline**: VRAM constraint (8GB) → qwen2.5:7B single
  model (~4.5GB).
- **Deterministic tests**: every layer has its own pytest target; LLM
  calls can be swapped for fixtures.
- **Quiet by default, verbose on opt-in**: verbose/debug/json are opt-in
  flags.
- **Turkish product, English docs**: the assistant's user-facing surface
  (prompts, UI, drift detection) stays in Turkish; the discovery surface
  (README, code comments, config) is English for OSS reach.

## Layout

```
src/arastirma_ussu/   # Package code (one submodule per layer)
tests/                # pytest (smoke + integration + experimental marker)
scripts/              # toolcall_sanity.py + helper scripts
data/                 # qdrant_storage/ + sample document / index files
app.py                # Gradio chat UI entry point
pyproject.toml        # Dependencies + optional layer extras
Makefile              # sanity / smoke / layer1-5 / lint / qdrant-up
CLAUDE.md             # Claude Code project guide (canonical status)
```

## Known Limitations

- Model evolution: `dolphin-mistral:7B` (weak Turkish) → `qwen2.5:3B`
  (2026-05-03) → `qwen2.5:7B` (2026-05-04, upgraded for quality);
  prompt + guard + retry give a 3-layer defence but not a 100% one.
- First-turn guard: if the LLM produces a Final Answer without calling a
  tool, it is forced into `doc_search` / `memory_search`.
- `pyarrow` raises a Windows access-violation warning on import —
  test outcomes are unaffected.

## Roadmap

- [x] **L0 Bootstrap** — toolcall sanity, manual ReAct fallback
- [x] **L1 Basic Agent** — LangGraph + Ollama, ReAct regex parser
- [x] **L2 Data Connection** — LlamaIndex + MiniLM
- [x] **L3 Smart Memory** — Qdrant 2 collections, hybrid embed
- [x] **L4 Multi-Agent** — CrewAI sequential
- [x] **L5 Quality & Security** — 7 guards + action whitelist
- [x] **E2E REPL live test** (2026-05-03)
- [x] **L5.5 RAGAS LLM-as-Judge** (2026-05-14) — skeleton + end-to-end
      wired with the `default_ollama_judge()` helper; 9 tests (7 contract
      + 2 Ollama-judge golden, gated behind `experimental + integration`);
      legacy `ragas.metrics` path (the modern `collections` API blocks
      Ollama until RAGAS 1.0 ships a local-LLM factory).

## License

MIT — see [LICENSE](LICENSE).
