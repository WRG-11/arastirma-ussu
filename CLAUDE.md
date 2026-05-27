# Arastirma Ussu (Research Base)

## Project Purpose
Personal AI research assistant. 5-layer architecture. Local Ollama LLM.
Non-security, creative portfolio piece. Solo developer, personal project.

The product itself is Turkish-language by design (the assistant answers in
Turkish, ships with Turkish drift detection and a Turkish UI). This file
and the rest of the developer surface are in English; the functional
Turkish surface is intentional and must be preserved.

## Memory & Context Discipline
- Memory directory: `<user-home>/.claude/projects/<project-memory-dir>/memory/`
- Cross-project pollination from other projects: NO AUTO-SYNC. Copy manually.
- Do NOT write code-derivable information (file layout, function signatures, etc.) into memory.
- Archive pattern: add `archived: true` frontmatter to stale memory entries; do not delete.

## Token Discipline
- RTK prefix REQUIRED: `rtk pip install`, `rtk docker logs`, `rtk git status`, etc.
- Reading large files: use the Read tool's offset/limit parameters.
- Wide-scope search: delegate to the Explore subagent — do not pollute the main context.

## Active Constraints
- Language: respond to the user in Turkish / keep code + variables in English.
- Git: `git add .` is FORBIDDEN — stage by file name.
- Multi-session: branch isolation (session/<short-tag>).
- VRAM: RTX 4070 8GB. qwen2.5:7B single model (~4.5GB). DO NOT load a second model.
- Venv: keep `.venv` active at all times. DO NOT touch the global pip.
- CI is wired (`.github/workflows/ci.yml` — ubuntu+windows × py3.11/3.12 matrix
  + lint job), coverage gate 70% (R12 audit 2026-05-13 hardened: layer2-5
  extras install + PYTHONUTF8=1 + fail-fast:false). No pre-commit.

## Technology Stack
- Python 3.12, venv `.venv/`
- LangGraph (orchestration) + LangChain (thin adapter, only model/tool protocol)
- Ollama: qwen2.5:7B (single model, multilingual, strong Turkish; 2026-05-04 3B→7B upgrade)
- Layer 2: LlamaIndex (data loading / indexing)
- Layer 3: Qdrant (vector DB, in place of ChromaDB)
- Layer 4: CrewAI (multi-agent, invoked as a LangGraph tool)
- Layer 5: Deterministic quality metrics + guard pipeline (sister-project fork)
- Layer 5.5: RAGAS LLM-as-judge (optional, experimental, not exposed to the user)
- UI: Gradio chat interface (`app.py`, port 7861)

## Layer Status
- [x] Layer 0: Bootstrap (toolcall sanity: FAIL → manual ReAct)
- [x] Layer 1: Basic Agent (LangGraph + Ollama) — ReAct regex parser, 23 smoke + 3 integration
- [x] Layer 2: Data Connection (LlamaIndex) — `doc_search`, MiniLM CPU embed, 13 tests
- [x] Layer 3: Smart Memory (Qdrant) — 2 collections, hybrid embed, LRU 5k, 20 tests
- [x] Layer 4: Multi-Agent (CrewAI) — tool-less sequential, allow_delegation=False, 17 tests
- [x] Layer 5: Security & Deterministic Quality — 7 guards, action whitelist, 41 tests
- [x] E2E: REPL live test (2026-05-03) — RAG, memory chain, guard pipeline all running
- [x] Coverage ratchet (2026-05-04) — 70% gate, `4590294`
- [x] CI workflow (2026-05-05) — GHA ubuntu+windows × py3.11/3.12 + lint, `0d4ef6b`
- [x] R12 audit hygiene (2026-05-13) — SECURITY/CoC/CONTRIBUTING + ci.yml v6 SHA-pin + full-coverage hardening, `9b8c3f0` + `49c9e4a`
- [x] R13 README sync (2026-05-13) — 3b→7b drift fix + badges + clone instruction, `38539c1`
- [x] Layer 5.5 skeleton + real wire-up (2026-05-14) — `src/arastirma_ussu/eval/` (types + ragas_judge wrapper + `default_ollama_judge()` helper); 12 tests total (10 contract + 2 Ollama-judge golden, gated behind `experimental + integration`); RAGAS 0.4.3 legacy `ragas.metrics` path (the modern `collections` API needs InstructorLLM which lacks Ollama support); `make layer55`
- [x] R14 audit (2026-05-14) — ddgs migration + coverage 70%→72% + `ingest/embed.py` sentence-transformers API fix (`get_embedding_dimension`→`get_sentence_embedding_dimension`)

## E2E Test Results (2026-05-03)
- 111/111 smoke tests PASSED
- REPL: question → `doc_search` → Turkish answer works
- RAG: index document → Qdrant → `doc_search` query works
- Memory: `memory_search` retrieves prior Q&A pairs
- Guard: `check_language` catches English drift; `_retry_turkish` routes to translation
- Encoding: `_ensure_utf8_stdio` forces Windows cp1254 → UTF-8

## Known Limitations
- Model evolution: dolphin-mistral:7B (weak Turkish) → qwen2.5:3B (2026-05-03) → qwen2.5:7B (2026-05-04, upgraded for quality); prompt + guard + retry give a 3-layer defence but not a 100% one
- First-turn guard: if the LLM produces a Final Answer without calling a tool, it is forced into `doc_search` / `memory_search`
- `pyarrow` import on Windows raises an access-violation warning — test outcomes are unaffected
- CI quota cap (org Actions): waiting for the period reset; until then use local-verified runs + merge discipline

## Common Commands
```bash
source .venv/Scripts/activate
pip install -e ".[dev]"
pip install -e ".[layer2,dev]"
make sanity                          # Layer 0 environment + toolcall check
make smoke                           # smoke tests across all layers
make layer1                          # Layer 1 smoke
make lint                            # ruff check
make qdrant-up / make qdrant-down    # Layer 3 Qdrant Docker
make clean                           # clean up __pycache__ etc.
python app.py                        # Gradio chat UI (http://127.0.0.1:7861)
```
