# Arastirma Ussu

## Proje Amaci
Kisisel AI arastirma asistani. 5 katmanli mimari. Ollama local LLM.
Guvenlik disi, yaratici portfolyo parcasi. Tek gelistirici, kisisel proje.

## Memory & Context Discipline
- Memory dizini: `<user-home>/.claude/projects/<project-memory-dir>/memory/`
- Diger proje cross-pollination: OTOMATIK SYNC YOK. Manuel kopyalama.
- Kod-derive edilebilen bilgileri memory'ye YAZMA (dosya yapisi, fonksiyon imzalari vb.)
- Archive pattern: stale memory'ye `archived: true` frontmatter ekle, silme

## Token Discipline
- RTK prefix ZORUNLU: `rtk pip install`, `rtk docker logs`, `rtk git status` vb.
- Buyuk dosya okuma: Read tool ile offset/limit kullan
- Genis arama: Explore subagent'a delege et, ana context'i kirletme

## Active Constraints
- Dil: Kullaniciya Turkce / Kod+degisken Ingilizce
- Git: `git add .` YASAK — dosya ismiyle stage et
- Multi-session: branch izolasyonu (session/<kisa-tanim>)
- VRAM: RTX 4070 8GB. qwen2.5:7B tek model (~4.5GB). Ikinci model YUKLEME.
- Venv: .venv DAIMA aktif. Global pip'e DOKUNMA.
- CI VAR (`.github/workflows/ci.yml` — ubuntu+windows × py3.11/3.12 matrix
  + lint job), coverage gate 70% (R12 audit 2026-05-13 hardened: layer2-5
  extras install + PYTHONUTF8=1 + fail-fast:false). pre-commit yok.

## Teknoloji Stack
- Python 3.12, venv .venv/
- LangGraph (orkestrasyon) + LangChain (thin adapter, sadece model/tool protocol)
- Ollama: qwen2.5:7B (tek model, cok dilli, Turkce guclu; 2026-05-04 3B→7B upgrade)
- Layer 2: LlamaIndex (veri okuma/indeksleme)
- Layer 3: Qdrant (vektor DB, ChromaDB yerine)
- Layer 4: CrewAI (multi-agent, LangGraph tool olarak cagirilir)
- Layer 5: Deterministik kalite metrikleri + guard pipeline (sister project fork)
- Layer 5.5: RAGAS LLM-as-judge (opsiyonel, deneysel, kullaniciya gosterilmez)
- UI: Gradio chat arayuzu (app.py, port 7861)

## Katman Durumu
- [x] Layer 0: Bootstrap (toolcall sanity: FAIL → manuel ReAct)
- [x] Layer 1: Basic Agent (LangGraph + Ollama) — ReAct regex parser, 23 smoke + 3 integration
- [x] Layer 2: Data Connection (LlamaIndex) — doc_search, MiniLM CPU embed, 13 test
- [x] Layer 3: Smart Memory (Qdrant) — 2 koleksiyon, hybrid embed, LRU 5k, 20 test
- [x] Layer 4: Multi-Agent (CrewAI) — tool-less sequential, allow_delegation=False, 17 test
- [x] Layer 5: Security & Deterministic Quality — 7 guard, action whitelist, 41 test
- [x] E2E: REPL canlı test (2026-05-03) — RAG, hafıza zinciri, guard pipeline calisiyor
- [x] Coverage ratchet (2026-05-04) — 70% gate, `4590294`
- [x] CI workflow (2026-05-05) — GHA ubuntu+windows × py3.11/3.12 + lint, `0d4ef6b`
- [x] R12 audit hygiene (2026-05-13) — SECURITY/CoC/CONTRIBUTING + ci.yml v6 SHA-pin + full-coverage hardening, `9b8c3f0` + `49c9e4a`
- [x] R13 README sync (2026-05-13) — 3b→7b drift fix + badges + clone instruction, `38539c1`
- [x] Layer 5.5 skeleton + real wire-up (2026-05-14) — `src/arastirma_ussu/eval/` (types + ragas_judge wrapper + `default_ollama_judge()` helper); 12 test total (10 contract + 2 Ollama-judge golden gated behind `experimental + integration`); RAGAS 0.4.3 legacy `ragas.metrics` path (modern `collections` API needs InstructorLLM which lacks Ollama support); `make layer55`
- [x] R14 audit (2026-05-14) — ddgs migration + coverage 70%→72% + `ingest/embed.py` sentence-transformers API fix (`get_embedding_dimension`→`get_sentence_embedding_dimension`)

## E2E Test Sonuclari (2026-05-03)
- 111/111 smoke test PASSED
- REPL: soru→doc_search→Turkce cevap calisiyor
- RAG: belge indeksle→Qdrant→doc_search sorgusu calisiyor
- Hafiza: memory_search onceki Q&A cifleri buluyor
- Guard: check_language Ingilizce drift'i yakaliyor, _retry_turkish ceviriye gonderiyor
- Encoding: _ensure_utf8_stdio Windows cp1254→UTF-8 zorluyor

## Bilinen Sinirlamalar
- Model evrimi: dolphin-mistral:7B (Turkce zayif) → qwen2.5:3B (2026-05-03) → qwen2.5:7B (2026-05-04, kalite icin); prompt+guard+retry 3 katmanli savunma var ama %100 degil
- Ilk tur guard: LLM tool cagirmadan Final Answer verirse doc_search/memory_search'e zorlanir
- pyarrow import sirasinda Windows access violation uyarisi — test sonucunu etkilemiyor
- CI quota siniri (org Actions): periyot reset bekleniyor; bu tarihe kadar local-verified + merge disiplini

## Sik Komutlar
```bash
source .venv/Scripts/activate
pip install -e ".[dev]"
pip install -e ".[layer2,dev]"
make sanity                          # Layer 0 environment + toolcall check
make smoke                           # Tum katman smoke testleri
make layer1                          # Layer 1 smoke
make lint                            # ruff check
make qdrant-up / make qdrant-down    # Layer 3 Qdrant Docker
make clean                           # __pycache__ vb. temizlik
python app.py                        # Gradio chat UI (http://127.0.0.1:7861)
```
