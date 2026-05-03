# Arastirma Ussu

## Proje Amaci
Kisisel AI arastirma asistani. 5 katmanli mimari. Ollama local LLM.
Guvenlik disi, yaratici portfolyo parcasi. Tek gelistirici, kisisel proje.

## Memory & Context Discipline
- Memory dizini: `C:\Users\lenovo\.claude\projects\D--dev-arastirma-ussu\memory\`
- WRG / ai-security cross-pollination: OTOMATIK SYNC YOK. Manuel kopyalama.
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
- VRAM: RTX 4070 8GB. dolphin-mistral:7B tek model. Ikinci model YUKLEME.
- Venv: .venv DAIMA aktif. Global pip'e DOKUNMA.
- CI yok, pre-commit yok (kisisel proje, agirlik katma)

## Teknoloji Stack
- Python 3.12, venv .venv/
- LangGraph (orkestrasyon) + LangChain (thin adapter, sadece model/tool protocol)
- Ollama: dolphin-mistral:7B (tek model, Garak ile paylasilir)
- Layer 2: LlamaIndex (veri okuma/indeksleme)
- Layer 3: Qdrant (vektor DB, ChromaDB yerine)
- Layer 4: CrewAI (multi-agent, LangGraph tool olarak cagirilir)
- Layer 5: Deterministik kalite metrikleri + guard pipeline (ai-security'den fork)
- Layer 5.5: RAGAS LLM-as-judge (opsiyonel, deneysel, kullaniciya gosterilmez)

## Katman Durumu (guncelle!)
- [x] Layer 0: Bootstrap (toolcall sanity: FAIL → manuel ReAct)
- [x] Layer 1: Basic Agent (LangGraph + Ollama) — ReAct regex parser, 23 smoke + 3 integration
- [x] Layer 2: Data Connection (LlamaIndex) — doc_search, MiniLM CPU embed, 13 test
- [x] Layer 3: Smart Memory (Qdrant) — 2 koleksiyon, hybrid embed, LRU 5k, 20 test
- [x] Layer 4: Multi-Agent (CrewAI) — tool-less sequential, allow_delegation=False, 17 test
- [ ] Layer 5: Security & Deterministic Quality
- [ ] Layer 5.5: RAGAS LLM-as-Judge (opsiyonel/deneysel)

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
```
