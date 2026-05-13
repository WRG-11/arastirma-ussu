# Arastirma Ussu

[![CI](https://github.com/WRG-11/arastirma-ussu/actions/workflows/ci.yml/badge.svg)](https://github.com/WRG-11/arastirma-ussu/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-70%25-brightgreen)](#)

> Kisisel AI arastirma asistani. 5 katman tum stack hazir, lokal-first
> (Ollama + Qdrant), production-RAG patterns. E2E canli (2026-05-03).

`arastirma-ussu`, kucuk olcekli arastirmacilarin gunluk soru-cevap, dokuman
tarama ve uzun donemli not-tutma akislarini tek bir yerden yonetebilmesi
icin tasarlanmis acik kaynakli bir AI asistanidir. Tasarim hedefi: **buyuk
SaaS RAG yiginlarinin 3-katman guard + semantik cache + golden eval
pattern'lerini solo bir kullanici icin tek-makine olceginde yeniden
uretmek**.

## Neden?

| Problem | Cozumumuz |
|--------|-----------|
| Ticari RAG araclari pahali + veri-onleyici | Tum LLM cagrilari Ollama uzerinden lokal |
| Tek-katman LLM yanitlari guvensiz | 3-katman guard (input + retrieval + output) |
| Cevaplar zamanla siniflanamiyor | Qdrant + persistent memory layer |
| Coklu kaynak takibi zor | LlamaIndex doc-store + DuckDuckGo web cache |

## 5-Katmanli Mimari (tamami uygulanmis)

```
+------------------------------------------------------+
|  L5: Quality & Security Guards (7 guard, 41 test)    |
+------------------------------------------------------+
|  L4: Multi-Agent (CrewAI sequential, 17 test)        |
+------------------------------------------------------+
|  L3: Smart Memory (Qdrant 2 koleksiyon, 20 test)     |
+------------------------------------------------------+
|  L2: Data Connection (LlamaIndex + MiniLM, 13 test)  |
+------------------------------------------------------+
|  L1: Basic Agent (LangGraph + Ollama, 26 test)       |
+------------------------------------------------------+
|  L0: Bootstrap (toolcall sanity, manuel ReAct fb)    |
+------------------------------------------------------+
```

| # | Katman | Teknoloji | Durum |
|---|--------|-----------|-------|
| 0 | Bootstrap | toolcall sanity check, manuel ReAct fallback | ✓ |
| 1 | Basic Agent | LangGraph + langchain-ollama, ReAct regex parser | ✓ (23 smoke + 3 integration) |
| 2 | Data Connection | LlamaIndex + MiniLM CPU embed | ✓ (13 test) |
| 3 | Smart Memory | Qdrant 2 koleksiyon, hybrid embed, LRU 5k | ✓ (20 test) |
| 4 | Multi-Agent | CrewAI tool-less sequential, allow_delegation=False | ✓ (17 test) |
| 5 | Quality & Security | 7 guard + action whitelist | ✓ (41 test) |
| 5.5 | RAGAS LLM-as-Judge | opsiyonel/deneysel | planned |

Her katman `pip install -e ".[layerN]"` extras'i ile aktive edilir; L1
calismasi minimum gereksinim.

## E2E Durum (2026-05-03 canli test)

- **111/111 smoke test PASSED**
- REPL: soru → `doc_search` → Turkce cevap calisiyor
- RAG: belge indeksle → Qdrant → `doc_search` sorgusu calisiyor
- Hafiza: `memory_search` onceki Q&A ciftlerini buluyor
- Guard: `check_language` Ingilizce drift'i yakaliyor, `_retry_turkish`
  ceviriye gonderiyor
- Encoding: `_ensure_utf8_stdio` Windows cp1254 → UTF-8 zorluyor

## Kurulum

```bash
git clone https://github.com/WRG-11/arastirma-ussu.git
cd arastirma-ussu

python -m venv .venv
source .venv/Scripts/activate   # Windows + Git Bash
# veya: .venv\Scripts\Activate.ps1  (PowerShell)

# Minimum (L0 + L1):
pip install -e ".[dev]"

# L2 ve sonrasi icin extras:
pip install -e ".[layer2,layer3,layer4,layer5,dev]"
```

### Ollama gereksinimi

L1 calismasi icin lokal Ollama gerek:

```bash
ollama serve &
ollama pull qwen2.5:7b   # tek model, cok dilli, Turkce guclu, ~4.5GB VRAM
```

Tek model disiplini (RTX 4070 8GB VRAM): ikinci model yukleme yok.
`OLLAMA_HOST` env var ile harici instance da kullanilabilir.

### Layer 3 — Qdrant Docker

```bash
make qdrant-up    # localhost:6333
make qdrant-down  # temizlik
```

## Hizli Kullanim

```bash
make sanity      # L0 environment + toolcall sanity check
make smoke       # tum katmanlar smoke test
make layer1      # L1 sadece (LangGraph + Ollama)
make layer2      # L2 sadece (LlamaIndex doc_search)
make layer3      # L3 sadece (Qdrant memory)
make layer4      # L4 sadece (CrewAI multi-agent)
make layer5      # L5 sadece (guard pipeline)

python app.py    # Gradio chat UI (http://127.0.0.1:7861)
```

## Tasarim Prensipleri

- **Lokal-first**: Tum LLM ve embedding cagrilari default olarak lokal
  (Ollama, fastembed, MiniLM CPU). Web aramasi opt-in.
- **Katman-zorunlu degil**: L2/L3/L4 olmadan da L1 calisir.
- **Tek model disiplini**: VRAM kisitlari (8GB) → qwen2.5:7B tek model (~4.5GB).
- **Determinik testler**: Her katmanin ayri pytest hedefi var; LLM
  cagrilari fixture'lara cevrilebilir.
- **Sessiz varsayilan, konuskan opsiyon**: verbose/debug/json opt-in.

## Yapi

```
src/arastirma_ussu/   # Paket kodu (per-layer alt modul)
tests/                # pytest (smoke + integration + experimental marker)
scripts/              # toolcall_sanity.py + yardimci script'ler
data/                 # qdrant_storage/ + ornek dokuman/index dosyalari
app.py                # Gradio chat UI giris noktasi
pyproject.toml        # Bagimliliklar + opsiyonel layer extras
Makefile              # sanity / smoke / layer1-5 / lint / qdrant-up
CLAUDE.md             # Claude Code icin proje rehberi (canonical status)
```

## Bilinen Sinirlamalar

- Model evrimi: `dolphin-mistral:7B` (Turkce zayif) → `qwen2.5:3B`
  (2026-05-03) → `qwen2.5:7B` (2026-05-04, kalite icin upgrade); prompt
  + guard + retry 3 katmanli savunma var ama %100 degil.
- Ilk tur guard: LLM tool cagirmadan Final Answer verirse `doc_search` /
  `memory_search`'e zorlanir.
- `pyarrow` import sirasinda Windows access violation uyarisi —
  test sonucunu etkilemiyor.

## Yol Haritasi

- [x] **L0 Bootstrap** — toolcall sanity, manuel ReAct fallback
- [x] **L1 Basic Agent** — LangGraph + Ollama, ReAct regex parser
- [x] **L2 Data Connection** — LlamaIndex + MiniLM
- [x] **L3 Smart Memory** — Qdrant 2 koleksiyon, hybrid embed
- [x] **L4 Multi-Agent** — CrewAI sequential
- [x] **L5 Quality & Security** — 7 guard + action whitelist
- [x] **E2E REPL canli test** (2026-05-03)
- [ ] **L5.5 RAGAS LLM-as-Judge** — opsiyonel, deneysel

## Lisans

MIT — bkz. [LICENSE](LICENSE).
