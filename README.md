# Araştırma Üssü

[![CI](https://github.com/WRG-11/arastirma-ussu/actions/workflows/ci.yml/badge.svg)](https://github.com/WRG-11/arastirma-ussu/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-72%25-brightgreen)](#)

> Kişisel AI araştırma asistanı. 5 katman tüm stack hazır, lokal-first
> (Ollama + Qdrant), production-RAG patterns. E2E canlı (2026-05-03).

`arastirma-ussu`, küçük ölçekli araştırmacıların günlük soru-cevap, doküman
tarama ve uzun dönemli not-tutma akışlarını tek bir yerden yönetebilmesi
için tasarlanmış açık kaynaklı bir AI asistanıdır. Tasarım hedefi: **büyük
SaaS RAG yığınlarının 3-katmanlı guard + semantik cache + golden eval
pattern'lerini solo bir kullanıcı için tek-makine ölçeğinde yeniden
üretmek**.

## Neden?

| Problem | Çözümümüz |
|--------|-----------|
| Ticari RAG araçları pahalı + veri-önleyici | Tüm LLM çağrıları Ollama üzerinden lokal |
| Tek-katmanlı LLM yanıtları güvensiz | 3-katmanlı guard (input + retrieval + output) |
| Cevaplar zamanla sınıflanamıyor | Qdrant + persistent memory layer |
| Çoklu kaynak takibi zor | LlamaIndex doc-store + DuckDuckGo web cache |

## 5-Katmanlı Mimari (tamamı uygulanmış)

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
| 5.5 | RAGAS LLM-as-Judge | opsiyonel/deneysel | ✓ end-to-end (7 contract + 2 Ollama-judge golden) |

Her katman `pip install -e ".[layerN]"` extras'ı ile aktive edilir; L1
çalışması minimum gereksinim.

## E2E Durum (2026-05-03 canlı test)

- **111/111 smoke test PASSED**.
- REPL: soru → `doc_search` → Türkçe cevap çalışıyor.
- RAG: belge indeksle → Qdrant → `doc_search` sorgusu çalışıyor.
- Hafıza: `memory_search` önceki Q&A çiftlerini buluyor.
- Guard: `check_language` İngilizce drift'i yakalıyor, `_retry_turkish`
  çeviriye gönderiyor.
- Encoding: `_ensure_utf8_stdio` Windows cp1254 → UTF-8 zorluyor.

## Kurulum

```bash
git clone https://github.com/WRG-11/arastirma-ussu.git
cd arastirma-ussu

python -m venv .venv
source .venv/Scripts/activate   # Windows + Git Bash
# veya: .venv\Scripts\Activate.ps1  (PowerShell)

# Minimum (L0 + L1):
pip install -e ".[dev]"

# L2 ve sonrası için extras:
pip install -e ".[layer2,layer3,layer4,layer5,dev]"
```

### Ollama gereksinimi

L1 çalışması için lokal Ollama gerek:

```bash
ollama serve &
ollama pull qwen2.5:7b   # tek model, çok dilli, Türkçe güçlü, ~4.5GB VRAM
```

Tek model disiplini (RTX 4070 8GB VRAM): ikinci model yükleme yok.
`OLLAMA_HOST` env var ile harici instance da kullanılabilir.

### Layer 3 — Qdrant Docker

```bash
make qdrant-up    # localhost:6333
make qdrant-down  # temizlik
```

## Hızlı Kullanım

```bash
make sanity      # L0 environment + toolcall sanity check
make smoke       # tüm katmanlar smoke test
make layer1      # L1 sadece (LangGraph + Ollama)
make layer2      # L2 sadece (LlamaIndex doc_search)
make layer3      # L3 sadece (Qdrant memory)
make layer4      # L4 sadece (CrewAI multi-agent)
make layer5      # L5 sadece (guard pipeline)
make layer55     # L5.5 RAGAS skeleton contract tests (deneysel)

python app.py    # Gradio chat UI (http://127.0.0.1:7861)
```

## Tasarım Prensipleri

- **Lokal-first**: Tüm LLM ve embedding çağrıları default olarak lokal
  (Ollama, fastembed, MiniLM CPU). Web araması opt-in.
- **Katman-zorunlu değil**: L2/L3/L4 olmadan da L1 çalışır.
- **Tek model disiplini**: VRAM kısıtları (8GB) → qwen2.5:7B tek model (~4.5GB).
- **Deterministik testler**: Her katmanın ayrı pytest hedefi var; LLM
  çağrıları fixture'lara çevrilebilir.
- **Sessiz varsayılan, konuşkan opsiyon**: verbose/debug/json opt-in.

## Yapı

```
src/arastirma_ussu/   # Paket kodu (per-layer alt modül)
tests/                # pytest (smoke + integration + experimental marker)
scripts/              # toolcall_sanity.py + yardımcı script'ler
data/                 # qdrant_storage/ + örnek doküman/index dosyaları
app.py                # Gradio chat UI giriş noktası
pyproject.toml        # Bağımlılıklar + opsiyonel layer extras
Makefile              # sanity / smoke / layer1-5 / lint / qdrant-up
CLAUDE.md             # Claude Code için proje rehberi (canonical status)
```

## Bilinen Sınırlamalar

- Model evrimi: `dolphin-mistral:7B` (Türkçe zayıf) → `qwen2.5:3B`
  (2026-05-03) → `qwen2.5:7B` (2026-05-04, kalite için upgrade);
  prompt + guard + retry 3-katmanlı savunma var ama %100 değil.
- İlk tur guard: LLM tool çağırmadan Final Answer verirse `doc_search` /
  `memory_search`'e zorlanır.
- `pyarrow` import sırasında Windows access violation uyarısı —
  test sonucunu etkilemiyor.

## Yol Haritası

- [x] **L0 Bootstrap** — toolcall sanity, manuel ReAct fallback
- [x] **L1 Basic Agent** — LangGraph + Ollama, ReAct regex parser
- [x] **L2 Data Connection** — LlamaIndex + MiniLM
- [x] **L3 Smart Memory** — Qdrant 2 koleksiyon, hybrid embed
- [x] **L4 Multi-Agent** — CrewAI sequential
- [x] **L5 Quality & Security** — 7 guard + action whitelist
- [x] **E2E REPL canlı test** (2026-05-03)
- [x] **L5.5 RAGAS LLM-as-Judge** (2026-05-14) — skeleton + end-to-end
      wired with `default_ollama_judge()` helper; 9 test (7 contract +
      2 Ollama-judge golden gated behind `experimental + integration`);
      legacy `ragas.metrics` path (modern `collections` API blocks Ollama
      until RAGAS 1.0 brings local-LLM factory).

## Lisans

MIT — bkz. [LICENSE](LICENSE).
