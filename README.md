# Arastirma Ussu

> Kisisel AI arastirma asistani. 5-katmanli, lokal-first (Ollama), production-RAG patterns.

`arastirma-ussu`, kucuk olcekli arastirmacilarin gunluk soru-cevap, dokuman tarama
ve uzun donemli not-tutma akislarini tek bir yerden yonetebilmesi icin tasarlanmis
acik kaynakli bir AI asistanidir. Tasarim hedefi: **buyuk SaaS RAG yiginlarinin
3-katman guard + semantik cache + golden eval pattern'lerini solo bir kullanici
icin tek-makine olceginde yeniden uretmek**.

## Neden?

| Problem | Cozumumuz |
|--------|-----------|
| Ticari RAG araclari pahali + veri-onleyici | Tum LLM cagrilari Ollama uzerinden lokal |
| Tek-katman LLM yanitlari guvensiz | 3-katman guard (input + retrieval + output) |
| Cevaplar zamanla siniflanamiyor | Qdrant + persistent memory layer |
| Coklu kaynak takibi zor | LlamaIndex doc-store + DuckDuckGo web cache |

## 5-Katmanli Mimari

```
+------------------------------------------------------+
|  L5: Quality & Security Guards (deterministik + AI)  |
+------------------------------------------------------+
|  L4: Multi-Agent Orchestration (CrewAI)              |
+------------------------------------------------------+
|  L3: Smart Memory (Qdrant + fastembed)               |
+------------------------------------------------------+
|  L2: Data Connection (LlamaIndex doc + web)          |
+------------------------------------------------------+
|  L1: Basic Agent (LangGraph + Ollama)                |
+------------------------------------------------------+
|  L0: Bootstrap (pyproject, venv, CLAUDE.md)          |
+------------------------------------------------------+
```

| # | Katman | Teknoloji | Durum |
|---|--------|-----------|-------|
| 0 | Bootstrap | pyproject, venv, CLAUDE.md | ✓ |
| 1 | Basic Agent | LangGraph + langchain-ollama | WIP |
| 2 | Data Connection | LlamaIndex + DuckDuckGo | planned |
| 3 | Smart Memory | Qdrant + fastembed | planned |
| 4 | Multi-Agent | CrewAI | planned |
| 5 | Quality & Security | Deterministik metrikler + Guards | planned |

Her katman bagimsiz olarak `pip install -e ".[layerN]"` extra'si ile aktive edilir;
`layer1` minimal calisma icin yeterlidir. Detay icin: `pyproject.toml`.

## Kurulum

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows + Git Bash
# veya: .venv\Scripts\Activate.ps1  (PowerShell)
pip install -e ".[dev]"
make sanity
```

### Ollama gereksinimi

L1 calismasi icin lokal Ollama gerek:

```bash
ollama serve &
ollama pull llama3.1:8b   # veya tercih ettiginiz model
```

`OLLAMA_HOST` env var ile harici instance da kullanilabilir.

## Hizli Kullanim

```bash
python app.py            # CLI moduna girer
# veya: make run
```

## Tasarim Prensipleri

- **Lokal-first**: Tum LLM ve embedding cagrilari default olarak lokal (Ollama,
  fastembed). Web aramasi opt-in.
- **Katman-zorunlu degil**: L2/L3/L4 olmadan da L1 calisir.
- **Determinik testler**: Her katmanin ayri pytest hedefi var; LLM cagrilari
  fixture'lara cevrilebilir.
- **Type-strict**: Tum public surface `mypy --strict` ile geciyor.

## Yapi

```
src/                     # Paket kodu (per-layer alt modul)
tests/                   # pytest (unit + integration)
scripts/                 # CLI yardimci script'leri
app.py                   # Giris noktasi
pyproject.toml           # Bagimliliklar + opsiyonel extras
Makefile                 # Sanity / test / lint hedefleri
CLAUDE.md                # Claude Code icin proje rehberi
```

## Roadmap

- [x] **L0 Bootstrap** — pyproject, venv, mypy, pre-commit
- [ ] **L1 Basic Agent** — LangGraph state machine + tool-call loop
- [ ] **L2 Data Connection** — LlamaIndex doc-store + DuckDuckGo wrapper
- [ ] **L3 Smart Memory** — Qdrant collection + retention politikasi
- [ ] **L4 Multi-Agent** — CrewAI ile rol-based arastirma akisi
- [ ] **L5 Quality & Security** — 3-layer guard + golden eval set

## Lisans

MIT — bkz. [LICENSE](LICENSE).
