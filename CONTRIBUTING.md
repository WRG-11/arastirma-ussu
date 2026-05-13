# Contributing

`arastirma-ussu` is a single-maintainer research project, but it follows
the same hygiene as a multi-developer repo so that future contributors
(or future-you) can work with confidence.

## Development setup

```bash
git clone https://github.com/WRG-11/arastirma-ussu.git
cd arastirma-ussu
pip install -e ".[dev]"
```

Python 3.11 or 3.12 is supported (matrix in `.github/workflows/ci.yml`).
Layer-2 web search and Layer-5 orchestrator paths require a running
Ollama instance for the `integration` and `experimental` test markers.

## Tests

```bash
# fast suite (default — what CI runs)
python -m pytest -m "not integration and not experimental" --tb=short -q

# full suite (requires Ollama)
python -m pytest -q

# coverage gate (matches CI threshold)
python -m pytest --cov=arastirma_ussu --cov-report=term-missing \
                 --cov-fail-under=70 \
                 -m "not integration and not experimental" -q
```

Test markers are defined in `pyproject.toml`:

- `integration` — needs Ollama running, slower.
- `experimental` — Layer 5.5 LLM-as-judge (research-stage).

## Lint

```bash
ruff check src/ tests/
```

Fix automatically with `ruff check --fix src/ tests/`. Style follows
`ruff`'s defaults plus any project-specific rules in `pyproject.toml`.

## Branch naming

```
<type>/<short-description>-YYYY-MM-DD
```

Examples:

- `feat/layer3-pdf-chunker-2026-05-10`
- `fix/memory-dedupe-2026-05-10`
- `chore/dependabot-bumps-2026-05-10`

Types we use: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`,
`perf`, `ci`, `build`.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary in imperative mood>

<optional body — what + why, not how>
```

Examples:

- `feat(layer4): add hybrid BM25 + dense retrieval scorer`
- `fix(memory): de-duplicate identical embeddings on insert`
- `chore(deps): bump openai-style chat client to 1.18`

**Do not** add the following trailers (anti-ban discipline):

- `Co-Authored-By: Claude ...`
- `🤖 Generated with [Claude Code]`
- Any other AI-tool watermark in commit messages or PR descriptions.

## Pull requests

- Open against `main`.
- Keep PRs focused; multi-layer changes should be separate PRs unless
  they share a single rationale.
- CI must be green (`test` + `lint` jobs) before merge.
- Maintainer self-reviews using `--comment` (single-developer repo).

## Memory and context

This repo runs an AI research agent that consults persistent memory
(see `CLAUDE.md` for the memory directory). When contributing changes
that affect agent behaviour or output, update or invalidate the
relevant memory entries — stale entries are archived with
`archived: true` rather than deleted.

## Reporting security issues

Do **not** file public issues for security problems. See [SECURITY.md](SECURITY.md)
for the private vulnerability reporting flow.
