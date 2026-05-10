# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest commit on `main` | Yes |
| Older commits | No |

This is a research-stage AI agent and has no tagged releases yet; the
moving `main` branch is the only supported surface.

## Reporting a Vulnerability

If you discover a security vulnerability in `arastirma-ussu` (the
research agent itself, its CI / publish workflows, or any bundled
script), please report it responsibly.

**Do not open a public issue.**

Instead, please use GitHub's
[private vulnerability reporting](https://github.com/WRG-11/arastirma-ussu/security/advisories/new)
or email the maintainer directly at **yakuphan.yucel11@gmail.com**.

### What to include

- Description of the vulnerability
- Steps to reproduce (input prompt, retrieval target, memory state, etc.)
- Affected layer (see Scope below) and component
- Potential impact (data exposure, agent hijack, exfiltration, …)
- Suggested fix or mitigation, if any

### Response timeline

- **Acknowledgment:** within 48 hours
- **Initial assessment:** within 1 week
- **Fix (if confirmed):** as soon as practical, typically within 2 weeks
- **Public disclosure:** coordinated; default 90 days from acknowledgment

## Scope

This policy covers the five-layer research agent and the threat surfaces
specific to each layer:

- **Layer 1 — Memory.** Persistent agent memory store. Threats include
  *memory poisoning* (an attacker inserts entries that bias future
  responses) and *cross-conversation leakage* (one user's data surfacing
  in another's session).
- **Layer 2 — Web search.** External web/HTTP retrieval. Threats include
  *SSRF* (search target pointing at internal hosts), *data exfiltration*
  via crafted query parameters, and *result injection* (search snippets
  containing prompt-injection payloads).
- **Layer 3 — Document analysis.** Local document parsing and chunking.
  Threats include *malformed-input crashes* (parser DoS) and
  *prompt-injection embedded inside seemingly benign documents* (PDFs,
  Office files, transcripts).
- **Layer 4 — RAG retrieval.** Vector / keyword index over indexed
  corpus. Threats include *retrieval poisoning* (attacker-controlled
  documents indexed and surfaced as authoritative context) and
  *retrieval-based prompt injection* (chunk content overrides the
  system prompt).
- **Layer 5 — Orchestrator.** The top-level reasoning + tool-use loop.
  Threats include *multi-step prompt injection* (later turns override
  earlier guards), *tool misuse* (LLM invokes tools with attacker
  parameters), and *secrets leakage* in tool I/O.

The CI workflow (`.github/workflows/ci.yml`) and any `scripts/` helpers
are also in scope — report supply-chain or workflow-permission issues
through the same channel.

Out of scope (please report upstream):

- Vulnerabilities in transitive Python dependencies — open an issue with
  the upstream package and reference here for tracking.
- Vulnerabilities in third-party GitHub Actions consumed by our
  workflows — report to the action maintainers; we will pin / replace
  as needed.
- Issues in upstream LLM providers or vector-index backends — report
  through their official channels.
