# Production Readiness Report

| Risk | Status before | Mitigation applied | Remaining action | Production impact |
|---|---|---|---|---|
| pgvector not active | PostgreSQL/pgvector existed only as a requirement in the target state | Added DB init SQL, DB health checks, Docker service, and explicit status docs | Build a real pgvector-backed index and retriever before claiming DB-backed retrieval | Low risk to current demo, because JSONL remains the active stable backend |
| Docker not validated | Compose files existed only as a target | Added Dockerfiles, Compose scaffold, and validation report workflow | Complete a full `docker compose up --build` runtime test in an environment with Docker and required secrets | Medium deployment risk until runtime-tested |
| RAGFlow auth gap | Wrapper could not send `X-API-Key` | Added optional `LEGAL_RAG_API_KEY` support and tests | Provide a real env value when backend auth is enabled | Low integration risk once configured |
| API key setup was manual-only | No helper for hashing keys | Added `scripts/create_api_key_hash.py` and security docs | Generate a real local `config/api_keys.json` outside the repo | Medium operational risk if skipped |
| Benchmark caveat unclear | Showcase metrics could be overread as corpus-invariant | Updated README and evaluation docs to say metrics are tied to the current corpus and should be re-run after corpus changes | Re-run benchmark after any corpus/index change | Low if release discipline is followed |
| Temp artifact git noise | `.pytest-tmp` deletions remained visible in `git status` | Documented release-checklist gate so temp deletions are not staged accidentally | Clean the working tree intentionally before release | Low runtime risk, medium release-hygiene risk |
| Ollama preflight could be skipped | `check_ollama.py` existed but was easy to miss | Elevated in setup and release docs | Run it before demo/release for the configured model | Medium demo reliability risk if skipped |

## Current Honest Status

- The repo is production-hardened around the working Legal RAG stack.
- The local runtime remains the stable JSONL-backed retrieval path.
- PostgreSQL/pgvector is deployment scaffolding, not the active retrieval backend.
- Docker support exists, but production claims should remain tied to actual validation logs.
