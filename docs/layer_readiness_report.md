# Layer Readiness Report

## Summary Table

| Layer | Status | Main Evidence | Remaining Risk |
|---|---|---|---|
| Configuration / Secrets | partial | `.env.example`, `.gitignore`, `config/api_keys.example.json`, `docs/security.md`, `git status` | tracked temp-artifact deletions still need intentional cleanup before release |
| API / FastAPI | ready | `src/legal_rag/api/app.py`, `src/legal_rag/api/schemas.py`, `python -m pytest tests/test_api.py` | CORS is still local-demo-oriented rather than environment-driven |
| Authentication / RBAC | ready | `src/legal_rag/api/security.py`, `tests/test_api.py`, `tests/test_ragflow_tool.py` | production still depends on operators creating a real local key file securely |
| Retrieval / Reranking | ready | `src/legal_rag/retrieval/`, `src/legal_rag/api/service.py`, `tests/test_retrieval.py` from the current working stack | metrics remain tied to current corpus and benchmark |
| Answer Generation / Abstention | ready | `src/legal_rag/generation/grounded.py`, `src/legal_rag/api/service.py`, `tests/test_api.py` | grounded quality still depends on retrieved evidence quality |
| GraphRAG / Agentic Layer | partial | `src/legal_rag/graph/`, `src/legal_rag/workflows/`, `tests/test_graph_retrieval.py`, docs wording | graph assist remains targeted/experimental rather than a production dependency |
| Ollama / LLM Transport | partial | `src/legal_rag/api/service.py`, `scripts/check_ollama.py`, `tests/test_api.py` | each environment must still pass `check_ollama.py` for the configured model |
| PostgreSQL / pgvector | partial | `src/legal_rag/storage/postgres.py`, `db/init/001_init.sql`, `docs/pgvector_status.md` | pgvector is scaffolded, not the active retrieval backend |
| Docker / Deployment | partial | `Dockerfile.backend`, `frontend/Dockerfile`, `docker-compose.yml`, `docs/docker_validation.md` | local validation succeeded, but this is not a full production deployment pipeline |
| Logging / Observability | ready | `src/legal_rag/api/logging_utils.py`, `src/legal_rag/api/service.py`, `tests/test_api.py` | no centralized log sink or metrics export yet |
| Evaluation / Benchmark | partial | `data/evaluation/final_gold_set_v2.*`, `scripts/evaluate_final_gold_set_v2.py`, `docs/evaluation.md` | benchmark claims must be re-run after corpus changes |
| Frontend / UX | ready | `frontend/src/`, `npm run build`, streaming UI already present | no auth UI flow yet when backend API key mode is enabled |
| RAGFlow Integration | partial | `ragflow_tools/legal_rag_tool.py`, `ragflow_tools/README.md`, `docs/ragflow_integration.md`, `tests/test_ragflow_tool.py` | registration JSON may still need adaptation per RAGFlow version |
| Tests / CI | partial | `tests/`, new `.github/workflows/ci.yml`, local pytest/build runs | CI workflow is added but has not been exercised in GitHub yet |
| GitHub / Documentation | ready | `README.md`, `docs/*`, this report | screenshots and hosted demo are still optional, not included |

## Layer 1: Configuration / Secrets

Layer:
- Configuration / Secrets

Status:
- partial

Evidence:
- Files inspected:
  - [/.env.example](/D:/Desktop/legal-rag-malaysia/.env.example)
  - [/.gitignore](/D:/Desktop/legal-rag-malaysia/.gitignore)
  - [/config/api_keys.example.json](/D:/Desktop/legal-rag-malaysia/config/api_keys.example.json)
  - [/docs/security.md](/D:/Desktop/legal-rag-malaysia/docs/security.md)
- Commands reviewed:
  - `git status --short`

Risks:
- `POSTGRES_PASSWORD` and `DATABASE_URL` are deployment secrets and still depend on operator setup.
- `git status` still shows tracked `.pytest-tmp` deletions that must not be released accidentally.

Fixes applied:
- Updated [/.env.example](/D:/Desktop/legal-rag-malaysia/.env.example) to include `POSTGRES_PASSWORD=`.
- Updated [/.gitignore](/D:/Desktop/legal-rag-malaysia/.gitignore) to ignore `.env.*` and `config/api_keys.json`.

Validation:
- Manual config review completed.
- Safe template confirmed in [/config/api_keys.example.json](/D:/Desktop/legal-rag-malaysia/config/api_keys.example.json).

Next action:
- follow-up needed

## Layer 2: API / FastAPI

Layer:
- API / FastAPI

Status:
- ready

Evidence:
- Files inspected:
  - [/src/legal_rag/api/app.py](/D:/Desktop/legal-rag-malaysia/src/legal_rag/api/app.py)
  - [/src/legal_rag/api/schemas.py](/D:/Desktop/legal-rag-malaysia/src/legal_rag/api/schemas.py)
- Endpoints present:
  - `GET /health`
  - `POST /chat`
  - `POST /chat_stream`
  - `GET /admin/security`

Risks:
- CORS is still pinned to local frontend origins and will need environment-based expansion for broader deployment.

Fixes applied:
- none

Validation:
- `python -m pytest tests/test_api.py --basetemp .pytest-tmp-risk-api`
- Result: `23 passed`

Next action:
- future enhancement

## Layer 3: Authentication / RBAC

Layer:
- Authentication / RBAC

Status:
- ready

Evidence:
- Files inspected:
  - [/src/legal_rag/api/security.py](/D:/Desktop/legal-rag-malaysia/src/legal_rag/api/security.py)
  - [/ragflow_tools/legal_rag_tool.py](/D:/Desktop/legal-rag-malaysia/ragflow_tools/legal_rag_tool.py)
  - [/tests/test_api.py](/D:/Desktop/legal-rag-malaysia/tests/test_api.py)
  - [/tests/test_ragflow_tool.py](/D:/Desktop/legal-rag-malaysia/tests/test_ragflow_tool.py)
- Verified:
  - `X-API-Key`
  - SHA-256 hash matching
  - `hmac.compare_digest`
  - roles: `admin`, `researcher`, `viewer`

Risks:
- Production still depends on secure creation of a real `config/api_keys.json` outside the repo.

Fixes applied:
- Updated [/ragflow_tools/legal_rag_tool.py](/D:/Desktop/legal-rag-malaysia/ragflow_tools/legal_rag_tool.py) to send `X-API-Key` when `LEGAL_RAG_API_KEY` is set.
- Updated [/tests/test_ragflow_tool.py](/D:/Desktop/legal-rag-malaysia/tests/test_ragflow_tool.py) to cover auth/no-auth wrapper behavior.
- Updated [/ragflow_tools/README.md](/D:/Desktop/legal-rag-malaysia/ragflow_tools/README.md) and [/docs/ragflow_integration.md](/D:/Desktop/legal-rag-malaysia/docs/ragflow_integration.md).

Validation:
- `python -m pytest tests/test_api.py --basetemp .pytest-tmp-risk-api`
- `python -m pytest tests/test_ragflow_tool.py --basetemp .pytest-tmp-risk-ragflow`
- Results: `23 passed`, `3 passed`

Next action:
- no action

## Layer 4: Retrieval / Reranking

Layer:
- Retrieval / Reranking

Status:
- ready

Evidence:
- Files inspected:
  - [/src/legal_rag/retrieval/in_memory.py](/D:/Desktop/legal-rag-malaysia/src/legal_rag/retrieval/in_memory.py)
  - [/src/legal_rag/api/service.py](/D:/Desktop/legal-rag-malaysia/src/legal_rag/api/service.py)
  - [/docs/evaluation.md](/D:/Desktop/legal-rag-malaysia/docs/evaluation.md)
- Current runtime remains JSONL-backed hybrid retrieval with reranking.

Risks:
- Benchmark-backed performance claims are corpus-specific and not universal.

Fixes applied:
- none

Validation:
- Relied on the current protected retrieval test suite and benchmark docs already present in repo history.
- No retrieval code changed in this audit pass.

Next action:
- no action

## Layer 5: Answer Generation / Abstention

Layer:
- Answer Generation / Abstention

Status:
- ready

Evidence:
- Files inspected:
  - [/src/legal_rag/api/service.py](/D:/Desktop/legal-rag-malaysia/src/legal_rag/api/service.py)
  - [/src/legal_rag/generation/grounded.py](/D:/Desktop/legal-rag-malaysia/src/legal_rag/generation/grounded.py)
  - [/tests/test_api.py](/D:/Desktop/legal-rag-malaysia/tests/test_api.py)
- Structured sections and fallback abstention paths are implemented.

Risks:
- If retrieval surfaces narrow evidence, the answer remains intentionally narrow.

Fixes applied:
- none

Validation:
- `python -m pytest tests/test_api.py --basetemp .pytest-tmp-risk-api`
- Result: `23 passed`

Next action:
- no action

## Layer 6: GraphRAG / Agentic Layer

Layer:
- GraphRAG / Agentic Layer

Status:
- partial

Evidence:
- Files inspected:
  - [/src/legal_rag/graph/](/D:/Desktop/legal-rag-malaysia/src/legal_rag/graph)
  - [/src/legal_rag/workflows/](/D:/Desktop/legal-rag-malaysia/src/legal_rag/workflows)
  - [/docs/architecture.md](/D:/Desktop/legal-rag-malaysia/docs/architecture.md)
- Graph assist is positioned as targeted structural help, not baseline replacement.

Risks:
- Overclaiming graph maturity would be misleading; it is still intentionally limited to supported query classes.

Fixes applied:
- none

Validation:
- No graph logic changed in this pass.
- Existing graph test suite remains the correctness evidence for this layer.

Next action:
- future enhancement

## Layer 7: Ollama / LLM Transport

Layer:
- Ollama / LLM Transport

Status:
- partial

Evidence:
- Files inspected:
  - [/src/legal_rag/api/service.py](/D:/Desktop/legal-rag-malaysia/src/legal_rag/api/service.py)
  - [/scripts/check_ollama.py](/D:/Desktop/legal-rag-malaysia/scripts/check_ollama.py)
  - [/docs/setup.md](/D:/Desktop/legal-rag-malaysia/docs/setup.md)
- Checks present:
  - `/api/tags` reachability
  - model availability
  - `/api/chat` readiness
  - timeout/auth/malformed-response handling

Risks:
- A deployment can still degrade to fallback if the configured model is missing or remote auth is wrong.

Fixes applied:
- Updated [/docs/setup.md](/D:/Desktop/legal-rag-malaysia/docs/setup.md) to use the new API-key helper and to document `POSTGRES_PASSWORD`/`DATABASE_URL` as external secrets.

Validation:
- Transport and fallback cases are covered by `tests/test_api.py`.
- `scripts/check_ollama.py` remains the required pre-demo manual validation step.

Next action:
- follow-up needed

## Layer 8: PostgreSQL / pgvector

Layer:
- PostgreSQL / pgvector

Status:
- partial

Evidence:
- Files inspected:
  - [/src/legal_rag/storage/postgres.py](/D:/Desktop/legal-rag-malaysia/src/legal_rag/storage/postgres.py)
  - [/db/init/001_init.sql](/D:/Desktop/legal-rag-malaysia/db/init/001_init.sql)
  - [/docs/pgvector_status.md](/D:/Desktop/legal-rag-malaysia/docs/pgvector_status.md)

Risks:
- The repo could be misunderstood as DB-backed retrieval if the scaffold status is not stated clearly.

Fixes applied:
- None in code during this pass.
- Existing clarification doc remains the main mitigation.

Validation:
- Docker `/health` validation previously showed database connectivity works when enabled.
- No retrieval tests depend on PostgreSQL, which is correct for the current architecture.

Next action:
- future enhancement

## Layer 9: Docker / Deployment

Layer:
- Docker / Deployment

Status:
- partial

Evidence:
- Files inspected:
  - [/Dockerfile.backend](/D:/Desktop/legal-rag-malaysia/Dockerfile.backend)
  - [/frontend/Dockerfile](/D:/Desktop/legal-rag-malaysia/frontend/Dockerfile)
  - [/docker-compose.yml](/D:/Desktop/legal-rag-malaysia/docker-compose.yml)
  - [/docker-compose.override.yml](/D:/Desktop/legal-rag-malaysia/docker-compose.override.yml)
  - [/docs/docker_validation.md](/D:/Desktop/legal-rag-malaysia/docs/docker_validation.md)
- Commands previously run and recorded:
  - `docker compose config`
  - `docker compose build`
  - bounded `docker compose up -d` + `/health` + `docker compose down`

Risks:
- Local validation is not the same as a production deployment pipeline.
- Secrets still need to be injected securely by operators.

Fixes applied:
- none beyond the existing validation report

Validation:
- Actual results are recorded in [/docs/docker_validation.md](/D:/Desktop/legal-rag-malaysia/docs/docker_validation.md).

Next action:
- follow-up needed

## Layer 10: Logging / Observability

Layer:
- Logging / Observability

Status:
- ready

Evidence:
- Files inspected:
  - [/src/legal_rag/api/logging_utils.py](/D:/Desktop/legal-rag-malaysia/src/legal_rag/api/logging_utils.py)
  - [/src/legal_rag/api/service.py](/D:/Desktop/legal-rag-malaysia/src/legal_rag/api/service.py)
- Verified:
  - request IDs
  - route/method/status
  - latency
  - retrieval mode
  - fallback reason
  - query hashing in production mode

Risks:
- No centralized sink, dashboard, or metrics exporter yet.

Fixes applied:
- none

Validation:
- Logging behavior is exercised indirectly by the API test suite and code inspection.

Next action:
- future enhancement

## Layer 11: Evaluation / Benchmark

Layer:
- Evaluation / Benchmark

Status:
- partial

Evidence:
- Files inspected:
  - [/data/evaluation/final_gold_set_v2.jsonl](/D:/Desktop/legal-rag-malaysia/data/evaluation/final_gold_set_v2.jsonl)
  - [/data/evaluation/final_gold_set_v2_report.json](/D:/Desktop/legal-rag-malaysia/data/evaluation/final_gold_set_v2_report.json)
  - [/scripts/evaluate_final_gold_set_v2.py](/D:/Desktop/legal-rag-malaysia/scripts/evaluate_final_gold_set_v2.py)
  - [/docs/evaluation.md](/D:/Desktop/legal-rag-malaysia/docs/evaluation.md)
  - [/README.md](/D:/Desktop/legal-rag-malaysia/README.md)

Risks:
- Reported metrics are valid only for the current processed corpus and Final Gold Set V2.

Fixes applied:
- none beyond existing benchmark caveat wording

Validation:
- Reproduction command is documented.
- No benchmark contents were changed in this pass.

Next action:
- follow-up needed

## Layer 12: Frontend / UX

Layer:
- Frontend / UX

Status:
- ready

Evidence:
- Files inspected:
  - [/frontend/src/](/D:/Desktop/legal-rag-malaysia/frontend/src)
- Verified UX surfaces already present:
  - streaming answers
  - mode display
  - source panel
  - warnings
  - health panel

Risks:
- There is no dedicated login/auth UX; backend auth is header-based.

Fixes applied:
- none

Validation:
- `cd frontend && npm run build`
- Result: success

Next action:
- future enhancement

## Layer 13: RAGFlow Integration

Layer:
- RAGFlow Integration

Status:
- partial

Evidence:
- Files inspected:
  - [/ragflow_tools/legal_rag_tool.py](/D:/Desktop/legal-rag-malaysia/ragflow_tools/legal_rag_tool.py)
  - [/ragflow_tools/README.md](/D:/Desktop/legal-rag-malaysia/ragflow_tools/README.md)
  - [/docs/ragflow_integration.md](/D:/Desktop/legal-rag-malaysia/docs/ragflow_integration.md)

Risks:
- Tool registration JSON may still need per-version adaptation in RAGFlow itself.

Fixes applied:
- Updated [/docs/ragflow_integration.md](/D:/Desktop/legal-rag-malaysia/docs/ragflow_integration.md) to reflect the current `LEGAL_RAG_API_KEY` support instead of describing it as future work.

Validation:
- `python -m pytest tests/test_ragflow_tool.py --basetemp .pytest-tmp-risk-ragflow`
- Result: `3 passed`

Next action:
- follow-up needed

## Layer 14: Tests / CI

Layer:
- Tests / CI

Status:
- partial

Evidence:
- Files inspected:
  - [/tests/test_api.py](/D:/Desktop/legal-rag-malaysia/tests/test_api.py)
  - [/tests/test_ragflow_tool.py](/D:/Desktop/legal-rag-malaysia/tests/test_ragflow_tool.py)
  - [/.github/workflows/ci.yml](/D:/Desktop/legal-rag-malaysia/.github/workflows/ci.yml)

Risks:
- CI has been added, but it has not yet been exercised in GitHub Actions.

Fixes applied:
- Added [/.github/workflows/ci.yml](/D:/Desktop/legal-rag-malaysia/.github/workflows/ci.yml) to run:
  - `tests/test_api.py`
  - `tests/test_retrieval.py`
  - `tests/test_graph_retrieval.py`
  - `tests/test_ragflow_tool.py`
  - frontend `npm run build`

Validation:
- Local command coverage in this audit:
  - `python -m pytest tests/test_api.py --basetemp .pytest-tmp-risk-api`
  - `python -m pytest tests/test_ragflow_tool.py --basetemp .pytest-tmp-risk-ragflow`
  - `cd frontend && npm run build`

Next action:
- follow-up needed

## Layer 15: GitHub / Documentation

Layer:
- GitHub / Documentation

Status:
- ready

Evidence:
- Files inspected:
  - [/README.md](/D:/Desktop/legal-rag-malaysia/README.md)
  - [/docs/architecture.md](/D:/Desktop/legal-rag-malaysia/docs/architecture.md)
  - [/docs/setup.md](/D:/Desktop/legal-rag-malaysia/docs/setup.md)
  - [/docs/deployment.md](/D:/Desktop/legal-rag-malaysia/docs/deployment.md)
  - [/docs/security.md](/D:/Desktop/legal-rag-malaysia/docs/security.md)
  - [/docs/evaluation.md](/D:/Desktop/legal-rag-malaysia/docs/evaluation.md)
  - [/docs/release_checklist.md](/D:/Desktop/legal-rag-malaysia/docs/release_checklist.md)

Risks:
- Screenshots and hosted demo artifacts are still optional rather than embedded in the repo.

Fixes applied:
- Added this report at [/docs/layer_readiness_report.md](/D:/Desktop/legal-rag-malaysia/docs/layer_readiness_report.md).
- Updated [/docs/release_checklist.md](/D:/Desktop/legal-rag-malaysia/docs/release_checklist.md) to include layer review, CI review, Docker status, Ollama status, secret checks, and temp-artifact review.

Validation:
- Manual doc review completed.

Next action:
- no action
