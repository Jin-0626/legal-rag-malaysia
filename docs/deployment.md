# Deployment

## Local Development

Backend:

```powershell
.\venv_new\Scripts\python.exe scripts/check_ollama.py
.\venv_new\Scripts\python.exe scripts/run_api.py
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

## Docker Compose

The repository includes a production-style Docker scaffold:

- `Dockerfile.backend`
- `frontend/Dockerfile`
- `docker-compose.yml`
- `docker-compose.override.yml`
- `.dockerignore`

Start the stack:

```powershell
docker compose up --build
```

### Services

- `backend`: FastAPI API and retrieval orchestration
- `frontend`: React UI served from Nginx
- `postgres`: PostgreSQL with pgvector extension enabled by init SQL

### Important Notes

- `DATABASE_URL` and `POSTGRES_PASSWORD` must be supplied securely for Docker deployments.
- `LEGAL_RAG_REQUIRE_API_KEY` defaults to `true` in `docker-compose.yml`.
- The current working retrieval path remains JSONL-based by default.
- PostgreSQL/pgvector is currently a production scaffold for storage, health checks, and schema readiness, not the default live retrieval backend.

## PostgreSQL / pgvector Status

Implemented:

- `db/init/001_init.sql`
- `src/legal_rag/storage/postgres.py`
- database connectivity and pgvector health reporting in `/health`

Not yet wired as the default live retriever:

- direct pgvector-backed retrieval execution
- migration of the existing JSONL corpus/index into PostgreSQL

This keeps the current local demo stable while making deployment hardening visible and reviewable.
