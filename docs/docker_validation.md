# Docker Validation

## Purpose

Track actual Docker validation status separately from the deployment scaffold so the repo does not over-claim deployment readiness.

## Commands Run

```powershell
docker compose config
docker compose build

$env:POSTGRES_PASSWORD="temporary-local-password"
$env:DATABASE_URL="postgresql://legal_rag:temporary-local-password@postgres:5432/legal_rag"
docker compose up -d
Invoke-WebRequest http://localhost:8000/health
docker compose down
```

## Validation Log

### `docker compose config`

- Status: pass
- Result:
  - Compose file resolved successfully.
  - `docker-compose.override.yml` was applied.
  - Default local override values disable API-key enforcement and database-backed retrieval wiring for the app runtime, while still allowing the PostgreSQL service to run.
- Warnings observed:
  - `DATABASE_URL` variable is not set. Defaulting to a blank string.
  - `POSTGRES_PASSWORD` variable is not set. Defaulting to a blank string.
- Interpretation:
  - The stack definition is valid, but secure environment variables must be supplied before production deployment.

### `docker compose build`

- Status: pass
- Result:
  - Backend image built successfully.
  - Frontend image built successfully.
- Warnings observed:
  - `DATABASE_URL` variable is not set. Defaulting to a blank string.
  - `POSTGRES_PASSWORD` variable is not set. Defaulting to a blank string.
- Interpretation:
  - The repo contains a buildable container scaffold for backend and frontend.

### `docker compose up -d`

- Status: pass with temporary local secrets
- Validation method:
  - Temporary, non-committed values were injected for `POSTGRES_PASSWORD` and `DATABASE_URL`.
  - The stack was started in detached mode, then checked through the backend `/health` endpoint.
  - The stack was torn down immediately after validation with `docker compose down`.
- Services verified:
  - `postgres`: started
  - `backend`: started
  - `frontend`: started
- Backend health result:

```json
{
  "status": "ok",
  "ollama_available": true,
  "ollama_base_url": "http://host.docker.internal:11434",
  "vector_store_loaded": true,
  "indexed_chunks": 12562,
  "model": "gpt-oss:120b-cloud",
  "model_available": true,
  "chat_ready": true,
  "database_enabled": true,
  "database_connected": true,
  "database_backend": "postgresql+pgvector",
  "database_error": null,
  "error": null
}
```

## Logs Summary

- Backend booted successfully inside Docker and responded on `http://localhost:8000/health`.
- PostgreSQL connectivity worked when `DATABASE_URL` and `POSTGRES_PASSWORD` were supplied explicitly.
- Ollama health checks succeeded from inside the backend container via `host.docker.internal`.
- This validation confirms the Docker stack is runnable for local/demo use.

## Manual Fixes Still Required For Production

- Set `POSTGRES_PASSWORD` securely instead of relying on placeholder or temporary values.
- Set `DATABASE_URL` securely in deployment secrets or environment configuration.
- Provide a real API key config file when `LEGAL_RAG_REQUIRE_API_KEY=true`.
- Re-run `scripts/check_ollama.py` against the target Ollama endpoint and model before release/demo.
- If running outside Docker Desktop-compatible environments, verify `host.docker.internal` resolution or replace it with a platform-appropriate Ollama host.

## Honesty Rule

Docker Compose has now been validated for config resolution, image builds, and a bounded local startup/health-check cycle. This does not by itself prove a full production deployment pipeline, secret management workflow, or database-backed retrieval migration.
