# Setup

## Python

```powershell
.\venv_new\Scripts\python.exe -m pip install -r requirements.txt
```

## Ollama

Local Ollama:

```powershell
ollama serve
ollama pull gpt-oss:120b-cloud
ollama pull nomic-embed-text
```

If you use a different local or remote model, set `OLLAMA_MODEL` accordingly.

## Environment

Copy `.env.example` to `.env` and adjust:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:120b-cloud
OLLAMA_TIMEOUT_SECONDS=60
OLLAMA_API_KEY=
APP_ENV=development
LOG_LEVEL=INFO
LEGAL_RAG_REQUIRE_API_KEY=false
LEGAL_RAG_API_KEYS_FILE=config/api_keys.example.json
LEGAL_RAG_DATABASE_ENABLED=false
DATABASE_URL=
POSTGRES_PASSWORD=
LEGAL_RAG_VECTOR_STORE=data/embeddings/legal-corpus.vectors.jsonl
LEGAL_RAG_API_HOST=127.0.0.1
LEGAL_RAG_API_PORT=8000
VITE_API_BASE_URL=http://127.0.0.1:8000
```

PowerShell local configuration:

```powershell
$env:OLLAMA_BASE_URL="http://localhost:11434"
$env:OLLAMA_MODEL="gpt-oss:120b-cloud"
$env:OLLAMA_API_KEY=""
$env:OLLAMA_TIMEOUT_SECONDS="120"
$env:LEGAL_RAG_REQUIRE_API_KEY="false"
```

PowerShell remote/proxy configuration:

```powershell
$env:OLLAMA_BASE_URL="https://your-endpoint"
$env:OLLAMA_MODEL="your-model"
$env:OLLAMA_API_KEY="your-token"
$env:OLLAMA_TIMEOUT_SECONDS="120"
$env:LEGAL_RAG_REQUIRE_API_KEY="true"
```

## API Keys

Generate a SHA-256 hash for a real API key before enabling auth:

```powershell
.\venv_new\Scripts\python.exe scripts/create_api_key_hash.py --name local-admin --role admin
```

Then place the generated hash in a real local `config/api_keys.json` file. Do not commit the real file.

If you use PostgreSQL in Docker or another deployed environment, supply `POSTGRES_PASSWORD` and `DATABASE_URL` securely outside the repo rather than writing them into version-controlled files.

## Validate Ollama

```powershell
.\venv_new\Scripts\python.exe scripts/check_ollama.py
```

## Backend

```powershell
.\venv_new\Scripts\python.exe scripts/run_api.py
```

## Rebuild Generated Data

`data/processed/` and `data/embeddings/` are generated locally and are not source-controlled.

```powershell
.\venv_new\Scripts\python.exe scripts/rebuild_all.py
```

## Frontend

```powershell
cd frontend
npm install
npm run dev
```

## Docker

```powershell
docker compose up --build
```
