# Production-ready Legal RAG Malaysia

A production-style Malaysian legal AI system combining hybrid retrieval, reranking, citation-grounded answer generation, GraphRAG-assisted reasoning, and secure FastAPI deployment.

## Why This Project Exists

This repository is built around one practical legal AI goal: return the correct Malaysian legal unit with traceable citations, and abstain when the retrieved sources do not support the answer.

## Features

- FastAPI backend
- React chatbot UI
- hybrid retrieval
- reranker
- citation-grounded answers
- abstention policy
- evaluation benchmark
- Docker Compose
- PostgreSQL / pgvector scaffold
- RBAC
- API key security
- structured logging
- Ollama local/cloud
- RAGFlow integration

## Architecture

```text
User -> React UI -> FastAPI -> Auth/RBAC -> Retrieval -> Reranker -> Graph Assist -> Answer Generator -> Sources
```

## Retrieval Pipeline

```text
Query -> lexical + vector retrieval -> hybrid merge -> legal-aware filtering -> reranker -> citation-grounded answer
```

The active retrieval runtime remains the JSONL-backed corpus and vector store. PostgreSQL/pgvector is included as a deployment scaffold, not as the live retrieval backend.

## Why Hybrid + Rerank

- embedding-only retrieval is too weak for exact legal-unit lookup and cross-document disambiguation
- lexical retrieval catches section names and act references but misses paraphrase
- hybrid retrieval improves candidate recall by combining both signals
- reranking is the critical step that resolves same-document section collisions and drives citation-ready top-1 accuracy

## Demo

Backend:

```powershell
.\venv_new\Scripts\python.exe -m pip install -r requirements.txt
.\venv_new\Scripts\python.exe scripts/check_ollama.py
.\venv_new\Scripts\python.exe scripts/run_api.py
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

Generated corpus and vector artifacts can be rebuilt from raw PDFs:

```powershell
.\venv_new\Scripts\python.exe scripts/rebuild_all.py
```

## FastAPI Backend

Documented API endpoints:

- `GET /health`
- `POST /chat`
- `POST /chat_stream`
- `GET /admin/security`

Health reports include:

- app status
- vector/index status
- Ollama status
- model availability
- database enablement/connectivity
- non-secret error details

## Citation Grounding and Abstention

The system is designed to:

- answer only from retrieved legal sources
- cite section/article/perkara references
- distinguish legal information from legal advice
- abstain when evidence is missing, weak, or ambiguous
- avoid hallucinating impossible unit lookups like `Section 999`

Structured answer format:

- `Direct Answer`
- `Legal Basis`
- `Practical Meaning`
- `Important Limits`
- `Sources`

## Evaluation

Backed by `data/evaluation/final_gold_set_v2_report.md`:

- Final Gold Set V2 benchmark
- `hybrid_rerank` hit@1: `1.000`
- `hybrid_filtered_rerank` hit@1: `1.000`
- `hybrid_plus_graph_with_graph_rerank` hit@1: `1.000`
- wrong-document rate: `0.000`
- wrong-section rate: `0.000`
- negative/no-answer accuracy: `1.000`

Metrics should be rerun after corpus or benchmark changes.

Reproduce:

```powershell
$env:PYTHONPATH='src'
.\venv_new\Scripts\python.exe scripts/evaluate_final_gold_set_v2.py
```

## Quickstart Local

Generated artifacts in `data/processed/` and `data/embeddings/` are rebuildable outputs and are not intended to be source-controlled.

Backend:

```powershell
.\venv_new\Scripts\python.exe -m pip install -r requirements.txt
.\venv_new\Scripts\python.exe scripts/check_ollama.py
.\venv_new\Scripts\python.exe scripts/run_api.py
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

Ollama:

```powershell
ollama serve
ollama pull gpt-oss:120b-cloud
ollama pull nomic-embed-text
```

Data rebuild:

```powershell
.\venv_new\Scripts\python.exe scripts/rebuild_all.py
```

## Quickstart Docker

```powershell
docker compose up --build
```

Notes:

- supply `DATABASE_URL` and `POSTGRES_PASSWORD` securely before using the Compose stack
- PostgreSQL/pgvector is scaffolded for deployment readiness and health checks, but the live local retrieval path still defaults to the JSONL vector store

## Example Queries

- What does Section 2 of the Employment Act 1955 define?
- Which section introduces data portability in PDPA Amendment Act 2024?
- What do I need to check before signing an employment agreement?
- Apakah kandungan Perkara 8 dalam Perlembagaan Persekutuan?

## Security

- local development can run with `LEGAL_RAG_REQUIRE_API_KEY=false`
- production deployment should use `LEGAL_RAG_REQUIRE_API_KEY=true`
- API keys are provided via `X-API-Key`
- only SHA-256 key hashes should be stored
- no `.env`, secrets, or real tokens should be committed
- request logging avoids raw API key disclosure

## Project Structure

```text
legal-rag-malaysia/
+-- src/legal_rag/
¦   +-- api/
¦   +-- chunking/
¦   +-- embeddings/
¦   +-- evaluation/
¦   +-- generation/
¦   +-- graph/
¦   +-- ingestion/
¦   +-- retrieval/
¦   +-- storage/
¦   +-- workflows/
+-- scripts/
+-- frontend/
+-- ragflow_tools/
+-- db/init/
+-- config/
+-- data/
¦   +-- raw_law_pdfs/
¦   +-- processed/        # generated locally
¦   +-- embeddings/       # generated locally
¦   +-- evaluation/
+-- docs/
+-- docker-compose.yml
+-- Dockerfile.backend
+-- README.md
```

## RAGFlow Integration

RAGFlow is used as an orchestration layer only. The backend remains the system of record for retrieval, reranking, graph assist, and grounded answer generation.

## Limitations

- corpus coverage is limited to the currently ingested Malaysian legal documents
- outputs are legal information, not legal advice
- GraphRAG-assisted routing is still targeted and experimental rather than a full graph-native retrieval replacement
- PostgreSQL/pgvector support is scaffolded for deployment readiness and health checks, not yet the default live retriever
- generated processed/vector artifacts are rebuilt locally rather than committed as durable source assets

## Tests

```powershell
.\venv_new\Scripts\python.exe -m pytest
cd frontend
npm run build
```

## License

This project is for educational, research, and portfolio purposes.
