# Architecture

## System Flow

```text
User -> React UI -> FastAPI -> Auth/RBAC -> Retrieval -> Reranker -> Graph Assist -> Answer Generator -> Sources
```

## Backend

`src/legal_rag/api/` contains the FastAPI app and production-hardening layers:

- `GET /health`
- `POST /chat`
- `POST /chat_stream`
- `GET /admin/security`

Production support around the API now includes:

- API key auth with `X-API-Key`
- RBAC roles: `admin`, `researcher`, `viewer`
- structured request logging with request IDs
- Ollama health checks with model readiness
- optional PostgreSQL/pgvector health checks

## Ingestion and Chunking

- `src/legal_rag/ingestion/`
  - parses real law PDFs with PyMuPDF
  - normalizes extracted text for legal chunking
- `src/legal_rag/chunking/`
  - preserves section/article/perkara boundaries
  - keeps overlap inside legal units
  - supports subsection- and clause-aware chunking

## Retrieval Pipeline

```text
Query -> lexical retrieval + vector retrieval -> hybrid merge -> legal-aware filtering -> reranker -> citation-grounded answer
```

Current runtime retrieval remains the working local path:

- JSONL vector store
- hybrid retrieval
- legal-aware reranking
- grouped section-level sources

## Graph Assist

- `src/legal_rag/graph/`
  - legal structure graph
  - amendment and reference link support
- `src/legal_rag/workflows/`
  - graph-assisted routing for structural queries
  - preserves the non-graph hybrid baseline for general queries

## Answer Generation

- `src/legal_rag/generation/`
  - citation-grounded answer synthesis
  - abstention on weak, missing, or ambiguous evidence
  - explicit legal-information/not-legal-advice framing
  - checklist-style synthesis for broad employment-agreement queries

## Storage and Deployment Modes

### Local / current working mode

- processed corpus under `data/processed/`
- JSONL embeddings under `data/embeddings/`
- retrieval runs against the current local vector-store path

### Production scaffold mode

- `src/legal_rag/storage/postgres.py`
- `db/init/001_init.sql`
- `docker-compose.yml`

This adds PostgreSQL/pgvector schema and health readiness without replacing the stable local retrieval backend.

## Frontend and Integration

- `frontend/`
  - React + Vite chatbot
  - grouped sources
  - graph path panel
  - streaming answer rendering
- `ragflow_tools/`
  - thin RAGFlow wrapper around the existing FastAPI API

## Evaluation

- `data/evaluation/final_gold_set_v2.jsonl`
- `data/evaluation/final_gold_set_v2_report.json`
- `data/evaluation/final_gold_set_v2_report.md`

The benchmark of record is Final Gold Set V2 and is preserved as a first-class repo artifact.
