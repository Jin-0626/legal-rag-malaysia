# pgvector Status

## Current Runtime Status

Current runtime retrieval backend:

- JSONL vector store

This is the active, working retrieval path used by the local demo and current benchmark runs.

## What Is Implemented

- PostgreSQL/pgvector schema scaffold in [db/init/001_init.sql](/D:/Desktop/legal-rag-malaysia/db/init/001_init.sql)
- database health checks in [src/legal_rag/storage/postgres.py](/D:/Desktop/legal-rag-malaysia/src/legal_rag/storage/postgres.py)
- `/health` reporting for:
  - database enabled
  - database connectivity
  - pgvector extension readiness
- Docker Compose service using `pgvector/pgvector:pg16`

## What Is Not Implemented

- pgvector as the active retrieval backend
- migration of the existing JSONL embeddings corpus into PostgreSQL
- live vector search against PostgreSQL from the retrieval pipeline
- parity evaluation between JSONL retrieval and a pgvector-backed retriever

## What Would Be Required To Make pgvector Active

1. Add a production vector-store adapter for PostgreSQL/pgvector.
2. Add an indexing path that writes embeddings and metadata into PostgreSQL.
3. Add retrieval execution against pgvector similarity search.
4. Verify metadata and citation parity with the current JSONL retriever.
5. Re-run Final Gold Set V2 after switching retrieval backends.

## Production Impact

The repo is honest about pgvector status:

- PostgreSQL/pgvector is a production scaffold for future DB-backed retrieval.
- It should not be described as the active retrieval backend today.
