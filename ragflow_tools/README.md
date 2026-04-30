# RAGFlow Integration

This folder exposes the existing Malaysian Legal RAG `/chat` API as a thin external tool for RAGFlow.

## Files

- `legal_rag_tool.py`
  Thin Python wrapper that calls the existing FastAPI backend.
- `legal_rag_tool.json`
  Lightweight tool registration metadata.
- `legal_rag_simple_flow.json`
  Minimal flow template: user input -> tool -> answer output.

## Assumption

This integration keeps the current system unchanged:

- retrieval
- reranking
- graph routing
- chunking
- embeddings

RAGFlow is used only as the orchestration layer.

## Backend requirement

Start the current backend first:

```powershell
.\venv_new\Scripts\python.exe scripts/run_api.py
```

The wrapper reads:

- `LEGAL_RAG_API_BASE_URL`
- `LEGAL_RAG_API_KEY` (optional)

Default:

```env
LEGAL_RAG_API_BASE_URL=http://127.0.0.1:8000
LEGAL_RAG_API_KEY=
```

## Wrapper behavior

The tool sends:

```json
{
  "query": "...",
  "mode": "auto",
  "top_k": 5
}
```

to:

```text
POST {LEGAL_RAG_API_BASE_URL}/chat
```

If `LEGAL_RAG_API_KEY` is set, the wrapper sends:

```text
X-API-Key: <value>
```

If it is empty, the wrapper behaves exactly as before.

and returns:

```json
{
  "answer": "...",
  "sources": [],
  "mode_used": "...",
  "graph_path": [],
  "warnings": []
}
```

## Example direct invocation

```powershell
$env:LEGAL_RAG_API_BASE_URL="http://127.0.0.1:8000"
$env:LEGAL_RAG_API_KEY=""
.\venv_new\Scripts\python.exe ragflow_tools\legal_rag_tool.py "What does Section 2 of Employment Act 1955 define?"
```

## RAGFlow registration note

Different RAGFlow releases can expect slightly different tool/flow metadata keys. The JSON files here are minimal templates intended to keep the integration reproducible inside this repo. If your RAGFlow instance expects a different import schema, keep the Python wrapper unchanged and adapt only the metadata file.
