# RAGFlow Integration

The repo includes a thin RAGFlow wrapper around the existing FastAPI `/chat` endpoint.

## Files

- `ragflow_tools/legal_rag_tool.py`
- `ragflow_tools/legal_rag_tool.json`
- `ragflow_tools/legal_rag_simple_flow.json`

## Backend Requirement

Run the backend first:

```powershell
.\venv_new\Scripts\python.exe scripts/run_api.py
```

## Wrapper Invocation

```powershell
$env:LEGAL_RAG_API_BASE_URL='http://127.0.0.1:8000'
$env:LEGAL_RAG_API_KEY=''
.\venv_new\Scripts\python.exe ragflow_tools\legal_rag_tool.py "Which section introduces data portability?"
```

## Notes

- RAGFlow is used only as the orchestration layer.
- Retrieval, rerank, graph logic, and answer synthesis continue to live in the Legal RAG backend.
- If `LEGAL_RAG_API_KEY` is set, the wrapper sends `X-API-Key` to the backend automatically.
- If backend auth is enabled, configure `LEGAL_RAG_API_KEY` in the RAGFlow environment rather than disabling auth globally.
- If your RAGFlow version expects slightly different tool metadata, adapt only the JSON registration files and keep the Python wrapper unchanged.
