"""Run the Legal RAG demo API with uvicorn."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    host = os.getenv("LEGAL_RAG_API_HOST", "127.0.0.1")
    port = int(os.getenv("LEGAL_RAG_API_PORT", "8000"))
    uvicorn.run("legal_rag.api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
