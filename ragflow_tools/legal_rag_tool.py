"""Thin RAGFlow tool wrapper for the existing Legal RAG /chat endpoint."""

from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_API_BASE_URL = os.getenv("LEGAL_RAG_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_API_KEY = os.getenv("LEGAL_RAG_API_KEY") or None
DEFAULT_TOP_K = 5


def tool_definition() -> dict[str, Any]:
    """Return a lightweight tool manifest for RAGFlow-style registration."""

    return {
        "name": "legal_rag_query",
        "description": "Query the Malaysian Legal RAG API and return grounded answers with sources.",
        "inputs": [
            {
                "name": "query",
                "type": "string",
                "required": True,
                "description": "The legal question to send to the existing /chat endpoint.",
            },
            {
                "name": "mode",
                "type": "string",
                "required": False,
                "default": "auto",
                "enum": ["auto", "hybrid", "graph"],
                "description": "Retrieval mode passed through to the Legal RAG API.",
            },
        ],
        "outputs": [
            {"name": "answer", "type": "string"},
            {"name": "sources", "type": "array"},
            {"name": "mode_used", "type": "string"},
            {"name": "graph_path", "type": "array"},
            {"name": "warnings", "type": "array"},
        ],
    }


def run(query: str, mode: str = "auto", top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Call the existing Legal RAG API and return a RAGFlow-friendly result payload."""

    headers: dict[str, str] = {}
    if DEFAULT_API_KEY:
        headers["X-API-Key"] = DEFAULT_API_KEY

    response = requests.post(
        f"{DEFAULT_API_BASE_URL}/chat",
        json={
            "query": query,
            "mode": mode,
            "top_k": top_k,
        },
        headers=headers or None,
        timeout=90,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "answer": payload.get("answer", ""),
        "sources": payload.get("sources", []),
        "mode_used": payload.get("mode_used", mode),
        "graph_path": payload.get("graph_path", []),
        "warnings": payload.get("warnings", []),
    }


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python ragflow_tools/legal_rag_tool.py \"<query>\" [mode]")
    print(json.dumps(run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "auto"), indent=2))
