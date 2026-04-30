"""Pydantic schemas for the Legal RAG demo API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: Literal["auto", "hybrid", "graph"] = "auto"
    top_k: int = Field(default=5, ge=1, le=10)


class SourceItem(BaseModel):
    document: str
    unit_type: str
    unit_id: str
    heading: str
    score: float
    chunk_count: int = 1
    preview: str


class ChatResponse(BaseModel):
    answer: str
    mode_used: str
    sources: list[SourceItem]
    graph_path: list[str]
    warnings: list[str]


class HealthResponse(BaseModel):
    status: str
    ollama_available: bool
    ollama_base_url: str
    vector_store_loaded: bool
    indexed_chunks: int
    model: str
    model_available: bool
    chat_ready: bool
    database_enabled: bool = False
    database_connected: bool = False
    database_backend: str = "postgresql+pgvector"
    database_error: str | None = None
    error: str | None = None
