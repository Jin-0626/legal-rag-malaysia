"""Embedding interfaces and Ollama-backed implementation."""

from .embedder import (
    EmbeddedChunk,
    EmbeddingError,
    EmbeddingTransport,
    OllamaEmbedder,
    OllamaHttpTransport,
)

__all__ = [
    "EmbeddedChunk",
    "EmbeddingError",
    "EmbeddingTransport",
    "OllamaEmbedder",
    "OllamaHttpTransport",
]
