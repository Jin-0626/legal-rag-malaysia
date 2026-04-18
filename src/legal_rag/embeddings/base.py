"""Backward-compatible exports for the Ollama embedder."""

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
