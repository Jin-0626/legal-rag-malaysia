"""Ollama-backed embedding client and helpers for chunk vectorization."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

import requests

from legal_rag.chunking.models import Chunk


class EmbeddingError(RuntimeError):
    """Raised when an embedding request fails or returns invalid data."""


class EmbeddingTransport(Protocol):
    """Transport interface to allow deterministic testing without a live Ollama server."""

    def embed(self, *, texts: list[str], model: str) -> list[list[float]]:
        """Return embeddings for the provided texts."""


@dataclass(frozen=True)
class EmbeddedChunk:
    """Chunk plus embedding vector for simple in-memory indexing."""

    chunk: Chunk
    embedding: list[float]


class OllamaHttpTransport:
    """Minimal HTTP transport for Ollama's embedding API."""

    def __init__(self, base_url: str, timeout_seconds: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def embed(self, *, texts: list[str], model: str) -> list[list[float]]:
        response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": model, "input": texts},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list):
            raise EmbeddingError("Ollama embedding response did not contain 'embeddings'.")

        normalized = [_normalize_embedding(vector) for vector in embeddings]
        if len(normalized) != len(texts):
            raise EmbeddingError("Ollama embedding response count did not match input count.")
        return normalized


class OllamaEmbedder:
    """Embed legal text using Ollama while caching repeated inputs."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        transport: EmbeddingTransport | None = None,
    ) -> None:
        self.model = model or os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.transport = transport or OllamaHttpTransport(self.base_url)
        self._cache: dict[str, list[float]] = {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        pending_texts: list[str] = []
        for text in texts:
            if text not in self._cache and text not in pending_texts:
                pending_texts.append(text)

        if pending_texts:
            embeddings = self.transport.embed(texts=pending_texts, model=self.model)
            for text, embedding in zip(pending_texts, embeddings, strict=True):
                self._cache[text] = embedding

        return [list(self._cache[text]) for text in texts]

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        embeddings = self.embed([chunk.text for chunk in chunks])
        return [
            EmbeddedChunk(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]


def _normalize_embedding(vector: Any) -> list[float]:
    if not isinstance(vector, list) or not vector:
        raise EmbeddingError("Embedding vector was empty or not a list.")

    normalized: list[float] = []
    for value in vector:
        if not isinstance(value, (int, float)):
            raise EmbeddingError("Embedding vector contained a non-numeric value.")
        normalized.append(float(value))
    return normalized
