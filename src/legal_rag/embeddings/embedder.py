"""Ollama-backed embedding client and helpers for chunk vectorization."""

from __future__ import annotations

import os
import re
import unicodedata
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
            embeddings = self._embed_with_retry(pending_texts)
            for text, embedding in zip(pending_texts, embeddings, strict=True):
                self._cache[text] = embedding

        return [list(self._cache[text]) for text in texts]

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        embeddings = self.embed([chunk.text for chunk in chunks])
        return [
            EmbeddedChunk(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        try:
            return self.transport.embed(texts=texts, model=self.model)
        except Exception:
            recovered_embeddings: list[list[float]] = []
            for text in texts:
                recovered_embeddings.append(self._embed_single_with_retry(text))
            return recovered_embeddings

    def _embed_single_with_retry(self, text: str) -> list[float]:
        try:
            return self.transport.embed(texts=[text], model=self.model)[0]
        except Exception as original_exc:
            sanitized = _sanitize_embedding_text(text)
            if not sanitized or sanitized == text:
                raise original_exc
            return self.transport.embed(texts=[sanitized], model=self.model)[0]


def _normalize_embedding(vector: Any) -> list[float]:
    if not isinstance(vector, list) or not vector:
        raise EmbeddingError("Embedding vector was empty or not a list.")

    normalized: list[float] = []
    for value in vector:
        if not isinstance(value, (int, float)):
            raise EmbeddingError("Embedding vector contained a non-numeric value.")
        normalized.append(float(value))
    return normalized


def _sanitize_embedding_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    cleaned_parts: list[str] = []
    for character in normalized:
        if character in {"\n", "\r", "\t"}:
            cleaned_parts.append(" ")
            continue
        if ord(character) < 128:
            cleaned_parts.append(character)
            continue

        category = unicodedata.category(character)
        if category.startswith("P") or category.startswith("S"):
            cleaned_parts.append(
                {
                    "“": '"',
                    "”": '"',
                    "‘": "'",
                    "’": "'",
                    "–": "-",
                    "—": "-",
                    "…": "...",
                }.get(character, " ")
            )
            continue

        if category.startswith("L") and "LATIN" in unicodedata.name(character, ""):
            ascii_equivalent = unicodedata.normalize("NFKD", character).encode("ascii", "ignore").decode("ascii")
            cleaned_parts.append(ascii_equivalent or " ")
            continue

        if category.startswith("N"):
            cleaned_parts.append(character)
            continue

        cleaned_parts.append(" ")

    sanitized = "".join(cleaned_parts)
    sanitized = re.sub(r"([.\-_])\1{3,}", " ... ", sanitized)
    sanitized = re.sub(r"([=*~])\1{3,}", " ", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    if len(sanitized) > 2400:
        truncated = sanitized[:2400]
        sanitized = truncated.rsplit(" ", 1)[0] or truncated
    return sanitized
