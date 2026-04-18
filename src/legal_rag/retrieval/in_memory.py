"""Simple embedding-based retrieval for scaffold and early pipeline wiring."""

from __future__ import annotations

import math
from dataclasses import dataclass

from legal_rag.chunking.models import Chunk
from legal_rag.embeddings.embedder import EmbeddedChunk, OllamaEmbedder


@dataclass(frozen=True)
class RetrievalResult:
    """Scored retrieval result for one chunk."""

    chunk: Chunk
    score: float


class SimpleVectorIndex:
    """Store embedded chunks in memory and search them by cosine similarity."""

    def __init__(self, embedder: OllamaEmbedder) -> None:
        self.embedder = embedder
        self._entries: list[EmbeddedChunk] = []

    def add(self, chunks: list[Chunk]) -> None:
        self._entries.extend(self.embedder.embed_chunks(chunks))

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        if not query.strip() or not self._entries:
            return []

        query_embedding = self.embedder.embed([query])[0]
        scored = [
            RetrievalResult(
                chunk=entry.chunk,
                score=_cosine_similarity(query_embedding, entry.embedding),
            )
            for entry in self._entries
        ]
        ranked = [item for item in scored if item.score > 0.0]
        return sorted(
            ranked,
            key=lambda item: (-item.score, item.chunk.chunk_id),
        )[:top_k]


class SimpleRetriever:
    """Compatibility wrapper around the embedding-based in-memory vector index."""

    def __init__(self, embedder: OllamaEmbedder) -> None:
        self.index = SimpleVectorIndex(embedder)

    def add(self, chunks: list[Chunk]) -> None:
        """Add embedded chunks to the current in-memory index."""

        self.index.add(chunks)

    def search(
        self, query: str, chunks: list[Chunk], top_k: int = 3
    ) -> list[RetrievalResult]:
        """Rebuild the simple in-memory index from chunks and search by embedding similarity."""

        self.index = SimpleVectorIndex(self.index.embedder)
        self.index.add(chunks)
        return self.index.search(query, top_k=top_k)


class EmbeddingRetriever(SimpleRetriever):
    """Explicit embedding-based retriever alias for downstream clarity."""


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    return dot_product / (left_norm * right_norm)
