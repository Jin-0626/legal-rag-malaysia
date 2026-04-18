"""File-backed vector-store helpers for indexed legal chunks."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from legal_rag.chunking.models import Chunk
from legal_rag.embeddings.embedder import OllamaEmbedder
from legal_rag.retrieval.in_memory import RetrievalResult


@dataclass(frozen=True)
class StoredVectorRecord:
    """Flat persisted vector-store record for one legal chunk."""

    chunk_id: str
    document_id: str
    act_title: str
    act_number: str
    section_heading: str
    section_id: str
    subsection_id: str | None
    paragraph_id: str | None
    source_file: str
    source_path: str
    chunk_index: int
    text: str
    embedding: list[float]


def load_chunk_records(jsonl_path: Path) -> list[Chunk]:
    """Load exported chunk JSONL records into Chunk objects."""

    path = Path(jsonl_path)
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            chunks.append(chunk_from_record(payload))
    return chunks


def chunk_from_record(record: dict[str, Any]) -> Chunk:
    """Convert a flat exported record into a Chunk."""

    return Chunk(
        chunk_id=record["chunk_id"],
        document_id=record["document_id"],
        section_heading=record["section_heading"],
        section_id=record["section_id"],
        subsection_id=record.get("subsection_id"),
        paragraph_id=record.get("paragraph_id"),
        text=record["text"],
        source_path=record["source_path"],
        act_title=record.get("act_title", ""),
        act_number=record.get("act_number", ""),
        source_file=record.get("source_file", ""),
        chunk_index=int(record.get("chunk_index", 0)),
    )


def chunk_to_stored_record(chunk: Chunk, embedding: list[float]) -> StoredVectorRecord:
    """Convert a chunk plus embedding into a persisted vector-store record."""

    return StoredVectorRecord(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        act_title=chunk.act_title,
        act_number=chunk.act_number,
        section_heading=chunk.section_heading,
        section_id=chunk.section_id,
        subsection_id=chunk.subsection_id,
        paragraph_id=chunk.paragraph_id,
        source_file=chunk.source_file,
        source_path=chunk.source_path,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        embedding=list(embedding),
    )


class JsonlVectorStore:
    """Persist and query embedded legal chunks from a JSONL vector store."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def index_chunks(self, chunks: list[Chunk], embedder: OllamaEmbedder) -> int:
        """Embed chunks and write them into the JSONL vector store."""

        embeddings = embedder.embed([chunk.text for chunk in chunks])
        records = [
            chunk_to_stored_record(chunk, embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self.write_records(records)
        return len(records)

    def write_records(self, records: list[StoredVectorRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def load_records(self) -> list[StoredVectorRecord]:
        if not self.path.exists():
            return []

        records: list[StoredVectorRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                records.append(
                    StoredVectorRecord(
                        chunk_id=payload["chunk_id"],
                        document_id=payload["document_id"],
                        act_title=payload.get("act_title", ""),
                        act_number=payload.get("act_number", ""),
                        section_heading=payload["section_heading"],
                        section_id=payload["section_id"],
                        subsection_id=payload.get("subsection_id"),
                        paragraph_id=payload.get("paragraph_id"),
                        source_file=payload.get("source_file", ""),
                        source_path=payload["source_path"],
                        chunk_index=int(payload.get("chunk_index", 0)),
                        text=payload["text"],
                        embedding=[float(value) for value in payload["embedding"]],
                    )
                )
        return records

    def search(
        self,
        query: str,
        embedder: OllamaEmbedder,
        top_k: int = 3,
    ) -> list[RetrievalResult]:
        if not query.strip():
            return []

        records = self.load_records()
        if not records:
            return []

        query_embedding = embedder.embed([query])[0]
        ranked: list[RetrievalResult] = []
        for record in records:
            score = _cosine_similarity(query_embedding, record.embedding)
            if score <= 0.0:
                continue
            ranked.append(
                RetrievalResult(
                    chunk=chunk_from_record(
                        {
                            "chunk_id": record.chunk_id,
                            "document_id": record.document_id,
                            "act_title": record.act_title,
                            "act_number": record.act_number,
                            "section_heading": record.section_heading,
                            "section_id": record.section_id,
                            "subsection_id": record.subsection_id,
                            "paragraph_id": record.paragraph_id,
                            "source_file": record.source_file,
                            "source_path": record.source_path,
                            "chunk_index": record.chunk_index,
                            "text": record.text,
                        }
                    ),
                    score=score,
                )
            )

        return sorted(ranked, key=lambda item: (-item.score, item.chunk.chunk_id))[:top_k]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    return dot_product / (left_norm * right_norm)
