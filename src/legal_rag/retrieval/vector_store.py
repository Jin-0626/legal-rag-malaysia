"""File-backed vector-store helpers for indexed legal chunks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from legal_rag.chunking.models import Chunk
from legal_rag.embeddings.embedder import EmbeddedChunk, OllamaEmbedder
from legal_rag.retrieval.in_memory import RetrievalResult, SearchMode, search_embedded_entries


@dataclass(frozen=True)
class StoredVectorRecord:
    """Flat persisted vector-store record for one legal chunk."""

    chunk_id: str
    document_id: str
    act_title: str
    act_number: str
    section_heading: str
    section_id: str
    unit_type: str
    unit_id: str
    subsection_id: str | None
    paragraph_id: str | None
    source_file: str
    source_path: str
    chunk_index: int
    document_aliases: list[str]
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
        unit_type=record.get("unit_type", "section"),
        unit_id=record.get("unit_id", record["section_id"]),
        subsection_id=record.get("subsection_id"),
        paragraph_id=record.get("paragraph_id"),
        text=record["text"],
        source_path=record["source_path"],
        act_title=record.get("act_title", ""),
        act_number=record.get("act_number", ""),
        source_file=record.get("source_file", ""),
        chunk_index=int(record.get("chunk_index", 0)),
        document_aliases=tuple(record.get("document_aliases", [])),
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
        unit_type=chunk.unit_type,
        unit_id=chunk.unit_id or chunk.section_id,
        subsection_id=chunk.subsection_id,
        paragraph_id=chunk.paragraph_id,
        source_file=chunk.source_file,
        source_path=chunk.source_path,
        chunk_index=chunk.chunk_index,
        document_aliases=list(chunk.document_aliases),
        text=chunk.text,
        embedding=list(embedding),
    )


class JsonlVectorStore:
    """Persist and query embedded legal chunks from a JSONL vector store."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def index_chunks(
        self,
        chunks: list[Chunk],
        embedder: OllamaEmbedder,
        batch_size: int = 64,
    ) -> int:
        """Embed chunks and write them into the JSONL vector store."""

        records: list[StoredVectorRecord] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embeddings = embedder.embed([chunk.text for chunk in batch])
            records.extend(
                chunk_to_stored_record(chunk, embedding)
                for chunk, embedding in zip(batch, embeddings, strict=True)
            )
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
                        unit_type=payload.get("unit_type", "section"),
                        unit_id=payload.get("unit_id", payload["section_id"]),
                        subsection_id=payload.get("subsection_id"),
                        paragraph_id=payload.get("paragraph_id"),
                        source_file=payload.get("source_file", ""),
                        source_path=payload["source_path"],
                        chunk_index=int(payload.get("chunk_index", 0)),
                        document_aliases=list(payload.get("document_aliases", [])),
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
        mode: SearchMode = "hybrid",
    ) -> list[RetrievalResult]:
        if not query.strip():
            return []

        records = self.load_records()
        if not records:
            return []

        entries = [
            EmbeddedChunk(
                chunk=chunk_from_record(
                    {
                        "chunk_id": record.chunk_id,
                        "document_id": record.document_id,
                        "act_title": record.act_title,
                        "act_number": record.act_number,
                        "section_heading": record.section_heading,
                        "section_id": record.section_id,
                        "unit_type": record.unit_type,
                        "unit_id": record.unit_id,
                        "subsection_id": record.subsection_id,
                        "paragraph_id": record.paragraph_id,
                        "source_file": record.source_file,
                        "source_path": record.source_path,
                        "chunk_index": record.chunk_index,
                        "document_aliases": record.document_aliases,
                        "text": record.text,
                    }
                ),
                embedding=record.embedding,
            )
            for record in records
        ]
        return search_embedded_entries(entries, query, embedder, top_k=top_k, mode=mode)
