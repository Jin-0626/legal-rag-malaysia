"""Index exported law chunk JSONL files into a JSONL vector store."""

from __future__ import annotations

from pathlib import Path

from legal_rag.config.settings import build_settings
from legal_rag.embeddings.embedder import OllamaEmbedder
from legal_rag.retrieval.vector_store import JsonlVectorStore, load_chunk_records


def main() -> None:
    settings = build_settings()
    embedder = OllamaEmbedder()
    processed_files = sorted(
        path
        for path in settings.processed_dir.glob("*.jsonl")
        if not path.stem.endswith(".smoke")
    )
    all_chunks = []

    for chunk_file in processed_files:
        chunks = load_chunk_records(chunk_file)
        if not chunks:
            print(f"{chunk_file.name}: skipped (no chunks)")
            continue

        all_chunks.extend(chunks)
        output_path = settings.embeddings_dir / f"{chunk_file.stem}.vectors.jsonl"
        store = JsonlVectorStore(output_path)
        count = store.index_chunks(chunks, embedder)
        print(f"{chunk_file.name}: indexed {count} chunk(s) -> {output_path.name}")

    if all_chunks:
        corpus_path = settings.embeddings_dir / "legal-corpus.vectors.jsonl"
        corpus_store = JsonlVectorStore(corpus_path)
        corpus_count = corpus_store.index_chunks(all_chunks, embedder)
        print(f"combined corpus: indexed {corpus_count} chunk(s) -> {corpus_path.name}")


if __name__ == "__main__":
    main()
