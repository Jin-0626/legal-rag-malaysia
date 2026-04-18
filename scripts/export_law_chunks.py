"""Export legal-aware chunks from law PDFs into JSONL files."""

from __future__ import annotations

from pathlib import Path

from legal_rag.config.settings import build_settings
from legal_rag.ingestion import export_chunks_to_jsonl, ingest_law_pdf_to_chunks


def main() -> None:
    settings = build_settings()
    pdf_paths = sorted(settings.raw_law_pdfs_dir.glob("*.pdf"))

    for pdf_path in pdf_paths:
        chunks = ingest_law_pdf_to_chunks(
            pdf_path,
            max_words=settings.chunk_size_words,
            overlap_words=settings.chunk_overlap_words,
        )
        output_path = settings.processed_dir / f"{pdf_path.stem}.jsonl"
        export_chunks_to_jsonl(chunks, output_path)
        print(f"{pdf_path.name}: exported {len(chunks)} chunk(s) -> {output_path.name}")


if __name__ == "__main__":
    main()
