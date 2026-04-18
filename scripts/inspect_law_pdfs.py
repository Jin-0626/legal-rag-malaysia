"""Inspect discovered law PDFs using the modular scaffold settings."""

from legal_rag.config.settings import build_settings
from legal_rag.ingestion.pdf_sources import discover_law_pdfs


def main() -> None:
    settings = build_settings()
    sources = discover_law_pdfs(settings.raw_law_pdfs_dir)
    print(f"Discovered {len(sources)} law PDF(s) in {settings.raw_law_pdfs_dir}")
    for source in sources:
        print(f"- {source.document_id}: {source.path}")


if __name__ == "__main__":
    main()
