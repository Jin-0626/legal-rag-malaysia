"""Rebuild the processed legal corpus from raw PDFs and emit a validation report."""

from __future__ import annotations

from legal_rag.config.settings import build_settings
from legal_rag.ingestion import rebuild_processed_corpus, write_corpus_report


def main() -> None:
    settings = build_settings()
    report = rebuild_processed_corpus(settings)
    output_path = settings.data_dir / "processed_corpus_report.json"
    write_corpus_report(report, output_path)

    print(f"Rebuilt processed corpus from {report['raw_pdf_count']} raw PDF(s).")
    print(f"Processed documents: {report['processed_document_count']}")
    print(f"Total chunks: {report['total_chunks']}")
    print(f"Failures: {len(report['failed_documents'])}")
    print(
        "Leakage summary: "
        + ", ".join(
            f"{entry['file_name']}={entry['leakage_count']}"
            for entry in report["leakage_summary"]
        )
    )
    print(f"Metadata issue documents: {len(report['metadata_issues'])}")
    print(f"Graph orphan edges: {report['graph_consistency']['orphan_edge_count']}")
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
