from pathlib import Path

from legal_rag.chunking.models import Chunk
from legal_rag.ingestion.corpus_rebuild import (
    collect_document_metadata_issues,
    compare_corpus_snapshots,
    snapshot_processed_corpus,
    validate_graph_consistency,
    write_corpus_report,
)


def _make_chunk(
    *,
    document_id: str = "employment_act_1955",
    act_title: str = "Employment Act 1955",
    source_file: str = "Akta Kerja 1955 (Akta 265).pdf",
    unit_type: str = "section",
    unit_id: str = "10",
    chunk_index: int = 0,
    text: str = "Section 10 Contracts to be in writing.\n(1) A contract of service shall be in writing.",
) -> Chunk:
    return Chunk(
        chunk_id=f"{document_id}:{unit_id}:{chunk_index}",
        document_id=document_id,
        section_heading=f"Section {unit_id} Heading",
        section_id=unit_id,
        subsection_id=None,
        paragraph_id=None,
        text=text,
        source_path=f"data/raw_law_pdfs/{source_file}",
        act_title=act_title,
        act_number="Act 265",
        source_file=source_file,
        chunk_index=chunk_index,
        unit_type=unit_type,
        unit_id=unit_id,
        document_aliases=("Employment Act 1955", "Akta Kerja 1955", "Act 265"),
    )


def test_collect_document_metadata_issues_flags_noisy_titles_and_mixed_units() -> None:
    pdf_path = Path("data/raw_law_pdfs/Act-333-English.pdf")
    chunks = [
        _make_chunk(act_title="As at 1 January 2023", unit_type="section", unit_id="1"),
        _make_chunk(act_title="As at 1 January 2023", unit_type="article", unit_id="2"),
    ]

    issues = collect_document_metadata_issues(pdf_path, chunks)

    assert any("title-noise" in issue for issue in issues)
    assert any("Mixed unit types" in issue for issue in issues)


def test_snapshot_processed_corpus_counts_jsonl_exports(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    report_path = processed_dir / "sample.jsonl"
    write_corpus_report(
        {
            "chunk_id": "not-used",
        },
        tmp_path / "dummy.json",
    )
    report_path.write_text(
        "\n".join(
            [
                '{"chunk_id":"doc:1:0","document_id":"doc","act_title":"Act","act_number":"Act 1","section_heading":"Section 1","section_id":"1","unit_type":"section","unit_id":"1","subsection_id":null,"paragraph_id":null,"source_file":"doc.pdf","source_path":"data/raw_law_pdfs/doc.pdf","chunk_index":0,"document_aliases":["Act"],"text":"Section 1 text"}',
                '{"chunk_id":"doc:2:1","document_id":"doc","act_title":"Act","act_number":"Act 1","section_heading":"Section 2","section_id":"2","unit_type":"section","unit_id":"2","subsection_id":null,"paragraph_id":null,"source_file":"doc.pdf","source_path":"data/raw_law_pdfs/doc.pdf","chunk_index":1,"document_aliases":["Act"],"text":"Section 2 text"}',
            ]
        ),
        encoding="utf-8",
    )

    snapshot = snapshot_processed_corpus(processed_dir)

    assert snapshot["document_count"] == 1
    assert snapshot["total_chunks"] == 2
    assert snapshot["documents"][0]["chunk_count"] == 2


def test_compare_corpus_snapshots_reports_chunk_deltas() -> None:
    previous = {
        "document_count": 1,
        "total_chunks": 2,
        "documents": [
            {"file_name": "doc.jsonl", "chunk_count": 2, "unit_types": ["section"]},
        ],
        "other_files": ["legacy_chunks.json"],
    }
    current = [
        {"file_name": "doc.jsonl", "chunk_count": 3, "unit_types": ["section"]},
        {"file_name": "new.jsonl", "chunk_count": 1, "unit_types": ["article"]},
    ]

    comparison = compare_corpus_snapshots(previous, current)

    assert comparison["added_documents"] == ["new.jsonl"]
    assert comparison["removed_documents"] == []
    assert comparison["changed_documents"][0]["previous_chunk_count"] == 2
    assert comparison["changed_documents"][0]["current_chunk_count"] == 3


def test_validate_graph_consistency_reports_no_orphans_for_simple_chunks() -> None:
    chunks = [
        _make_chunk(unit_id="1", chunk_index=0, text="Part I\nSection 1 Preliminary\n(1) This Act applies."),
        _make_chunk(unit_id="2", chunk_index=1, text="Section 2 Interpretation\n(1) In this Act..."),
    ]

    report = validate_graph_consistency(chunks)

    assert report["document_count"] == 1
    assert report["unit_node_count"] == 2
    assert report["orphan_edge_count"] == 0


def test_validate_graph_consistency_treats_unmaterialized_insert_target_as_informational() -> None:
    principal_chunk = Chunk(
        chunk_id="personal-data-protection-act-2010:43:0",
        document_id="personal-data-protection-act-2010",
        section_heading="Section 43 Access to personal data",
        section_id="43",
        subsection_id=None,
        paragraph_id=None,
        text="Section 43 Access to personal data\n43. A data subject shall be given access to personal data.",
        source_path="data/raw_law_pdfs/personal-data-protection-act-2010.pdf",
        act_title="Personal Data Protection Act 2010",
        act_number="Act 709",
        source_file="personal-data-protection-act-2010.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="43",
        document_aliases=("Personal Data Protection Act 2010", "PDPA", "Act 709"),
    )
    amendment_chunk = Chunk(
        chunk_id="act-a1727:9:0",
        document_id="act-a1727",
        section_heading="Section 9 New section 43a",
        section_id="9",
        subsection_id=None,
        paragraph_id=None,
        text="Section 9 New section 43a\n9. The principal Act is amended by inserting after section 43 the following section:\n43a. Rights to data portability.",
        source_path="data/raw_law_pdfs/Act-A1727.pdf",
        act_title="Personal Data Protection (Amendment) Act 2024",
        act_number="Act A1727",
        source_file="Act-A1727.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="9",
        document_aliases=("Personal Data Protection (Amendment) Act 2024", "PDPA Amendment Act 2024", "Act A1727"),
    )

    report = validate_graph_consistency([principal_chunk, amendment_chunk])

    assert report["orphan_edge_count"] == 0
    assert report["informational_edge_count"] == 1
    assert report["informational_edges"][0]["edge_type"] == "INSERTS_UNMATERIALIZED_TARGET"
