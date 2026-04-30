import json
from pathlib import Path

from legal_rag.evaluation import build_gold_set_v2_candidates, format_gold_set_summary, load_unit_records


def test_gold_set_generator_builds_balanced_candidates_with_required_fields(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    records = [
        {
            "chunk_id": "employment:2:0",
            "document_id": "employment",
            "act_title": "Employment Act 1955",
            "act_number": "Act 265",
            "section_heading": "Section 2 Interpretation",
            "section_id": "2",
            "unit_type": "section",
            "unit_id": "2",
            "subsection_id": None,
            "paragraph_id": None,
            "source_file": "Akta Kerja 1955 (Akta 265).pdf",
            "source_path": "data/raw_law_pdfs/Akta Kerja 1955 (Akta 265).pdf",
            "chunk_index": 0,
            "document_aliases": ["Employment Act 1955", "Akta Kerja 1955", "Act 265"],
            "text": "Section 2 Interpretation\nIn this Act, unless the context otherwise requires...",
        },
        {
            "chunk_id": "employment:4:1",
            "document_id": "employment",
            "act_title": "Employment Act 1955",
            "act_number": "Act 265",
            "section_heading": "Section 4 Appeals",
            "section_id": "4",
            "unit_type": "section",
            "unit_id": "4",
            "subsection_id": None,
            "paragraph_id": None,
            "source_file": "Akta Kerja 1955 (Akta 265).pdf",
            "source_path": "data/raw_law_pdfs/Akta Kerja 1955 (Akta 265).pdf",
            "chunk_index": 1,
            "document_aliases": ["Employment Act 1955", "Akta Kerja 1955", "Act 265"],
            "text": "PART I\nSection 4 Appeals\nAn employee may appeal.",
        },
        {
            "chunk_id": "pdpa:2:0",
            "document_id": "pdpa",
            "act_title": "Personal Data Protection Act 2010",
            "act_number": "Act 709",
            "section_heading": "Section 2 Application",
            "section_id": "2",
            "unit_type": "section",
            "unit_id": "2",
            "subsection_id": None,
            "paragraph_id": None,
            "source_file": "personal-data-protection-act-2010.pdf",
            "source_path": "data/raw_law_pdfs/personal-data-protection-act-2010.pdf",
            "chunk_index": 0,
            "document_aliases": ["Personal Data Protection Act 2010", "PDPA", "Act 709"],
            "text": "Section 2 Application\nThis Act applies to any person who processes personal data.",
        },
        {
            "chunk_id": "constitution:8:0",
            "document_id": "constitution",
            "act_title": "Federal Constitution",
            "act_number": "",
            "section_heading": "Article 8 Equality",
            "section_id": "8",
            "unit_type": "article",
            "unit_id": "8",
            "subsection_id": None,
            "paragraph_id": None,
            "source_file": "Federal Constitution (Reprint 2020).pdf",
            "source_path": "data/raw_law_pdfs/Federal Constitution (Reprint 2020).pdf",
            "chunk_index": 0,
            "document_aliases": ["Federal Constitution", "Constitution of Malaysia"],
            "text": "Part II\nArticle 8 Equality\nAll persons are equal before the law.",
        },
        {
            "chunk_id": "perlembagaan:8:0",
            "document_id": "perlembagaan",
            "act_title": "Perlembagaan Persekutuan",
            "act_number": "",
            "section_heading": "Perkara 8 Kesamarataan",
            "section_id": "8",
            "unit_type": "perkara",
            "unit_id": "8",
            "subsection_id": None,
            "paragraph_id": None,
            "source_file": "Perlembagaan Persekutuan (Cetakan Semula 2020).pdf",
            "source_path": "data/raw_law_pdfs/Perlembagaan Persekutuan (Cetakan Semula 2020).pdf",
            "chunk_index": 0,
            "document_aliases": ["Perlembagaan Persekutuan"],
            "text": "Bahagian II\nPerkara 8 Kesamarataan\nSemua orang adalah sama rata di sisi undang-undang.",
        },
    ]
    file_path = processed_dir / "gold-test.jsonl"
    file_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    units = load_unit_records(processed_dir)
    candidates = build_gold_set_v2_candidates(processed_dir)
    summary = format_gold_set_summary(candidates)

    assert len(units) == 5
    assert any(candidate.category == "direct_lookup" for candidate in candidates)
    assert any(candidate.category == "definition" for candidate in candidates)
    assert any(candidate.category == "capability" for candidate in candidates)
    assert any(candidate.category == "hierarchy" for candidate in candidates)
    assert any(candidate.category == "bilingual" for candidate in candidates)
    assert any(candidate.category == "negative" for candidate in candidates)
    assert all(candidate.generation_method for candidate in candidates)
    assert all(candidate.language in {"en", "ms"} for candidate in candidates)
    assert "Gold Set V2 Candidate Summary" in summary
    assert "Counts By Category" in summary
