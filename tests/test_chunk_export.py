import json
from pathlib import Path

import fitz

from legal_rag.ingestion import export_chunks_to_jsonl, ingest_law_pdf_to_chunks


def _create_sample_law_pdf(pdf_path: Path, page_texts: list[str]) -> None:
    document = fitz.open()
    try:
        for page_text in page_texts:
            page = document.new_page()
            page.insert_text((72, 72), page_text, fontsize=12)
        document.save(pdf_path)
    finally:
        document.close()


def test_ingest_law_pdf_to_chunks_and_export_jsonl(tmp_path: Path) -> None:
    pdf_path = tmp_path / "personal-data-protection-act-2010.pdf"
    _create_sample_law_pdf(
        pdf_path,
        [
            "\n".join(
                [
                    "Personal Data Protection Act 2010",
                    "Act 709",
                    "Part II",
                    "PERSONAL DATA PROTECTION",
                    "General Principle",
                    "6. (1) A data user shall not process personal data unless the data subject has given consent.",
                    "(a) the data subject has given consent to the processing of the personal data; or",
                    "(b) the processing is necessary for compliance with a legal obligation.",
                    "(2) Notwithstanding subsection (1), a data user may process personal data if the processing is necessary for the performance of a contract.",
                ]
            )
        ],
    )

    chunks = ingest_law_pdf_to_chunks(pdf_path, max_words=35, overlap_words=8)

    assert chunks
    assert chunks[0].act_title == "Personal Data Protection Act 2010"
    assert chunks[0].act_number == "Act 709"
    assert chunks[0].section_id == "6"
    assert chunks[0].source_file == pdf_path.name

    output_path = tmp_path / "processed" / "pdpa.jsonl"
    export_chunks_to_jsonl(chunks, output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert lines
    first_record = json.loads(lines[0])
    assert first_record["act_title"] == "Personal Data Protection Act 2010"
    assert first_record["act_number"] == "Act 709"
    assert first_record["section_id"] == "6"
    assert first_record["source_file"] == "personal-data-protection-act-2010.pdf"
    assert isinstance(first_record["chunk_index"], int)
    assert any(json.loads(line)["subsection_id"] == "1" for line in lines)
