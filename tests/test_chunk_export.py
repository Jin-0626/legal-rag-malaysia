import json
from pathlib import Path

import fitz

from legal_rag.ingestion import export_chunks_to_jsonl, ingest_law_pdf_to_chunks
from legal_rag.ingestion.chunk_export import derive_act_title, derive_document_aliases, normalize_law_document_text
from legal_rag.ingestion.parse_pdf import ExtractedPdfPage, LawDocumentText


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
    assert "PDPA" in chunks[0].document_aliases

    output_path = tmp_path / "processed" / "pdpa.jsonl"
    export_chunks_to_jsonl(chunks, output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert lines
    first_record = json.loads(lines[0])
    assert first_record["act_title"] == "Personal Data Protection Act 2010"
    assert first_record["act_number"] == "Act 709"
    assert first_record["section_id"] == "6"
    assert first_record["source_file"] == "personal-data-protection-act-2010.pdf"
    assert "PDPA" in first_record["document_aliases"]
    assert isinstance(first_record["chunk_index"], int)
    assert any(json.loads(line)["subsection_id"] == "1" for line in lines)


def test_normalize_law_document_text_skips_front_matter_until_body_start() -> None:
    document = LawDocumentText(
        document_id="employment_act_1955",
        title="Employment Act 1955",
        source_path="data/raw_law_pdfs/Akta Kerja 1955 (Akta 265).pdf",
        pages=[
            ExtractedPdfPage(
                page_number=1,
                lines=[
                    "LAWS OF MALAYSIA",
                    "ONLINE VERSION OF UPDATED",
                    "TEXT OF REPRINT",
                    "EMPLOYMENT ACT 1955",
                ],
                text="",
            ),
            ExtractedPdfPage(
                page_number=13,
                lines=[
                    "An Act relating to employment.",
                    "PART I",
                    "PRELIMINARY",
                    "1. (1) This Act may be cited as the Employment Act 1955.",
                    "(2) This Act applies throughout Malaysia.",
                ],
                text="",
            ),
        ],
        full_text="",
    )

    normalized = normalize_law_document_text(document)

    assert "ONLINE VERSION OF UPDATED" not in normalized
    assert "PART I" in normalized
    assert "PRELIMINARY" in normalized
    assert "1. (1) This Act may be cited as the Employment Act 1955." in normalized
    assert derive_act_title(document) == "Employment Act 1955"
    assert derive_document_aliases(document)[:2] == ("Employment Act 1955", "Akta Kerja 1955")


def test_normalize_law_document_text_skips_constitution_contents_and_note_pages() -> None:
    document = LawDocumentText(
        document_id="federal_constitution",
        title="Federal Constitution",
        source_path="data/raw_law_pdfs/Federal Constitution (Reprint 2020).pdf",
        pages=[
            ExtractedPdfPage(
                page_number=5,
                lines=[
                    "CONTENTS",
                    "ARRANGEMENT OF ARTICLES",
                    "THE FEDERATION",
                    "Article",
                    "1.",
                    "Name, States and territories of the Federation",
                ],
                text="",
            ),
            ExtractedPdfPage(
                page_number=19,
                lines=[
                    "FEDERAL CONSTITUTION",
                    "Part I",
                    "THE FEDERATION",
                    "Name, States and territories of the Federation",
                    "1. (1) The Federation shall be known as Malaysia.",
                    "(2) The States of the Federation shall be Johore and Kedah.",
                ],
                text="",
            ),
            ExtractedPdfPage(
                page_number=21,
                lines=[
                    "FEDERAL CONSTITUTION",
                    "NOTE:",
                    "This note is not part of the authoritative text.",
                ],
                text="",
            ),
        ],
        full_text="",
    )

    normalized = normalize_law_document_text(document)

    assert "ARRANGEMENT OF ARTICLES" not in normalized
    assert "NOTE:" not in normalized
    assert "Part I" in normalized
    assert "1. (1) The Federation shall be known as Malaysia." in normalized


def test_derive_act_title_skips_as_at_front_matter_and_uses_uppercase_title_block() -> None:
    document = LawDocumentText(
        document_id="road_transport_act_1987",
        title="Act-333-English",
        source_path="data/raw_law_pdfs/Act-333-English.pdf",
        pages=[
            ExtractedPdfPage(
                page_number=1,
                lines=[
                    "LAWS OF MALAYSIA",
                    "ONLINE VERSION OF UPDATED",
                    "TEXT OF REPRINT",
                    "Act 333",
                    "ROAD TRANSPORT ACT 1987",
                    "As at 15 October 2023",
                ],
                text="",
            )
        ],
        full_text="",
    )

    assert derive_act_title(document) == "Road Transport Act 1987"
    assert "Road Transport Act 1987" in derive_document_aliases(document)


def test_derive_act_title_handles_bilingual_order_front_page() -> None:
    document = LawDocumentText(
        document_id="pua_376",
        title="PUA 376",
        source_path="data/raw_law_pdfs/PUA 376.pdf",
        pages=[
            ExtractedPdfPage(
                page_number=1,
                lines=[
                    "4 Disember 2024",
                    "4 December 2024",
                    "P.U. (A) 376",
                    "WARTA KERAJAAN PERSEKUTUAN",
                    "FEDERAL GOVERNMENT",
                    "GAZETTE",
                    "PERINTAH GAJI MINIMUM 2024",
                    "MINIMUM WAGES ORDER 2024",
                ],
                text="",
            )
        ],
        full_text="",
    )

    assert derive_act_title(document) == "Minimum Wages Order 2024"
    assert derive_document_aliases(document) == (
        "Minimum Wages Order 2024",
        "Perintah Gaji Minimum 2024",
        "P.U. (A) 376",
    )


def test_normalize_law_document_text_skips_toc_page_and_starts_on_real_sale_of_goods_body() -> None:
    document = LawDocumentText(
        document_id="sale_of_goods_act_1957",
        title="SALE_OF_GOODS_ACT_1957___ACT_382",
        source_path="data/raw_law_pdfs/SALE_OF_GOODS_ACT_1957___ACT_382.pdf",
        pages=[
            ExtractedPdfPage(
                page_number=1,
                lines=[
                    "ACT 382",
                    "SALE OF GOODS ACT 1957",
                    "SECTION",
                    "1.Short title and application",
                    "2.Interpretation",
                    "3.Application of Contracts Act 1950",
                    "Chapter II formation of the contract",
                ],
                text="",
            ),
            ExtractedPdfPage(
                page_number=5,
                lines=[
                    "ACT 382,,/1.Short title and application",
                    "1. Short title and application",
                    "(1) This Act may be cited as the Sale of Goods Act 1957.",
                    "(2) This Act shall apply to West Malaysia.",
                ],
                text="",
            ),
        ],
        full_text="",
    )

    normalized = normalize_law_document_text(document)

    assert "2.Interpretation" not in normalized
    assert "1. Short title and application" in normalized
    assert "(1) This Act may be cited as the Sale of Goods Act 1957." in normalized


def test_normalize_law_document_text_stitches_split_numbered_gazette_provision_start() -> None:
    document = LawDocumentText(
        document_id="minimum_wages_order_2024",
        title="PUA 376",
        source_path="data/raw_law_pdfs/PUA 376.pdf",
        pages=[
            ExtractedPdfPage(
                page_number=2,
                lines=[
                    "Nama dan permulaan kuat kuasa",
                    "1.",
                    "(1) Perintah ini bolehlah dinamakan Perintah Gaji Minimum 2024.",
                    "(2) Perintah ini mula berkuat kuasa pada 1 Februari 2025.",
                ],
                text="",
            ),
        ],
        full_text="",
    )

    normalized = normalize_law_document_text(document)

    assert "1. (1) Perintah ini bolehlah dinamakan Perintah Gaji Minimum 2024." in normalized
    assert "(2) Perintah ini mula berkuat kuasa pada 1 Februari 2025." in normalized


def test_derive_act_title_and_aliases_fix_known_epf_metadata() -> None:
    document = LawDocumentText(
        document_id="epf_act_1991",
        title="F1515447514_MYS43880 2017",
        source_path="data/raw_law_pdfs/F1515447514_MYS43880 2017.pdf",
        pages=[
            ExtractedPdfPage(
                page_number=1,
                lines=[
                    "LAWS OF MALAYSIA",
                    "ONLINE VERSION OF UPDATED",
                    "TEXT OF REPRINT",
                    "Act 452",
                    "EMPOYEES PROVIDENT FUND",
                    "ACT 1991",
                ],
                text="",
            ),
        ],
        full_text="",
    )

    assert derive_act_title(document) == "Employees Provident Fund Act 1991"
    assert derive_document_aliases(document) == (
        "Employees Provident Fund Act 1991",
        "EPF Act 1991",
        "Act 452",
    )
