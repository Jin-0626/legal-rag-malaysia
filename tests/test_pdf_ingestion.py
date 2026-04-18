from pathlib import Path

import fitz
import pytest

from legal_rag.ingestion import PdfIngestionError, extract_law_document_text


def _create_sample_law_pdf(pdf_path: Path, page_texts: list[str]) -> None:
    document = fitz.open()
    try:
        for page_text in page_texts:
            page = document.new_page()
            page.insert_text((72, 72), page_text, fontsize=12)
        document.save(pdf_path)
    finally:
        document.close()


def test_extract_law_document_text_preserves_lines(tmp_path: Path) -> None:
    pdf_path = tmp_path / "Personal Data Protection Act.pdf"
    _create_sample_law_pdf(
        pdf_path,
        [
            "Personal Data Protection Act 2010\nSection 6 General Principle\n"
            "Personal data shall not be processed without consent."
        ],
    )

    extracted = extract_law_document_text(pdf_path)

    assert extracted.document_id == "personal_data_protection_act"
    assert extracted.title == "Personal Data Protection Act"
    assert extracted.source_path == str(pdf_path)
    assert len(extracted.pages) == 1
    assert extracted.pages[0].page_number == 1
    assert extracted.pages[0].lines == [
        "Personal Data Protection Act 2010",
        "Section 6 General Principle",
        "Personal data shall not be processed without consent.",
    ]
    assert "Section 6 General Principle" in extracted.full_text
    assert "\n" in extracted.pages[0].text


def test_extract_law_document_text_raises_for_empty_extraction(tmp_path: Path) -> None:
    pdf_path = tmp_path / "empty-law.pdf"
    _create_sample_law_pdf(pdf_path, ["   "])

    with pytest.raises(PdfIngestionError, match="produced no usable text"):
        extract_law_document_text(pdf_path)


def test_extract_law_document_text_raises_for_unreadable_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "broken-law.pdf"
    pdf_path.write_bytes(b"this is not a valid pdf")

    with pytest.raises(PdfIngestionError, match="Unable to open PDF for extraction"):
        extract_law_document_text(pdf_path)
