from legal_rag.chunking.section_chunker import (
    chunk_legal_text,
    chunk_section_text,
    find_section_boundary_leaks,
)


def test_chunk_legal_text_detects_section_subsection_and_paragraph_metadata() -> None:
    legal_text = """
Section 6 General Principle
(1) A data user shall not process personal data unless this Act permits it.
(a) the data subject has given consent to the processing; and
(b) the processing is necessary for compliance with a legal obligation.
(2) This section applies subject to any regulations made under this Act.
""".strip()

    chunks = chunk_legal_text(
        document_id="pdpa_2010",
        text=legal_text,
        source_path="data/raw_law_pdfs/pdpa.pdf",
        act_title="Personal Data Protection Act 2010",
        act_number="Act 709",
        max_words=30,
        overlap_words=8,
    )

    assert len(chunks) == 2
    assert chunks[0].section_heading == "Section 6 General Principle"
    assert chunks[0].section_id == "6"
    assert chunks[0].subsection_id == "1"
    assert chunks[0].paragraph_id is None
    assert chunks[0].act_title == "Personal Data Protection Act 2010"
    assert chunks[0].act_number == "Act 709"
    assert chunks[0].source_file == "pdpa.pdf"
    assert chunks[0].chunk_index == 0
    assert chunks[0].unit_type == "section"
    assert chunks[0].unit_id == "6"
    assert "(a) the data subject has given consent to the processing; and" in chunks[0].text
    assert "(b) the processing is necessary for compliance with a legal obligation." in chunks[0].text

    assert chunks[1].section_id == "6"
    assert chunks[1].subsection_id is None
    assert chunks[1].chunk_index == 1
    assert "(2) This section applies subject to any regulations made under this Act." in chunks[1].text


def test_chunk_legal_text_maintains_overlap_across_legal_units() -> None:
    legal_text = """
Section 16 Retention Principle
(1) Personal data shall not be kept longer than is necessary.
(2) A data user shall take reasonable steps to ensure compliance with retention duties.
(3) A data user shall dispose of personal data securely when no longer required.
""".strip()

    chunks = chunk_legal_text(
        document_id="pdpa_2010",
        text=legal_text,
        source_path="data/raw_law_pdfs/pdpa.pdf",
        act_title="Personal Data Protection Act 2010",
        act_number="Act 709",
        max_words=18,
        overlap_words=8,
    )

    assert len(chunks) >= 2
    assert "(1) Personal data shall not be kept longer than is necessary." in chunks[0].text
    assert "(1) Personal data shall not be kept longer than is necessary." in chunks[1].text
    assert chunks[0].section_id == "16"
    assert chunks[1].section_id == "16"


def test_chunk_section_text_preserves_clause_integrity_for_realistic_legal_style_text() -> None:
    section_body = """
(1) Where personal data is collected, the notice shall state the purpose of collection.
(a) The notice shall be in both Bahasa Malaysia and English where required.
(b) The notice shall identify the class of third parties to whom disclosure may be made.
""".strip()

    chunks = chunk_section_text(
        document_id="pdpa_2010",
        section_heading="Section 7 Notice and Choice Principle",
        text=section_body,
        source_path="data/raw_law_pdfs/pdpa.pdf",
        act_title="Personal Data Protection Act 2010",
        act_number="Act 709",
        max_words=24,
        overlap_words=6,
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.section_heading == "Section 7 Notice and Choice Principle"
    assert chunk.section_id == "7"
    assert chunk.subsection_id == "1"
    assert chunk.paragraph_id is None
    assert "(a) The notice shall be in both Bahasa Malaysia and English where required." in chunk.text
    assert "(b) The notice shall identify the class of third parties to whom disclosure may be made." in chunk.text
    assert isinstance(chunk.section_id, str)
    assert chunk.subsection_id is None or isinstance(chunk.subsection_id, str)
    assert chunk.paragraph_id is None or isinstance(chunk.paragraph_id, str)
    assert chunk.text.endswith(".")


def test_chunk_section_text_attaches_flat_serializable_metadata() -> None:
    chunks = chunk_section_text(
        document_id="contracts_act_1950",
        section_heading="Section 10 What Agreements Are Contracts",
        text="(1) All agreements are contracts if they are made by the free consent of parties competent to contract.",
        source_path="data/raw_law_pdfs/contracts_act_1950.pdf",
        act_title="Contracts Act 1950",
        act_number="Act 136",
        max_words=40,
        overlap_words=8,
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.act_title == "Contracts Act 1950"
    assert chunk.act_number == "Act 136"
    assert chunk.section_id == "10"
    assert chunk.subsection_id == "1"
    assert chunk.paragraph_id is None
    assert chunk.source_file == "contracts_act_1950.pdf"
    assert chunk.chunk_index == 0
    assert chunk.unit_type == "section"
    assert chunk.unit_id == "10"
    assert isinstance(chunk.act_title, str)
    assert isinstance(chunk.act_number, str)
    assert isinstance(chunk.section_id, str)
    assert isinstance(chunk.source_file, str)
    assert isinstance(chunk.chunk_index, int)


def test_chunk_legal_text_detects_numbered_section_format_from_realistic_pdf_text() -> None:
    legal_text = """
Part II
PERSONAL DATA PROTECTION
Division 1
Personal Data Protection Principles
General Principle
6. (1) A data user shall not process personal data unless the data subject has given consent.
(a) the data subject has given consent to the processing of the personal data; or
(b) the processing is necessary for compliance with a legal obligation.
(2) Notwithstanding subsection (1), a data user may process personal data if the processing is necessary for the performance of a contract.
""".strip()

    chunks = chunk_legal_text(
        document_id="pdpa_2010",
        text=legal_text,
        source_path="data/raw_law_pdfs/pdpa.pdf",
        act_title="Personal Data Protection Act 2010",
        act_number="Act 709",
        max_words=40,
        overlap_words=10,
    )

    assert len(chunks) >= 1
    first_chunk = chunks[0]
    assert first_chunk.section_heading == "Section 6 General Principle"
    assert first_chunk.section_id == "6"
    assert first_chunk.subsection_id == "1"
    assert first_chunk.act_title == "Personal Data Protection Act 2010"
    assert first_chunk.act_number == "Act 709"
    assert first_chunk.source_file == "pdpa.pdf"
    assert "6. (1) A data user shall not process personal data unless the data subject has given consent." in first_chunk.text


def test_chunk_legal_text_preserves_clause_lines_for_numbered_section_format() -> None:
    legal_text = """
General Principle
6. (1) A data user shall not—
(a) in the case of personal data other than sensitive personal data, process personal data about a data subject unless the data subject has given consent to the processing of the personal data; or
(b) in the case of sensitive personal data, process sensitive personal data about a data subject except in accordance with section 40.
""".strip()

    chunks = chunk_legal_text(
        document_id="pdpa_2010",
        text=legal_text,
        source_path="data/raw_law_pdfs/pdpa.pdf",
        act_title="Personal Data Protection Act 2010",
        act_number="Act 709",
        max_words=80,
        overlap_words=12,
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert "(a) in the case of personal data other than sensitive personal data, process personal data about a data subject unless the data subject has given consent to the processing of the personal data; or" in chunk.text
    assert "(b) in the case of sensitive personal data, process sensitive personal data about a data subject except in accordance with section 40." in chunk.text
    assert "consent to the processing of the personal data; or" in chunk.text


def test_chunk_legal_text_does_not_cross_section_boundaries_when_sections_fit_together() -> None:
    legal_text = """
Section 1 Short title and commencement
1. (1) This Act may be cited as the Personal Data Protection Act 2010.
(2) This Act comes into operation on a date to be appointed by the Minister.
Section 2 Application
2. (1) This Act applies to any person who processes personal data.
""".strip()

    chunks = chunk_legal_text(
        document_id="pdpa_2010",
        text=legal_text,
        source_path="data/raw_law_pdfs/pdpa.pdf",
        act_title="Personal Data Protection Act 2010",
        act_number="Act 709",
        max_words=80,
        overlap_words=12,
    )

    assert len(chunks) == 2
    assert chunks[0].section_id == "1"
    assert "Section 2 Application" not in chunks[0].text
    assert chunks[1].section_id == "2"
    assert "Section 1 Short title and commencement" not in chunks[1].text


def test_find_section_boundary_leaks_reports_cross_section_heading_lines() -> None:
    chunks = chunk_legal_text(
        document_id="pdpa_2010",
        text="""
Section 1 Short title and commencement
1. (1) This Act may be cited as the Personal Data Protection Act 2010.
Section 2 Application
2. (1) This Act applies to any person who processes personal data.
""".strip(),
        source_path="data/raw_law_pdfs/pdpa.pdf",
        act_title="Personal Data Protection Act 2010",
        act_number="Act 709",
        max_words=80,
        overlap_words=12,
    )

    leaks = find_section_boundary_leaks(chunks)

    assert leaks == []


def test_chunk_legal_text_detects_article_based_constitutional_text() -> None:
    legal_text = """
Part II
FUNDAMENTAL LIBERTIES
Liberty of the person
5. (1) No person shall be deprived of his life or personal liberty save in accordance with law.
(2) Where complaint is made to a High Court that a person is being unlawfully detained, the Court shall inquire into the complaint.
""".strip()

    chunks = chunk_legal_text(
        document_id="federal_constitution",
        text=legal_text,
        source_path="data/raw_law_pdfs/Federal Constitution.pdf",
        act_title="Federal Constitution",
        max_words=35,
        overlap_words=8,
        unit_type_hint="article",
    )

    assert len(chunks) == 2
    assert chunks[0].unit_type == "article"
    assert chunks[0].unit_id == "5"
    assert chunks[0].section_heading == "Article 5 Liberty of the person"
    assert chunks[0].section_id == "5"


def test_chunk_legal_text_detects_perkara_based_malay_constitutional_text() -> None:
    legal_text = """
Bahagian II
KEBEBASAN ASASI
Kebebasan diri
5. (1) Tiada seorang pun boleh diambil nyawanya atau dilucutkan kebebasan dirinya kecuali mengikut undang-undang.
(2) Jika pengaduan dibuat kepada Mahkamah Tinggi, mahkamah itu hendaklah menyiasat pengaduan itu.
""".strip()

    chunks = chunk_legal_text(
        document_id="perlembagaan_persekutuan",
        text=legal_text,
        source_path="data/raw_law_pdfs/Perlembagaan Persekutuan.pdf",
        act_title="Perlembagaan Persekutuan",
        max_words=35,
        overlap_words=8,
        unit_type_hint="perkara",
    )

    assert len(chunks) >= 1
    assert chunks[0].unit_type == "perkara"
    assert chunks[0].unit_id == "5"
    assert chunks[0].section_heading == "Perkara 5 Kebebasan diri"


def test_chunk_legal_text_keeps_overlap_within_same_subsection_when_possible() -> None:
    legal_text = """
Section 40 Processing of sensitive personal data
(1) A data user shall not process sensitive personal data unless one of the statutory grounds applies.
(a) the data subject has given explicit consent.
(b) the processing is necessary for legal proceedings.
(2) This section does not prevent processing required by law.
""".strip()

    chunks = chunk_legal_text(
        document_id="pdpa_2010",
        text=legal_text,
        source_path="data/raw_law_pdfs/pdpa.pdf",
        act_title="Personal Data Protection Act 2010",
        act_number="Act 709",
        max_words=22,
        overlap_words=8,
    )

    assert len(chunks) >= 2
    assert chunks[0].section_id == "40"
    assert chunks[1].section_id == "40"
    assert "(1) A data user shall not process sensitive personal data unless one of the statutory grounds applies." in chunks[1].text
    assert all(chunk.section_id == "40" for chunk in chunks)


def test_oversized_constitutional_unit_falls_back_without_crossing_top_level_boundaries() -> None:
    long_clause = " ".join(["liberty"] * 120)
    legal_text = f"""
Liberty of the person
5. (1) {long_clause}
Right against slavery
6. (1) Slavery and forced labour are prohibited.
""".strip()

    chunks = chunk_legal_text(
        document_id="federal_constitution",
        text=legal_text,
        source_path="data/raw_law_pdfs/Federal Constitution.pdf",
        act_title="Federal Constitution",
        max_words=40,
        overlap_words=8,
        unit_type_hint="article",
    )

    article_five_chunks = [chunk for chunk in chunks if chunk.section_id == "5"]
    article_six_chunks = [chunk for chunk in chunks if chunk.section_id == "6"]
    assert len(article_five_chunks) > 1
    assert len(article_six_chunks) == 1
    assert all("Article 6" not in chunk.text for chunk in article_five_chunks)


def test_find_section_boundary_leaks_detects_mixed_article_chunks() -> None:
    from legal_rag.chunking.models import Chunk

    leaks = find_section_boundary_leaks(
        [
            Chunk(
                chunk_id="constitution:5:0",
                document_id="constitution",
                section_heading="Article 5 Liberty of the person",
                section_id="5",
                subsection_id=None,
                paragraph_id=None,
                text="Article 5 Liberty of the person\n5. (1) No person shall be deprived of liberty.\nArticle 6 Slavery and forced labour prohibited\n6. (1) Slavery is prohibited.",
                source_path="data/raw_law_pdfs/Federal Constitution.pdf",
                unit_type="article",
                unit_id="5",
            )
        ]
    )

    assert len(leaks) == 1
    assert leaks[0].assigned_unit_type == "article"
    assert leaks[0].found_unit_type == "article"
    assert leaks[0].found_section_id == "6"


def test_find_section_boundary_leaks_ignores_cross_referenced_article_lines_inside_section_chunk() -> None:
    from legal_rag.chunking.models import Chunk

    leaks = find_section_boundary_leaks(
        [
            Chunk(
                chunk_id="land-code:416:0",
                document_id="land-code",
                section_heading="Section 416 Vesting",
                section_id="416",
                subsection_id=None,
                paragraph_id=None,
                text="Section 416 Vesting\n416. Action may be taken under Article 85 and Article 166 of the Federal Constitution where applicable.",
                source_path="data/raw_law_pdfs/Act 828. National Land Code. Revised 2020.pdf",
                unit_type="section",
                unit_id="416",
            )
        ]
    )

    assert leaks == []


def test_chunk_legal_text_does_not_promote_cross_referenced_article_to_top_level_unit_in_section_document() -> None:
    legal_text = """
Section 416B Vesting in relation to Federal lands
416B. Where any land in Malacca or Penang is occupied under Clause (3) of Article 166 of the Federal Constitution immediately before a statutory vesting takes effect, the State Authority may permit the land to be occupied by the transferee.
416C. Where the whole or part of any alienated land is occupied by the transferee, the transferee's right may be endorsed on the register document of title.
""".strip()

    chunks = chunk_legal_text(
        document_id="national_land_code",
        text=legal_text,
        source_path="data/raw_law_pdfs/Act 828. National Land Code. Revised 2020.pdf",
        act_title="National Land Code",
        act_number="Act 828",
        max_words=120,
        overlap_words=10,
        unit_type_hint="section",
    )

    assert chunks
    assert all(chunk.unit_type == "section" for chunk in chunks)
    assert chunks[0].section_id == "416B"
