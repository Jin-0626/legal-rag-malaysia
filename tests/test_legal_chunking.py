from legal_rag.chunking.section_chunker import chunk_legal_text, chunk_section_text


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
