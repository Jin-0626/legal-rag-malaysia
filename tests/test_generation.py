from legal_rag.chunking.models import Chunk
from legal_rag.generation.grounded import GroundedAnswerGenerator
from legal_rag.retrieval.in_memory import RetrievalResult


def _result(
    *,
    chunk_id: str,
    document_id: str,
    section_heading: str,
    section_id: str,
    text: str,
    score: float,
    subsection_id: str | None = None,
    paragraph_id: str | None = None,
    source_path: str = "data/raw_law_pdfs/pdpa.pdf",
) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            section_heading=section_heading,
            section_id=section_id,
            subsection_id=subsection_id,
            paragraph_id=paragraph_id,
            text=text,
            source_path=source_path,
        ),
        score=score,
    )


def test_grounded_answer_generator_supported_query() -> None:
    generator = GroundedAnswerGenerator()
    retrieved = [
        _result(
            chunk_id="pdpa_2010:6:0",
            document_id="pdpa_2010",
            section_heading="Section 6 General Principle",
            section_id="6",
            subsection_id="1",
            text="A data user shall not process personal data unless this Act permits it.",
            score=0.93,
        )
    ]

    answer = generator.answer("What does Section 6 require?", retrieved)

    assert answer.grounded is True
    assert answer.citations == ["pdpa_2010, Section 6(1)"]
    assert "Direct Answer:" in answer.answer
    assert "Legal Basis:" in answer.answer
    assert "Practical Meaning:" in answer.answer
    assert "Important Limits:" in answer.answer
    assert "Sources:" in answer.answer
    assert "not legal advice" in answer.answer


def test_grounded_answer_generator_unsupported_query() -> None:
    generator = GroundedAnswerGenerator()

    answer = generator.answer("What is the penalty under a different Act?", [])

    assert answer.grounded is False
    assert answer.citations == []
    assert "Direct Answer:" in answer.answer
    assert "Legal Basis:" in answer.answer
    assert "No retrieved legal provision supports an answer to this question." in answer.answer
    assert "Important Limits:" in answer.answer
    assert "This response is limited by the absence of supporting retrieved evidence." in answer.answer


def test_grounded_answer_generator_ambiguous_query() -> None:
    generator = GroundedAnswerGenerator()
    retrieved = [
        _result(
            chunk_id="pdpa_2010:6:0",
            document_id="pdpa_2010",
            section_heading="Section 6 General Principle",
            section_id="6",
            text="Personal data shall not be processed without consent unless permitted by the Act.",
            score=0.81,
        ),
        _result(
            chunk_id="pdpa_2010:7:0",
            document_id="pdpa_2010",
            section_heading="Section 7 Notice and Choice Principle",
            section_id="7",
            text="Written notice shall be given to the data subject about the purpose of collection.",
            score=0.78,
        ),
    ]

    answer = generator.answer("What must happen before processing personal data?", retrieved)

    assert answer.grounded is False
    assert answer.citations == ["pdpa_2010, Section 6", "pdpa_2010, Section 7"]
    assert "Direct Answer:" in answer.answer
    assert "The retrieved sources point to more than one plausible provision" in answer.answer
    assert "Legal Basis:" in answer.answer
    assert "Important Limits:" in answer.answer


def test_grounded_answer_generator_supports_multiple_evidence_references() -> None:
    generator = GroundedAnswerGenerator()
    retrieved = [
        _result(
            chunk_id="pdpa_2010:6:0",
            document_id="pdpa_2010",
            section_heading="Section 6 General Principle",
            section_id="6",
            subsection_id="1",
            text="A data user shall not process personal data unless this Act permits it.",
            score=0.93,
        ),
        _result(
            chunk_id="pdpa_2010:7:0",
            document_id="pdpa_2010",
            section_heading="Section 7 Notice and Choice Principle",
            section_id="7",
            subsection_id="1",
            paragraph_id="a",
            text="Written notice shall identify the purpose of collection.",
            score=0.74,
        ),
        _result(
            chunk_id="pdpa_2010:7:1",
            document_id="pdpa_2010",
            section_heading="Section 7 Notice and Choice Principle",
            section_id="7",
            subsection_id="1",
            paragraph_id="b",
            text="The notice shall identify the class of third parties to whom disclosure may be made.",
            score=0.68,
        ),
    ]

    answer = generator.answer("What retrieved context speaks to lawful processing and notice?", retrieved)

    assert answer.grounded is True
    assert answer.citations == [
        "pdpa_2010, Section 6(1)",
        "pdpa_2010, Section 7(1)(a)",
        "pdpa_2010, Section 7(1)(b)",
    ]
    assert "Direct Answer:" in answer.answer
    assert "Legal Basis:" in answer.answer
    assert "The closest retrieved provision is pdpa_2010, Section 6(1)." in answer.answer
    assert "Additional supporting sources:" not in answer.answer
    assert "pdpa_2010, Section 7(1)(a)" in answer.answer
    assert "pdpa_2010, Section 7(1)(b)" in answer.answer
