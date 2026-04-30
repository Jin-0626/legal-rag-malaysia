from legal_rag.chunking.models import Chunk
from legal_rag.embeddings.embedder import OllamaEmbedder
from legal_rag.retrieval.in_memory import EmbeddingRetriever, SimpleVectorIndex


class FakeTransport:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed(self, *, texts: list[str], model: str) -> list[list[float]]:
        return [self.vectors[text] for text in texts]


def test_embedding_retriever_returns_top_k_embedding_matches_with_traceable_metadata() -> None:
    chunk_a = Chunk(
        chunk_id="pdpa_2010:6:0",
        document_id="pdpa_2010",
        section_heading="Section 6 General Principle",
        section_id="6",
        subsection_id="1",
        paragraph_id="a",
        text="consent lawful basis processing",
        source_path="data/raw_law_pdfs/pdpa.pdf",
    )
    chunk_b = Chunk(
        chunk_id="pdpa_2010:7:0",
        document_id="pdpa_2010",
        section_heading="Section 7 Notice and Choice Principle",
        section_id="7",
        subsection_id="1",
        paragraph_id=None,
        text="notice disclosure purpose statement",
        source_path="data/raw_law_pdfs/pdpa.pdf",
    )
    chunk_c = Chunk(
        chunk_id="contracts_act:10:0",
        document_id="contracts_act",
        section_heading="Section 10 What Agreements Are Contracts",
        section_id="10",
        subsection_id=None,
        paragraph_id=None,
        text="offer acceptance lawful consideration",
        source_path="data/raw_law_pdfs/contracts_act.pdf",
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                "consent lawful basis processing": [1.0, 0.0, 0.0],
                "notice disclosure purpose statement": [0.0, 1.0, 0.0],
                "offer acceptance lawful consideration": [0.6, 0.0, 0.8],
                "consent lawful basis": [0.95, 0.05, 0.0],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="consent lawful basis",
        chunks=[chunk_a, chunk_b, chunk_c],
        top_k=2,
    )

    assert [result.chunk.chunk_id for result in results] == [
        "pdpa_2010:6:0",
        "contracts_act:10:0",
    ]
    assert results[0].score > results[1].score
    assert results[0].chunk.section_id == "6"
    assert results[0].chunk.subsection_id == "1"
    assert results[0].chunk.paragraph_id == "a"
    assert results[0].chunk.source_path == "data/raw_law_pdfs/pdpa.pdf"
    assert results[0].score > 0.0
    assert results[1].chunk.document_id == "contracts_act"
    assert results[1].chunk.source_path == "data/raw_law_pdfs/contracts_act.pdf"
    assert results[0].chunk.unit_type == "section"
    assert results[0].chunk.unit_id == ""


def test_simple_vector_index_returns_empty_for_blank_query() -> None:
    chunk = Chunk(
        chunk_id="pdpa_2010:6:0",
        document_id="pdpa_2010",
        section_heading="Section 6 General Principle",
        section_id="6",
        subsection_id="1",
        paragraph_id=None,
        text="consent lawful basis processing",
        source_path="data/raw_law_pdfs/pdpa.pdf",
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport({"consent lawful basis processing": [1.0, 0.0]}),
    )
    index = SimpleVectorIndex(embedder)
    index.add([chunk])

    results = index.search("   ", top_k=3)

    assert results == []


def test_simple_vector_index_uses_stable_top_k_ordering_for_equal_scores() -> None:
    chunk_a = Chunk(
        chunk_id="pdpa_2010:6:0",
        document_id="pdpa_2010",
        section_heading="Section 6 General Principle",
        section_id="6",
        subsection_id="1",
        paragraph_id=None,
        text="consent basis one",
        source_path="data/raw_law_pdfs/pdpa.pdf",
    )
    chunk_b = Chunk(
        chunk_id="pdpa_2010:6:1",
        document_id="pdpa_2010",
        section_heading="Section 6 General Principle",
        section_id="6",
        subsection_id="2",
        paragraph_id=None,
        text="consent basis two",
        source_path="data/raw_law_pdfs/pdpa.pdf",
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                "consent basis one": [1.0, 0.0],
                "consent basis two": [1.0, 0.0],
                "consent basis query": [1.0, 0.0],
            }
        ),
    )
    index = SimpleVectorIndex(embedder)
    index.add([chunk_b, chunk_a])

    results = index.search("consent basis query", top_k=2)

    assert [result.chunk.chunk_id for result in results] == [
        "pdpa_2010:6:0",
        "pdpa_2010:6:1",
    ]


def test_hybrid_retrieval_prefers_exact_legal_unit_lookup_with_document_hint() -> None:
    article_chunk = Chunk(
        chunk_id="constitution:5:0",
        document_id="constitution",
        section_heading="Article 5 Liberty of the person",
        section_id="5",
        subsection_id=None,
        paragraph_id=None,
        text="No person shall be deprived of his life or personal liberty save in accordance with law.",
        source_path="data/raw_law_pdfs/federal-constitution.pdf",
        act_title="Federal Constitution",
        source_file="Federal Constitution.pdf",
        chunk_index=0,
        unit_type="article",
        unit_id="5",
    )
    section_chunk = Chunk(
        chunk_id="pdpa:5:0",
        document_id="pdpa",
        section_heading="Section 5 Personal data protection principles",
        section_id="5",
        subsection_id=None,
        paragraph_id=None,
        text="The following personal data protection principles apply.",
        source_path="data/raw_law_pdfs/pdpa.pdf",
        act_title="Personal Data Protection Act 2010",
        source_file="pdpa.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="5",
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                article_chunk.text: [0.2, 0.8],
                section_chunk.text: [0.8, 0.2],
                "Article 5 Federal Constitution": [0.7, 0.3],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="Article 5 Federal Constitution",
        chunks=[section_chunk, article_chunk],
        top_k=2,
        mode="hybrid",
    )

    assert [result.chunk.chunk_id for result in results] == ["constitution:5:0"]
    assert results[0].chunk.unit_type == "article"
    assert results[0].chunk.unit_id == "5"


def test_hybrid_retrieval_uses_document_aliases_for_canonical_title_matching() -> None:
    employment_chunk = Chunk(
        chunk_id="employment:2:0",
        document_id="employment",
        section_heading="Section 2 Interpretation",
        section_id="2",
        subsection_id=None,
        paragraph_id=None,
        text="In this Act, employee means a person employed under a contract of service.",
        source_path="data/raw_law_pdfs/Akta Kerja 1955 (Akta 265).pdf",
        act_title="Employment Act 1955",
        source_file="Akta Kerja 1955 (Akta 265).pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="2",
        document_aliases=("Employment Act 1955", "Akta Kerja 1955", "Act 265"),
    )
    other_chunk = Chunk(
        chunk_id="constitution:160:0",
        document_id="constitution",
        section_heading="Article 160 Interpretation",
        section_id="160",
        subsection_id=None,
        paragraph_id=None,
        text="In this Constitution, interpretation provisions define expressions used in the Constitution.",
        source_path="data/raw_law_pdfs/federal-constitution.pdf",
        act_title="Federal Constitution",
        source_file="Federal Constitution.pdf",
        chunk_index=1,
        unit_type="article",
        unit_id="160",
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                employment_chunk.text: [0.2, 0.8],
                other_chunk.text: [1.0, 0.0],
                "What does Akta Kerja 1955 say about interpretation?": [0.8, 0.2],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="What does Akta Kerja 1955 say about interpretation?",
        chunks=[other_chunk, employment_chunk],
        top_k=2,
        mode="hybrid",
    )

    assert results[0].chunk.chunk_id == "employment:2:0"


def test_hybrid_recall_normalizes_appeal_and_appeals_for_heading_match() -> None:
    appeals_chunk = Chunk(
        chunk_id="employment:4:0",
        document_id="employment",
        section_heading="Section 4 Appeals",
        section_id="4",
        subsection_id=None,
        paragraph_id=None,
        text="Any person aggrieved may bring an appeal under this section.",
        source_path="data/raw_law_pdfs/employment.pdf",
        act_title="Employment Act 1955",
        source_file="employment.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="4",
        document_aliases=("Employment Act 1955",),
    )
    distractor = Chunk(
        chunk_id="employment:77:0",
        document_id="employment",
        section_heading="Section 77 Complaints to Director General",
        section_id="77",
        subsection_id=None,
        paragraph_id=None,
        text="The Director General may inquire into complaints made by an employee.",
        source_path="data/raw_law_pdfs/employment.pdf",
        act_title="Employment Act 1955",
        source_file="employment.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="77",
        document_aliases=("Employment Act 1955",),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                appeals_chunk.text: [0.1, 0.9],
                distractor.text: [0.9, 0.1],
                "Who can appeal under the Employment Act 1955?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="Who can appeal under the Employment Act 1955?",
        chunks=[distractor, appeals_chunk],
        top_k=2,
        mode="hybrid",
    )

    assert results[0].chunk.chunk_id == "employment:4:0"


def test_hybrid_recall_expands_apply_query_to_application_heading() -> None:
    application_chunk = Chunk(
        chunk_id="pdpa:2:0",
        document_id="pdpa",
        section_heading="Section 2 Application",
        section_id="2",
        subsection_id=None,
        paragraph_id=None,
        text="This Act applies to any person who processes personal data in respect of commercial transactions.",
        source_path="data/raw_law_pdfs/pdpa.pdf",
        act_title="Personal Data Protection Act 2010",
        source_file="pdpa.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="2",
        document_aliases=("Personal Data Protection Act 2010", "PDPA"),
    )
    distractor = Chunk(
        chunk_id="pdpa:30:0",
        document_id="pdpa",
        section_heading="Section 30 Right of access to personal data",
        section_id="30",
        subsection_id=None,
        paragraph_id=None,
        text="A data subject shall be given access to personal data on request.",
        source_path="data/raw_law_pdfs/pdpa.pdf",
        act_title="Personal Data Protection Act 2010",
        source_file="pdpa.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="30",
        document_aliases=("Personal Data Protection Act 2010", "PDPA"),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                application_chunk.text: [0.2, 0.8],
                distractor.text: [0.9, 0.1],
                "To whom does the PDPA apply in respect of personal data in commercial transactions?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="To whom does the PDPA apply in respect of personal data in commercial transactions?",
        chunks=[distractor, application_chunk],
        top_k=2,
        mode="hybrid",
    )

    assert results[0].chunk.chunk_id == "pdpa:2:0"


def test_hybrid_recall_normalizes_bilingual_annual_leave_query() -> None:
    annual_leave_chunk = Chunk(
        chunk_id="employment:60e:0",
        document_id="employment",
        section_heading="Section 60E Annual leave",
        section_id="60E",
        subsection_id=None,
        paragraph_id=None,
        text="An employee shall be entitled to paid annual leave after twelve months of continuous service.",
        source_path="data/raw_law_pdfs/Akta Kerja 1955 (Akta 265).pdf",
        act_title="Employment Act 1955",
        source_file="Akta Kerja 1955 (Akta 265).pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="60E",
        document_aliases=("Employment Act 1955", "Akta Kerja 1955", "Act 265"),
    )
    distractor = Chunk(
        chunk_id="employment:1:0",
        document_id="employment",
        section_heading="Section 1 Short title and application",
        section_id="1",
        subsection_id=None,
        paragraph_id=None,
        text="This Act may be cited as the Employment Act 1955 and applies throughout Malaysia.",
        source_path="data/raw_law_pdfs/Akta Kerja 1955 (Akta 265).pdf",
        act_title="Employment Act 1955",
        source_file="Akta Kerja 1955 (Akta 265).pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="1",
        document_aliases=("Employment Act 1955", "Akta Kerja 1955", "Act 265"),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                annual_leave_chunk.text: [0.2, 0.8],
                distractor.text: [0.9, 0.1],
                "Seksyen manakah berkaitan cuti tahunan dalam Akta Kerja 1955?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="Seksyen manakah berkaitan cuti tahunan dalam Akta Kerja 1955?",
        chunks=[distractor, annual_leave_chunk],
        top_k=2,
        mode="hybrid",
    )

    assert results[0].chunk.chunk_id == "employment:60e:0"


def test_hybrid_recall_expands_amendment_commencement_query() -> None:
    commencement_chunk = Chunk(
        chunk_id="a1727:1:0",
        document_id="a1727",
        section_heading="Section 1 Short title and commencement",
        section_id="1",
        subsection_id=None,
        paragraph_id=None,
        text="This Act may be cited as the Personal Data Protection (Amendment) Act 2024 and comes into operation on a date appointed by the Minister.",
        source_path="data/raw_law_pdfs/pdpa-amendment.pdf",
        act_title="Personal Data Protection (Amendment) Act 2024",
        source_file="pdpa-amendment.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="1",
        document_aliases=("Personal Data Protection (Amendment) Act 2024", "Act A1727"),
    )
    distractor = Chunk(
        chunk_id="pdpa:5:0",
        document_id="pdpa",
        section_heading="Section 5 Personal Data Protection Principles",
        section_id="5",
        subsection_id=None,
        paragraph_id=None,
        text="The personal data protection principles govern processing of personal data.",
        source_path="data/raw_law_pdfs/pdpa.pdf",
        act_title="Personal Data Protection Act 2010",
        source_file="pdpa.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="5",
        document_aliases=("Personal Data Protection Act 2010", "PDPA"),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                commencement_chunk.text: [0.2, 0.8],
                distractor.text: [0.9, 0.1],
                "When does the Personal Data Protection (Amendment) Act 2024 come into force?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="When does the Personal Data Protection (Amendment) Act 2024 come into force?",
        chunks=[distractor, commencement_chunk],
        top_k=2,
        mode="hybrid",
    )

    assert results[0].chunk.chunk_id == "a1727:1:0"


def test_hybrid_recall_expands_gazette_order_effective_date_query() -> None:
    rates_chunk = Chunk(
        chunk_id="pua:3:0",
        document_id="pua_376",
        section_heading="Section 3 Minimum wage rates with effect from 1 February 2025",
        section_id="3",
        subsection_id=None,
        paragraph_id=None,
        text="The minimum wage rates with effect from 1 February 2025 are set out in this section.",
        source_path="data/raw_law_pdfs/minimum-wages-order-2024.pdf",
        act_title="Minimum Wages Order 2024",
        source_file="minimum-wages-order-2024.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="3",
        document_aliases=("Minimum Wages Order 2024", "P.U. (A) 376", "PUA 376"),
    )
    distractor = Chunk(
        chunk_id="pua:2:0",
        document_id="pua_376",
        section_heading="Section 2 Non-application",
        section_id="2",
        subsection_id=None,
        paragraph_id=None,
        text="This Order does not apply to domestic servants.",
        source_path="data/raw_law_pdfs/minimum-wages-order-2024.pdf",
        act_title="Minimum Wages Order 2024",
        source_file="minimum-wages-order-2024.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="2",
        document_aliases=("Minimum Wages Order 2024", "P.U. (A) 376", "PUA 376"),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                rates_chunk.text: [0.2, 0.8],
                distractor.text: [0.9, 0.1],
                "What minimum wage rates apply from 1 February 2025 under the Minimum Wages Order 2024?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="What minimum wage rates apply from 1 February 2025 under the Minimum Wages Order 2024?",
        chunks=[distractor, rates_chunk],
        top_k=2,
        mode="hybrid",
    )

    assert results[0].chunk.chunk_id == "pua:3:0"


def test_hybrid_retrieval_normalizes_malay_exact_unit_lookup() -> None:
    section_two = Chunk(
        chunk_id="employment:2:0",
        document_id="employment",
        section_heading="Section 2 Interpretation",
        section_id="2",
        subsection_id=None,
        paragraph_id=None,
        text="Section 2 Interpretation\nIn this Act, employee means a person employed under a contract of service.",
        source_path="data/raw_law_pdfs/Akta Kerja 1955 (Akta 265).pdf",
        act_title="Employment Act 1955",
        source_file="Akta Kerja 1955 (Akta 265).pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="2",
        document_aliases=("Employment Act 1955", "Akta Kerja 1955", "Act 265"),
    )
    distractor = Chunk(
        chunk_id="employment:20:0",
        document_id="employment",
        section_heading="Section 20 Representations on dismissal",
        section_id="20",
        subsection_id=None,
        paragraph_id=None,
        text="An employee may make representations to the Director General.",
        source_path="data/raw_law_pdfs/Akta Kerja 1955 (Akta 265).pdf",
        act_title="Employment Act 1955",
        source_file="Akta Kerja 1955 (Akta 265).pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="20",
        document_aliases=("Employment Act 1955", "Akta Kerja 1955", "Act 265"),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                section_two.text: [0.2, 0.8],
                distractor.text: [0.9, 0.1],
                "Apakah kandungan Seksyen 2 Akta Kerja 1955?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="Apakah kandungan Seksyen 2 Akta Kerja 1955?",
        chunks=[distractor, section_two],
        top_k=2,
        mode="hybrid",
    )

    assert results[0].chunk.chunk_id == "employment:2:0"


def test_hybrid_recall_expands_excluded_query_to_non_application_heading() -> None:
    non_application_chunk = Chunk(
        chunk_id="pua:2:0",
        document_id="pua_376",
        section_heading="Section 2 Non-application",
        section_id="2",
        subsection_id=None,
        paragraph_id=None,
        text="This Order does not apply to domestic servants.",
        source_path="data/raw_law_pdfs/minimum-wages-order-2024.pdf",
        act_title="Minimum Wages Order 2024",
        source_file="minimum-wages-order-2024.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="2",
        document_aliases=("Minimum Wages Order 2024", "P.U. (A) 376", "PUA 376"),
    )
    distractor = Chunk(
        chunk_id="penal:174:0",
        document_id="penal",
        section_heading="Section 174 Non-attendance in obedience to an order from a public servant",
        section_id="174",
        subsection_id=None,
        paragraph_id=None,
        text="Non-attendance in obedience to an order from a public servant is an offence.",
        source_path="data/raw_law_pdfs/penal-code.pdf",
        act_title="Penal Code",
        source_file="penal-code.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="174",
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                non_application_chunk.text: [0.2, 0.8],
                distractor.text: [0.9, 0.1],
                "Who is excluded from the Minimum Wages Order 2024?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="Who is excluded from the Minimum Wages Order 2024?",
        chunks=[distractor, non_application_chunk],
        top_k=2,
        mode="hybrid",
    )

    assert results[0].chunk.chunk_id == "pua:2:0"


def test_hybrid_retrieval_prefers_referenced_document_for_exact_unit_collision() -> None:
    employment_section = Chunk(
        chunk_id="employment:2:0",
        document_id="employment",
        section_heading="Section 2 Interpretation",
        section_id="2",
        subsection_id=None,
        paragraph_id=None,
        text="In this Act, employee means a person employed under a contract of service.",
        source_path="data/raw_law_pdfs/Akta Kerja 1955 (Akta 265).pdf",
        act_title="Employment Act 1955",
        source_file="Akta Kerja 1955 (Akta 265).pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="2",
        document_aliases=("Employment Act 1955", "Akta Kerja 1955", "Act 265"),
    )
    incidental_reference = Chunk(
        chunk_id="osha:2:0",
        document_id="osha",
        section_heading="Section 2 Employment Act 1955 [Act 265]",
        section_id="2",
        subsection_id=None,
        paragraph_id=None,
        text="The First Schedule references the Employment Act 1955 [Act 265].",
        source_path="data/raw_law_pdfs/osha.pdf",
        act_title="Occupational Safety and Health Act 1994",
        source_file="osha.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="2",
        document_aliases=("Occupational Safety and Health Act 1994",),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                employment_section.text: [0.2, 0.8],
                incidental_reference.text: [0.9, 0.1],
                "Apakah kandungan Seksyen 2 Akta Kerja 1955?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="Apakah kandungan Seksyen 2 Akta Kerja 1955?",
        chunks=[incidental_reference, employment_section],
        top_k=2,
        mode="hybrid",
    )

    assert results[0].chunk.chunk_id == "employment:2:0"


def test_hybrid_retrieval_prefers_target_amendment_section_within_same_act() -> None:
    target_section = Chunk(
        chunk_id="a1727:3:0",
        document_id="a1727",
        section_heading="Section 3 Amendment of section 4",
        section_id="3",
        subsection_id=None,
        paragraph_id=None,
        text="This Act amends section 4 of the principal Act.",
        source_path="data/raw_law_pdfs/pdpa-amendment.pdf",
        act_title="Personal Data Protection (Amendment) Act 2024",
        source_file="pdpa-amendment.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="3",
        document_aliases=("Personal Data Protection (Amendment) Act 2024", "Act A1727", "PDPA Amendment Act 2024"),
    )
    adjacent_section = Chunk(
        chunk_id="a1727:4:0",
        document_id="a1727",
        section_heading="Section 4 Amendment of section 5",
        section_id="4",
        subsection_id=None,
        paragraph_id=None,
        text="This Act amends section 5 of the principal Act.",
        source_path="data/raw_law_pdfs/pdpa-amendment.pdf",
        act_title="Personal Data Protection (Amendment) Act 2024",
        source_file="pdpa-amendment.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="4",
        document_aliases=("Personal Data Protection (Amendment) Act 2024", "Act A1727", "PDPA Amendment Act 2024"),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                target_section.text: [0.2, 0.8],
                adjacent_section.text: [0.9, 0.1],
                "Which section of Act A1727 amends section 4 of the PDPA?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="Which section of Act A1727 amends section 4 of the PDPA?",
        chunks=[adjacent_section, target_section],
        top_k=2,
        mode="hybrid",
    )

    assert results[0].chunk.chunk_id == "a1727:3:0"


def test_hybrid_retrieval_prefers_revocation_section_within_order_document() -> None:
    revocation_chunk = Chunk(
        chunk_id="pua:6:0",
        document_id="pua_376",
        section_heading="Section 6 Revocation",
        section_id="6",
        subsection_id=None,
        paragraph_id=None,
        text="The Minimum Wages Order 2022 is revoked.",
        source_path="data/raw_law_pdfs/minimum-wages-order-2024.pdf",
        act_title="Minimum Wages Order 2024",
        source_file="minimum-wages-order-2024.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="6",
        document_aliases=("Minimum Wages Order 2024", "P.U. (A) 376", "PUA 376"),
    )
    generic_order_chunk = Chunk(
        chunk_id="employment:87a:0",
        document_id="employment",
        section_heading="Section 87A Court order for payments due to employee",
        section_id="87A",
        subsection_id=None,
        paragraph_id=None,
        text="The Court may make an order for payments due to an employee.",
        source_path="data/raw_law_pdfs/employment.pdf",
        act_title="Employment Act 1955",
        source_file="employment.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="87A",
        document_aliases=("Employment Act 1955",),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                revocation_chunk.text: [0.2, 0.8],
                generic_order_chunk.text: [0.9, 0.1],
                "Which earlier order is revoked by the Minimum Wages Order 2024?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="Which earlier order is revoked by the Minimum Wages Order 2024?",
        chunks=[generic_order_chunk, revocation_chunk],
        top_k=2,
        mode="hybrid",
    )

    assert results[0].chunk.chunk_id == "pua:6:0"


def test_hybrid_retrieval_prefers_principal_act_for_direct_lookup_over_amendment_act() -> None:
    principal_section = Chunk(
        chunk_id="pdpa:1:0",
        document_id="pdpa",
        section_heading="Section 1 Short title and commencement",
        section_id="1",
        subsection_id=None,
        paragraph_id=None,
        text="This Act may be cited as the Personal Data Protection Act 2010.",
        source_path="data/raw_law_pdfs/pdpa.pdf",
        act_title="Personal Data Protection Act 2010",
        source_file="pdpa.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="1",
        document_aliases=("Personal Data Protection Act 2010", "PDPA"),
    )
    amendment_section = Chunk(
        chunk_id="a1727:1:0",
        document_id="a1727",
        section_heading="Section 1 Short title and commencement",
        section_id="1",
        subsection_id=None,
        paragraph_id=None,
        text="This Act may be cited as the Personal Data Protection (Amendment) Act 2024.",
        source_path="data/raw_law_pdfs/pdpa-amendment.pdf",
        act_title="Personal Data Protection (Amendment) Act 2024",
        source_file="pdpa-amendment.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="1",
        document_aliases=("Personal Data Protection (Amendment) Act 2024", "Act A1727"),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                principal_section.text: [0.2, 0.8],
                amendment_section.text: [0.9, 0.1],
                "What does Section 1 of the Personal Data Protection Act 2010 say?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="What does Section 1 of the Personal Data Protection Act 2010 say?",
        chunks=[amendment_section, principal_section],
        top_k=2,
        mode="hybrid",
    )

    assert results[0].chunk.chunk_id == "pdpa:1:0"


def test_hybrid_retrieval_prefers_non_application_heading_for_exclusion_query() -> None:
    exclusion_chunk = Chunk(
        chunk_id="pua:2:0",
        document_id="pua_376",
        section_heading="Section 2 Non-application",
        section_id="2",
        subsection_id=None,
        paragraph_id=None,
        text="This Order does not apply to domestic servants.",
        source_path="data/raw_law_pdfs/minimum-wages-order-2024.pdf",
        act_title="Minimum Wages Order 2024",
        source_file="minimum-wages-order-2024.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="2",
        document_aliases=("Minimum Wages Order 2024", "P.U. (A) 376", "PUA 376"),
    )
    rate_chunk = Chunk(
        chunk_id="pua:3:0",
        document_id="pua_376",
        section_heading="Section 3 Minimum wage rates with effect from 1 February 2025",
        section_id="3",
        subsection_id=None,
        paragraph_id=None,
        text="The minimum wage rates with effect from 1 February 2025 are set out in this section.",
        source_path="data/raw_law_pdfs/minimum-wages-order-2024.pdf",
        act_title="Minimum Wages Order 2024",
        source_file="minimum-wages-order-2024.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="3",
        document_aliases=("Minimum Wages Order 2024", "P.U. (A) 376", "PUA 376"),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                exclusion_chunk.text: [0.2, 0.8],
                rate_chunk.text: [0.9, 0.1],
                "Who is excluded from the Minimum Wages Order 2024?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="Who is excluded from the Minimum Wages Order 2024?",
        chunks=[rate_chunk, exclusion_chunk],
        top_k=2,
        mode="hybrid",
    )

    assert results[0].chunk.chunk_id == "pua:2:0"


def test_hybrid_retrieval_prefers_citation_and_commencement_heading() -> None:
    citation_chunk = Chunk(
        chunk_id="pua:1:0",
        document_id="pua_376",
        section_heading="Section 1 Citation and commencement",
        section_id="1",
        subsection_id=None,
        paragraph_id=None,
        text="This Order may be cited as the Minimum Wages Order 2024 and comes into operation on 1 February 2025.",
        source_path="data/raw_law_pdfs/minimum-wages-order-2024.pdf",
        act_title="Minimum Wages Order 2024",
        source_file="minimum-wages-order-2024.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="1",
        document_aliases=("Minimum Wages Order 2024", "P.U. (A) 376", "PUA 376"),
    )
    rate_chunk = Chunk(
        chunk_id="pua:3:0",
        document_id="pua_376",
        section_heading="Section 3 Minimum wage rates with effect from 1 February 2025",
        section_id="3",
        subsection_id=None,
        paragraph_id=None,
        text="The minimum wage rates with effect from 1 February 2025 are set out in this section.",
        source_path="data/raw_law_pdfs/minimum-wages-order-2024.pdf",
        act_title="Minimum Wages Order 2024",
        source_file="minimum-wages-order-2024.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="3",
        document_aliases=("Minimum Wages Order 2024", "P.U. (A) 376", "PUA 376"),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                citation_chunk.text: [0.2, 0.8],
                rate_chunk.text: [0.9, 0.1],
                "Which section of the Minimum Wages Order 2024 deals with citation and commencement?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="Which section of the Minimum Wages Order 2024 deals with citation and commencement?",
        chunks=[rate_chunk, citation_chunk],
        top_k=2,
        mode="hybrid",
    )

    assert results[0].chunk.chunk_id == "pua:1:0"


def test_hybrid_retrieval_prefers_new_section_for_data_portability_introduction() -> None:
    new_section_chunk = Chunk(
        chunk_id="a1727:9:0",
        document_id="a1727",
        section_heading="Section 9 New section 43a",
        section_id="9",
        subsection_id=None,
        paragraph_id=None,
        text="This amendment introduces the right to data portability by inserting new section 43A.",
        source_path="data/raw_law_pdfs/pdpa-amendment.pdf",
        act_title="Personal Data Protection (Amendment) Act 2024",
        source_file="pdpa-amendment.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="9",
        document_aliases=("Personal Data Protection (Amendment) Act 2024", "Act A1727"),
    )
    amendment_chunk = Chunk(
        chunk_id="a1727:3:0",
        document_id="a1727",
        section_heading="Section 3 Amendment of section 4",
        section_id="3",
        subsection_id=None,
        paragraph_id=None,
        text="This Act amends section 4 of the principal Act.",
        source_path="data/raw_law_pdfs/pdpa-amendment.pdf",
        act_title="Personal Data Protection (Amendment) Act 2024",
        source_file="pdpa-amendment.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="3",
        document_aliases=("Personal Data Protection (Amendment) Act 2024", "Act A1727"),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                new_section_chunk.text: [0.2, 0.8],
                amendment_chunk.text: [0.9, 0.1],
                "Seksyen manakah memperkenalkan hak pemindahan data dalam Akta Pindaan PDPA 2024?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="Seksyen manakah memperkenalkan hak pemindahan data dalam Akta Pindaan PDPA 2024?",
        chunks=[amendment_chunk, new_section_chunk],
        top_k=2,
        mode="hybrid",
    )

    assert results[0].chunk.chunk_id == "a1727:9:0"


def test_embedding_only_mode_preserves_metadata_and_uses_scores() -> None:
    perkara_chunk = Chunk(
        chunk_id="perlembagaan:10:0",
        document_id="perlembagaan",
        section_heading="Perkara 10 Kebebasan bercakap",
        section_id="10",
        subsection_id=None,
        paragraph_id=None,
        text="Tiap-tiap warganegara berhak kepada kebebasan bercakap dan bersuara.",
        source_path="data/raw_law_pdfs/perlembagaan.pdf",
        act_title="Perlembagaan Persekutuan",
        source_file="Perlembagaan.pdf",
        chunk_index=0,
        unit_type="perkara",
        unit_id="10",
    )
    other_chunk = Chunk(
        chunk_id="perlembagaan:11:0",
        document_id="perlembagaan",
        section_heading="Perkara 11 Kebebasan beragama",
        section_id="11",
        subsection_id=None,
        paragraph_id=None,
        text="Tiap-tiap orang berhak menganuti dan mengamalkan agamanya.",
        source_path="data/raw_law_pdfs/perlembagaan.pdf",
        act_title="Perlembagaan Persekutuan",
        source_file="Perlembagaan.pdf",
        chunk_index=1,
        unit_type="perkara",
        unit_id="11",
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                perkara_chunk.text: [1.0, 0.0],
                other_chunk.text: [0.0, 1.0],
                "kebebasan bercakap": [1.0, 0.0],
            }
        ),
    )
    index = SimpleVectorIndex(embedder)
    index.add([other_chunk, perkara_chunk])

    results = index.search("kebebasan bercakap", top_k=1, mode="embedding")

    assert results[0].chunk.chunk_id == "perlembagaan:10:0"
    assert results[0].chunk.unit_type == "perkara"
    assert results[0].chunk.unit_id == "10"


def test_hybrid_reranker_boosts_definition_heading_over_operational_section() -> None:
    definition_chunk = Chunk(
        chunk_id="pdpa:4:0",
        document_id="pdpa",
        section_heading="Section 4 Interpretation",
        section_id="4",
        subsection_id=None,
        paragraph_id=None,
        text="In this Act, unless the context otherwise requires, sensitive personal data means personal data consisting of information as to health.",
        source_path="data/raw_law_pdfs/pdpa.pdf",
        act_title="Personal Data Protection Act 2010",
        source_file="pdpa.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="4",
    )
    operational_chunk = Chunk(
        chunk_id="pdpa:40:0",
        document_id="pdpa",
        section_heading="Section 40 Processing of sensitive personal data",
        section_id="40",
        subsection_id=None,
        paragraph_id=None,
        text="A data user shall not process any sensitive personal data except in accordance with this section and section 129.",
        source_path="data/raw_law_pdfs/pdpa.pdf",
        act_title="Personal Data Protection Act 2010",
        source_file="pdpa.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="40",
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                definition_chunk.text: [0.2, 0.8],
                operational_chunk.text: [0.9, 0.1],
                "How does the PDPA define sensitive personal data?": [1.0, 0.0],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    hybrid_results = retriever.search(
        query="How does the PDPA define sensitive personal data?",
        chunks=[definition_chunk, operational_chunk],
        top_k=2,
        mode="hybrid",
    )
    reranked_results = retriever.search(
        query="How does the PDPA define sensitive personal data?",
        chunks=[definition_chunk, operational_chunk],
        top_k=2,
        mode="hybrid_rerank",
    )

    assert hybrid_results[0].chunk.chunk_id in {"pdpa:4:0", "pdpa:40:0"}
    assert reranked_results[0].chunk.chunk_id == "pdpa:4:0"


def test_hybrid_reranker_penalizes_cross_reference_heavy_article() -> None:
    target_chunk = Chunk(
        chunk_id="constitution:8:0",
        document_id="constitution",
        section_heading="Article 8 Equality",
        section_id="8",
        subsection_id=None,
        paragraph_id=None,
        text="All persons are equal before the law and entitled to the equal protection of the law.",
        source_path="data/raw_law_pdfs/federal-constitution.pdf",
        act_title="Federal Constitution",
        source_file="Federal Constitution.pdf",
        chunk_index=0,
        unit_type="article",
        unit_id="8",
    )
    cross_reference_chunk = Chunk(
        chunk_id="constitution:122:0",
        document_id="constitution",
        section_heading="Article 122 Constitution of Federal Court",
        section_id="122",
        subsection_id=None,
        paragraph_id=None,
        text="Equality and courts are discussed subject to Article 8, Article 121, Article 122 and Article 128 in this Constitution.",
        source_path="data/raw_law_pdfs/federal-constitution.pdf",
        act_title="Federal Constitution",
        source_file="Federal Constitution.pdf",
        chunk_index=1,
        unit_type="article",
        unit_id="122",
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                target_chunk.text: [0.1, 0.9],
                cross_reference_chunk.text: [0.9, 0.1],
                "Which article is titled Equality in the Federal Constitution?": [1.0, 0.0],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    hybrid_results = retriever.search(
        query="Which article is titled Equality in the Federal Constitution?",
        chunks=[target_chunk, cross_reference_chunk],
        top_k=2,
        mode="hybrid",
    )
    reranked_results = retriever.search(
        query="Which article is titled Equality in the Federal Constitution?",
        chunks=[target_chunk, cross_reference_chunk],
        top_k=2,
        mode="hybrid_rerank",
    )

    assert hybrid_results[0].chunk.chunk_id == "constitution:122:0"
    assert reranked_results[0].chunk.chunk_id == "constitution:8:0"


def test_hybrid_filtered_rerank_recovers_heading_match_just_outside_base_rerank_pool() -> None:
    query = "Which article is titled Equality in the Federal Constitution?"
    target_chunk = Chunk(
        chunk_id="constitution:8:0",
        document_id="constitution",
        section_heading="Article 8 Equality",
        section_id="8",
        subsection_id=None,
        paragraph_id=None,
        text="All persons are equal before the law and entitled to the equal protection of the law.",
        source_path="data/raw_law_pdfs/federal-constitution.pdf",
        act_title="Federal Constitution",
        source_file="Federal Constitution.pdf",
        chunk_index=99,
        unit_type="article",
        unit_id="8",
    )
    distractors = [
        Chunk(
            chunk_id=f"constitution:{index}:0",
            document_id="constitution",
            section_heading=f"Article {index} Administrative provisions",
            section_id=str(index),
            subsection_id=None,
            paragraph_id=None,
            text=(
                f"Constitution equality administration references Article 8, "
                f"Article 121 and Article {index} in constitutional practice."
            ),
            source_path="data/raw_law_pdfs/federal-constitution.pdf",
            act_title="Federal Constitution",
            source_file="Federal Constitution.pdf",
            chunk_index=index,
            unit_type="article",
            unit_id=str(index),
        )
        for index in range(20, 31)
    ]
    vectors = {
        query: [1.0, 0.0],
        target_chunk.text: [0.05, 0.95],
    }
    for distractor in distractors:
        vectors[distractor.text] = [1.0, 0.0]
    embedder = OllamaEmbedder(model="test-model", transport=FakeTransport(vectors))
    retriever = EmbeddingRetriever(embedder)

    reranked_results = retriever.search(
        query=query,
        chunks=[target_chunk, *distractors],
        top_k=3,
        mode="hybrid_rerank",
    )
    filtered_results = retriever.search(
        query=query,
        chunks=[target_chunk, *distractors],
        top_k=3,
        mode="hybrid_filtered_rerank",
    )

    assert all(result.chunk.chunk_id != "constitution:8:0" for result in reranked_results)
    assert filtered_results[0].chunk.chunk_id == "constitution:8:0"


def test_hybrid_filtered_rerank_limits_weak_embedding_candidates_when_heading_matches_exist() -> None:
    heading_chunk = Chunk(
        chunk_id="constitution:8:0",
        document_id="constitution",
        section_heading="Article 8 Equality",
        section_id="8",
        subsection_id=None,
        paragraph_id=None,
        text="All persons are equal before the law and entitled to the equal protection of the law.",
        source_path="data/raw_law_pdfs/federal-constitution.pdf",
        act_title="Federal Constitution",
        source_file="Federal Constitution.pdf",
        chunk_index=0,
        unit_type="article",
        unit_id="8",
    )
    weak_one = Chunk(
        chunk_id="constitution:122:0",
        document_id="constitution",
        section_heading="Article 122 Constitution of Federal Court",
        section_id="122",
        subsection_id=None,
        paragraph_id=None,
        text="Court composition and Article 8, Article 121, Article 122 and Article 128 references appear here.",
        source_path="data/raw_law_pdfs/federal-constitution.pdf",
        act_title="Federal Constitution",
        source_file="Federal Constitution.pdf",
        chunk_index=1,
        unit_type="article",
        unit_id="122",
    )
    weak_two = Chunk(
        chunk_id="constitution:145:0",
        document_id="constitution",
        section_heading="Article 145 Attorney General",
        section_id="145",
        subsection_id=None,
        paragraph_id=None,
        text="Prosecutorial powers are described with Article 8 and Article 145 references.",
        source_path="data/raw_law_pdfs/federal-constitution.pdf",
        act_title="Federal Constitution",
        source_file="Federal Constitution.pdf",
        chunk_index=2,
        unit_type="article",
        unit_id="145",
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                heading_chunk.text: [0.1, 0.9],
                weak_one.text: [1.0, 0.0],
                weak_two.text: [0.95, 0.05],
                "Which article is titled Equality in the Federal Constitution?": [1.0, 0.0],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="Which article is titled Equality in the Federal Constitution?",
        chunks=[heading_chunk, weak_one, weak_two],
        top_k=2,
        mode="hybrid_filtered_rerank",
    )

    assert results[0].chunk.chunk_id == "constitution:8:0"
    assert all(result.chunk.chunk_id != "constitution:145:0" for result in results)


def test_hybrid_filtered_rerank_strongly_prefers_query_document_alias() -> None:
    pdpa_chunk = Chunk(
        chunk_id="pdpa:7:0",
        document_id="pdpa",
        section_heading="Section 7 Notice and Choice Principle",
        section_id="7",
        subsection_id=None,
        paragraph_id=None,
        text="A data user shall by written notice inform the data subject of the purpose of collection and the choices available.",
        source_path="data/raw_law_pdfs/pdpa.pdf",
        act_title="Personal Data Protection Act 2010",
        source_file="pdpa.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="7",
    )
    constitution_chunk = Chunk(
        chunk_id="constitution:160:0",
        document_id="constitution",
        section_heading="Article 160 Interpretation",
        section_id="160",
        subsection_id=None,
        paragraph_id=None,
        text="In this Constitution, interpretation provisions define expressions and terms used in the Constitution.",
        source_path="data/raw_law_pdfs/federal-constitution.pdf",
        act_title="Federal Constitution",
        source_file="Federal Constitution.pdf",
        chunk_index=1,
        unit_type="article",
        unit_id="160",
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                pdpa_chunk.text: [0.1, 0.9],
                constitution_chunk.text: [1.0, 0.0],
                "What does the PDPA Notice and Choice Principle require?": [1.0, 0.0],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="What does the PDPA Notice and Choice Principle require?",
        chunks=[pdpa_chunk, constitution_chunk],
        top_k=2,
        mode="hybrid_filtered_rerank",
    )

    assert results[0].chunk.chunk_id == "pdpa:7:0"
    assert len(results) >= 1


def test_hybrid_filtered_rerank_does_not_overboost_short_unrelated_act_acronyms() -> None:
    employment_chunk = Chunk(
        chunk_id="employment:4:0",
        document_id="employment",
        section_heading="Section 4 Appeals",
        section_id="4",
        subsection_id=None,
        paragraph_id=None,
        text="An employee may appeal under this section in the prescribed manner.",
        source_path="data/raw_law_pdfs/employment.pdf",
        act_title="Employment Act 1955",
        source_file="employment.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="4",
        document_aliases=("Employment Act 1955",),
    )
    income_tax_chunk = Chunk(
        chunk_id="ita:99:0",
        document_id="ita",
        section_heading="Section 99 Right of appeal",
        section_id="99",
        subsection_id=None,
        paragraph_id=None,
        text="A taxpayer may appeal under this section against an assessment.",
        source_path="data/raw_law_pdfs/income-tax.pdf",
        act_title="Income Tax Act 1967",
        source_file="income-tax.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="99",
        document_aliases=("Income Tax Act 1967",),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                employment_chunk.text: [0.1, 0.9],
                income_tax_chunk.text: [0.9, 0.1],
                "Which section of the Employment Act 1955 deals with appeals?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    filtered_results = retriever.search(
        query="Which section of the Employment Act 1955 deals with appeals?",
        chunks=[income_tax_chunk, employment_chunk],
        top_k=2,
        mode="hybrid_filtered_rerank",
    )

    assert filtered_results[0].chunk.chunk_id == "employment:4:0"


def test_hybrid_filtered_rerank_applies_to_capability_style_appeal_query() -> None:
    appeals_chunk = Chunk(
        chunk_id="employment:4:0",
        document_id="employment",
        section_heading="Section 4 Appeals",
        section_id="4",
        subsection_id=None,
        paragraph_id=None,
        text="An employee may appeal under this section in the prescribed manner.",
        source_path="data/raw_law_pdfs/employment.pdf",
        act_title="Employment Act 1955",
        source_file="employment.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="4",
        document_aliases=("Employment Act 1955",),
    )
    distractor = Chunk(
        chunk_id="employment:77:0",
        document_id="employment",
        section_heading="Section 77 Complaints to Director General",
        section_id="77",
        subsection_id=None,
        paragraph_id=None,
        text="The Director General may inquire into complaints made by employees under this Part.",
        source_path="data/raw_law_pdfs/employment.pdf",
        act_title="Employment Act 1955",
        source_file="employment.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="77",
        document_aliases=("Employment Act 1955",),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                appeals_chunk.text: [0.1, 0.9],
                distractor.text: [0.95, 0.05],
                "Who can appeal under the Employment Act 1955?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    hybrid_results = retriever.search(
        query="Who can appeal under the Employment Act 1955?",
        chunks=[distractor, appeals_chunk],
        top_k=2,
        mode="hybrid",
    )
    reranked_results = retriever.search(
        query="Who can appeal under the Employment Act 1955?",
        chunks=[distractor, appeals_chunk],
        top_k=2,
        mode="hybrid_filtered_rerank",
    )

    assert any(result.chunk.chunk_id == "employment:4:0" for result in hybrid_results)
    assert reranked_results[0].chunk.chunk_id == "employment:4:0"


def test_hybrid_rerank_applies_to_heading_lookup_and_keeps_exact_heading_top() -> None:
    access_chunk = Chunk(
        chunk_id="pdpa:12:0",
        document_id="pdpa",
        section_heading="Section 12 Access Principle",
        section_id="12",
        subsection_id=None,
        paragraph_id=None,
        text="Section 12 Access Principle\nA data user shall allow access to personal data subject to this Act.",
        source_path="data/raw_law_pdfs/pdpa.pdf",
        act_title="Personal Data Protection Act 2010",
        source_file="pdpa.pdf",
        chunk_index=12,
        unit_type="section",
        unit_id="12",
        document_aliases=("Personal Data Protection Act 2010", "PDPA"),
    )
    security_chunk = Chunk(
        chunk_id="pdpa:9:0",
        document_id="pdpa",
        section_heading="Section 9 Security Principle",
        section_id="9",
        subsection_id=None,
        paragraph_id=None,
        text="Section 9 Security Principle\nA data user shall take practical steps to protect personal data.",
        source_path="data/raw_law_pdfs/pdpa.pdf",
        act_title="Personal Data Protection Act 2010",
        source_file="pdpa.pdf",
        chunk_index=9,
        unit_type="section",
        unit_id="9",
        document_aliases=("Personal Data Protection Act 2010", "PDPA"),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                access_chunk.text: [0.2, 0.8],
                security_chunk.text: [0.9, 0.1],
                "Which section of the PDPA sets out the Access Principle?": [1.0, 0.0],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    reranked_results = retriever.search(
        query="Which section of the PDPA sets out the Access Principle?",
        chunks=[security_chunk, access_chunk],
        top_k=2,
        mode="hybrid_rerank",
    )

    assert reranked_results[0].chunk.chunk_id == "pdpa:12:0"


def test_hybrid_rerank_applies_to_amendment_query_and_preserves_best_candidate() -> None:
    portability_chunk = Chunk(
        chunk_id="a1727:9:0",
        document_id="a1727",
        section_heading="Section 9 New section 43a",
        section_id="9",
        subsection_id=None,
        paragraph_id=None,
        text="Section 9 New section 43a\nThis amendment introduces the right to data portability.",
        source_path="data/raw_law_pdfs/act-a1727.pdf",
        act_title="Personal Data Protection (Amendment) Act 2024",
        source_file="Act-A1727.pdf",
        chunk_index=9,
        unit_type="section",
        unit_id="9",
        document_aliases=("Personal Data Protection (Amendment) Act 2024", "Act A1727"),
    )
    division_chunk = Chunk(
        chunk_id="a1727:6:0",
        document_id="a1727",
        section_heading="Section 6 New Division 1a of Part II",
        section_id="6",
        subsection_id=None,
        paragraph_id=None,
        text="Section 6 New Division 1a of Part II\nThis amendment inserts a new Division 1A into Part II.",
        source_path="data/raw_law_pdfs/act-a1727.pdf",
        act_title="Personal Data Protection (Amendment) Act 2024",
        source_file="Act-A1727.pdf",
        chunk_index=6,
        unit_type="section",
        unit_id="6",
        document_aliases=("Personal Data Protection (Amendment) Act 2024", "Act A1727"),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                portability_chunk.text: [0.2, 0.8],
                division_chunk.text: [0.9, 0.1],
                "Which section of Act A1727 introduces the right to data portability?": [1.0, 0.0],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    reranked_results = retriever.search(
        query="Which section of Act A1727 introduces the right to data portability?",
        chunks=[division_chunk, portability_chunk],
        top_k=2,
        mode="hybrid_rerank",
    )

    assert reranked_results[0].chunk.chunk_id == "a1727:9:0"


def test_hybrid_rerank_skips_hierarchy_queries_and_keeps_base_successor() -> None:
    perkara_empat = Chunk(
        chunk_id="perlembagaan:4:4",
        document_id="perlembagaan",
        section_heading="Perkara 4 Undang-undang utama Persekutuan",
        section_id="4",
        subsection_id=None,
        paragraph_id=None,
        text="(3) Kesahan undang-undang.\nBahagian II\nKEBEBASAN ASASI",
        source_path="data/raw_law_pdfs/Perlembagaan Persekutuan (Cetakan Semula 2020)(1).pdf",
        act_title="Federal Constitution",
        source_file="Perlembagaan Persekutuan (Cetakan Semula 2020)(1).pdf",
        chunk_index=4,
        unit_type="perkara",
        unit_id="4",
        document_aliases=("Federal Constitution", "Constitution of Malaysia"),
    )
    perkara_lima = Chunk(
        chunk_id="perlembagaan:5:5",
        document_id="perlembagaan",
        section_heading="Perkara 5 Kebebasan diri",
        section_id="5",
        subsection_id=None,
        paragraph_id=None,
        text="Perkara 5 Kebebasan diri\nTiada seorang pun boleh diambil nyawanya kecuali mengikut undang-undang.",
        source_path="data/raw_law_pdfs/Perlembagaan Persekutuan (Cetakan Semula 2020)(1).pdf",
        act_title="Federal Constitution",
        source_file="Perlembagaan Persekutuan (Cetakan Semula 2020)(1).pdf",
        chunk_index=5,
        unit_type="perkara",
        unit_id="5",
        document_aliases=("Federal Constitution", "Constitution of Malaysia"),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                perkara_empat.text: [0.1, 0.9],
                perkara_lima.text: [0.2, 0.8],
                "Perkara manakah memulakan Bahagian II tentang Kebebasan Asasi dalam Perlembagaan Persekutuan?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    reranked_results = retriever.search(
        query="Perkara manakah memulakan Bahagian II tentang Kebebasan Asasi dalam Perlembagaan Persekutuan?",
        chunks=[perkara_empat, perkara_lima],
        top_k=2,
        mode="hybrid_rerank",
    )

    assert reranked_results[0].chunk.chunk_id == "perlembagaan:5:5"


def test_hybrid_filtered_rerank_still_skips_explicit_unit_lookup() -> None:
    section_four = Chunk(
        chunk_id="employment:4:0",
        document_id="employment",
        section_heading="Section 4 Appeals",
        section_id="4",
        subsection_id=None,
        paragraph_id=None,
        text="Appeal rights are set out in this section.",
        source_path="data/raw_law_pdfs/employment.pdf",
        act_title="Employment Act 1955",
        source_file="employment.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="4",
        document_aliases=("Employment Act 1955",),
    )
    section_forty = Chunk(
        chunk_id="employment:40:0",
        document_id="employment",
        section_heading="Section 40 General powers",
        section_id="40",
        subsection_id=None,
        paragraph_id=None,
        text="General powers under this Act are described here.",
        source_path="data/raw_law_pdfs/employment.pdf",
        act_title="Employment Act 1955",
        source_file="employment.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="40",
        document_aliases=("Employment Act 1955",),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                section_four.text: [0.1, 0.9],
                section_forty.text: [0.9, 0.1],
                "Section 4 Employment Act 1955": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="Section 4 Employment Act 1955",
        chunks=[section_forty, section_four],
        top_k=2,
        mode="hybrid_filtered_rerank",
    )

    assert [result.chunk.chunk_id for result in results] == ["employment:4:0"]


def test_hybrid_retrieval_boosts_first_section_of_requested_part_hierarchy() -> None:
    part_start_chunk = Chunk(
        chunk_id="employment:5:0",
        document_id="employment",
        section_heading="Section 5 Application of this Part",
        section_id="5",
        subsection_id=None,
        paragraph_id=None,
        text="Section 5 Application of this Part\nThis Part applies notwithstanding any other law.\nPart II",
        source_path="data/raw_law_pdfs/employment.pdf",
        act_title="Employment Act 1955",
        source_file="employment.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="5",
        document_aliases=("Employment Act 1955",),
    )
    distractor = Chunk(
        chunk_id="employment:77:0",
        document_id="employment",
        section_heading="Section 77 Appeal against Director General's order to High Court",
        section_id="77",
        subsection_id=None,
        paragraph_id=None,
        text="Employment Act 1955 procedural appeals are heard by the High Court.",
        source_path="data/raw_law_pdfs/employment.pdf",
        act_title="Employment Act 1955",
        source_file="employment.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="77",
        document_aliases=("Employment Act 1955",),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                part_start_chunk.text: [0.2, 0.8],
                distractor.text: [0.9, 0.1],
                "Which section begins Part II of the Employment Act 1955?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="Which section begins Part II of the Employment Act 1955?",
        chunks=[distractor, part_start_chunk],
        top_k=2,
        mode="hybrid",
    )

    assert results[0].chunk.chunk_id == "employment:5:0"


def test_hybrid_retrieval_can_promote_successor_unit_after_hierarchy_marker() -> None:
    article_four_tail = Chunk(
        chunk_id="constitution:4:1",
        document_id="constitution",
        section_heading="Article 4 Supreme law of the Federation",
        section_id="4",
        subsection_id=None,
        paragraph_id=None,
        text="(4) Existing laws continue in force.\nPart II\nFUNDAMENTAL LIBERTIES",
        source_path="data/raw_law_pdfs/federal-constitution.pdf",
        act_title="Federal Constitution",
        source_file="federal-constitution.pdf",
        chunk_index=0,
        unit_type="article",
        unit_id="4",
        document_aliases=("Federal Constitution", "Constitution of Malaysia"),
    )
    article_five = Chunk(
        chunk_id="constitution:5:0",
        document_id="constitution",
        section_heading="Article 5 Liberty of the person",
        section_id="5",
        subsection_id=None,
        paragraph_id=None,
        text="Article 5 Liberty of the person\nNo person shall be deprived of life or personal liberty save in accordance with law.",
        source_path="data/raw_law_pdfs/federal-constitution.pdf",
        act_title="Federal Constitution",
        source_file="federal-constitution.pdf",
        chunk_index=1,
        unit_type="article",
        unit_id="5",
        document_aliases=("Federal Constitution", "Constitution of Malaysia"),
    )
    distractor = Chunk(
        chunk_id="land:2:0",
        document_id="land",
        section_heading="Article 2",
        section_id="2",
        subsection_id=None,
        paragraph_id=None,
        text="Article 2 contains general land provisions.",
        source_path="data/raw_law_pdfs/national-land-code.pdf",
        act_title="National Land Code",
        source_file="national-land-code.pdf",
        chunk_index=2,
        unit_type="article",
        unit_id="2",
        document_aliases=("National Land Code",),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                article_four_tail.text: [0.1, 0.9],
                article_five.text: [0.2, 0.8],
                distractor.text: [0.95, 0.05],
                "Which Article begins Part II on Fundamental Liberties in the Federal Constitution?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="Which Article begins Part II on Fundamental Liberties in the Federal Constitution?",
        chunks=[distractor, article_five, article_four_tail],
        top_k=2,
        mode="hybrid",
    )

    assert results[0].chunk.chunk_id == "constitution:5:0"


def test_hybrid_retrieval_prefers_early_clean_hierarchy_marker_over_late_duplicate() -> None:
    article_thirteen = Chunk(
        chunk_id="constitution:13:19",
        document_id="constitution",
        section_heading="Article 13 Rights to property",
        section_id="13",
        subsection_id=None,
        paragraph_id=None,
        text="Article 13 Rights to property\n13. (1) No person shall be deprived of property save in accordance with law.\nPart III\nCITIZENSHIP",
        source_path="data/raw_law_pdfs/federal-constitution.pdf",
        act_title="Federal Constitution",
        source_file="Federal Constitution (Reprint 2020)(1).pdf",
        chunk_index=19,
        unit_type="article",
        unit_id="13",
        document_aliases=("Federal Constitution", "Constitution of Malaysia"),
    )
    article_fourteen = Chunk(
        chunk_id="constitution:14:20",
        document_id="constitution",
        section_heading="Article 14 Citizenship by operation of law",
        section_id="14",
        subsection_id=None,
        paragraph_id=None,
        text="Article 14 Citizenship by operation of law\n14. Subject to this Part, the following persons are citizens by operation of law.",
        source_path="data/raw_law_pdfs/federal-constitution.pdf",
        act_title="Federal Constitution",
        source_file="Federal Constitution (Reprint 2020)(1).pdf",
        chunk_index=20,
        unit_type="article",
        unit_id="14",
        document_aliases=("Federal Constitution", "Constitution of Malaysia"),
    )
    late_duplicate = Chunk(
        chunk_id="constitution:21:473",
        document_id="constitution",
        section_heading="Article 21 Composition of Legislative Assembly",
        section_id="21",
        subsection_id=None,
        paragraph_id=None,
        text="Article 21 Composition of Legislative Assembly\nNotwithstanding anything in section 6 of the Eighth Schedule to the Federal Constitution.\nPart III\nMODIFICATIONS OF PARTS I AND II IN RELATION TO THE STATES OF BORNEO",
        source_path="data/raw_law_pdfs/federal-constitution.pdf",
        act_title="Federal Constitution",
        source_file="Federal Constitution (Reprint 2020)(1).pdf",
        chunk_index=473,
        unit_type="article",
        unit_id="21",
        document_aliases=("Federal Constitution", "Constitution of Malaysia"),
    )
    article_twenty_two = Chunk(
        chunk_id="constitution:22:474",
        document_id="constitution",
        section_heading="Article 22 Additional provisions",
        section_id="22",
        subsection_id=None,
        paragraph_id=None,
        text="Article 22 Additional provisions\nThis Article continues the Schedule discussion.",
        source_path="data/raw_law_pdfs/federal-constitution.pdf",
        act_title="Federal Constitution",
        source_file="Federal Constitution (Reprint 2020)(1).pdf",
        chunk_index=474,
        unit_type="article",
        unit_id="22",
        document_aliases=("Federal Constitution", "Constitution of Malaysia"),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                article_thirteen.text: [0.1, 0.9],
                article_fourteen.text: [0.2, 0.8],
                late_duplicate.text: [0.95, 0.05],
                article_twenty_two.text: [0.9, 0.1],
                "Which Article begins Part III on Citizenship in the Federal Constitution?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="Which Article begins Part III on Citizenship in the Federal Constitution?",
        chunks=[late_duplicate, article_twenty_two, article_fourteen, article_thirteen],
        top_k=2,
        mode="hybrid",
    )

    assert results[0].chunk.chunk_id == "constitution:14:20"


def test_hybrid_retrieval_supports_malay_document_alias_from_source_name() -> None:
    perkara_empat = Chunk(
        chunk_id="perlembagaan:4:4",
        document_id="perlembagaan",
        section_heading="Perkara 4 Undang-undang utama Persekutuan",
        section_id="4",
        subsection_id=None,
        paragraph_id=None,
        text="(3) Kesahan undang-undang.\nBahagian II\nKEBEBASAN ASASI",
        source_path="data/raw_law_pdfs/Perlembagaan Persekutuan (Cetakan Semula 2020)(1).pdf",
        act_title="Federal Constitution",
        source_file="Perlembagaan Persekutuan (Cetakan Semula 2020)(1).pdf",
        chunk_index=4,
        unit_type="perkara",
        unit_id="4",
        document_aliases=("Federal Constitution", "Constitution of Malaysia"),
    )
    perkara_lima = Chunk(
        chunk_id="perlembagaan:5:5",
        document_id="perlembagaan",
        section_heading="Perkara 5 Kebebasan diri",
        section_id="5",
        subsection_id=None,
        paragraph_id=None,
        text="Perkara 5 Kebebasan diri\nTiada seorang pun boleh diambil nyawanya kecuali mengikut undang-undang.",
        source_path="data/raw_law_pdfs/Perlembagaan Persekutuan (Cetakan Semula 2020)(1).pdf",
        act_title="Federal Constitution",
        source_file="Perlembagaan Persekutuan (Cetakan Semula 2020)(1).pdf",
        chunk_index=5,
        unit_type="perkara",
        unit_id="5",
        document_aliases=("Federal Constitution", "Constitution of Malaysia"),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                perkara_empat.text: [0.1, 0.9],
                perkara_lima.text: [0.2, 0.8],
                "Perkara manakah memulakan Bahagian II dalam Perlembagaan Persekutuan?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="Perkara manakah memulakan Bahagian II dalam Perlembagaan Persekutuan?",
        chunks=[perkara_lima, perkara_empat],
        top_k=2,
        mode="hybrid",
    )

    assert results[0].chunk.chunk_id == "perlembagaan:5:5"


def test_hybrid_retrieval_boosts_obligation_language_over_definition_heading() -> None:
    duty_chunk = Chunk(
        chunk_id="osha:15:0",
        document_id="osha",
        section_heading="Section 15 General duties of employers and self-employed persons to their employees",
        section_id="15",
        subsection_id=None,
        paragraph_id=None,
        text="It shall be the duty of every employer to ensure, so far as is practicable, the safety, health and welfare at work of all his employees.",
        source_path="data/raw_law_pdfs/osha.pdf",
        act_title="Occupational Safety and Health Act 1994",
        source_file="osha.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="15",
        document_aliases=("Occupational Safety and Health Act 1994",),
    )
    definition_chunk = Chunk(
        chunk_id="osha:3:0",
        document_id="osha",
        section_heading="Section 3 Interpretation",
        section_id="3",
        subsection_id=None,
        paragraph_id=None,
        text="In this Act, employer means a person who has the control of a place of work.",
        source_path="data/raw_law_pdfs/osha.pdf",
        act_title="Occupational Safety and Health Act 1994",
        source_file="osha.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="3",
        document_aliases=("Occupational Safety and Health Act 1994",),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                duty_chunk.text: [0.2, 0.8],
                definition_chunk.text: [0.9, 0.1],
                "What duties does the Occupational Safety and Health Act 1994 impose?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="What duties does the Occupational Safety and Health Act 1994 impose?",
        chunks=[definition_chunk, duty_chunk],
        top_k=2,
        mode="hybrid",
    )

    assert results[0].chunk.chunk_id == "osha:15:0"


def test_hybrid_retrieval_prefers_broader_earlier_duties_section_within_same_document() -> None:
    employers = Chunk(
        chunk_id="osha:15:27",
        document_id="osha",
        section_heading="Section 15 General duties of employers",
        section_id="15",
        subsection_id=None,
        paragraph_id=None,
        text="Section 15 General duties of employers\nIt shall be the duty of every employer to ensure the safety, health and welfare at work of all his employees.",
        source_path="data/raw_law_pdfs/osha.pdf",
        act_title="Occupational Safety and Health Act 1994",
        source_file="Occupational-Safety-and-Health-Act-1994-Act-514_Reprint-Version-1.6.2024_English.pdf",
        chunk_index=27,
        unit_type="section",
        unit_id="15",
        document_aliases=("Occupational Safety and Health Act 1994",),
    )
    employees = Chunk(
        chunk_id="osha:24:48",
        document_id="osha",
        section_heading="Section 24 General duties of employees at work",
        section_id="24",
        subsection_id=None,
        paragraph_id=None,
        text="Section 24 General duties of employees at work\nIt shall be the duty of every employee while at work to take reasonable care for safety and health.\nSection 15 duties also continue to apply.",
        source_path="data/raw_law_pdfs/osha.pdf",
        act_title="Occupational Safety and Health Act 1994",
        source_file="Occupational-Safety-and-Health-Act-1994-Act-514_Reprint-Version-1.6.2024_English.pdf",
        chunk_index=48,
        unit_type="section",
        unit_id="24",
        document_aliases=("Occupational Safety and Health Act 1994",),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                employers.text: [0.1, 0.9],
                employees.text: [0.9, 0.1],
                "What duties does the Occupational Safety and Health Act 1994 impose?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="What duties does the Occupational Safety and Health Act 1994 impose?",
        chunks=[employees, employers],
        top_k=2,
        mode="hybrid",
    )

    assert results[0].chunk.chunk_id == "osha:15:27"


def test_hybrid_retrieval_returns_empty_for_impossible_explicit_unit_lookup() -> None:
    valid_chunk = Chunk(
        chunk_id="mw:6:0",
        document_id="mw",
        section_heading="Section 6 Revocation",
        section_id="6",
        subsection_id=None,
        paragraph_id=None,
        text="The Minimum Wages Order 2022 is revoked.",
        source_path="data/raw_law_pdfs/minimum-wages-order.pdf",
        act_title="Minimum Wages Order 2024",
        source_file="minimum-wages-order.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="6",
        document_aliases=("Minimum Wages Order 2024", "P.U. (A) 376"),
    )
    earlier_chunk = Chunk(
        chunk_id="mw:1:0",
        document_id="mw",
        section_heading="Section 1 Citation and commencement",
        section_id="1",
        subsection_id=None,
        paragraph_id=None,
        text="This Order may be cited as the Minimum Wages Order 2024.",
        source_path="data/raw_law_pdfs/minimum-wages-order.pdf",
        act_title="Minimum Wages Order 2024",
        source_file="minimum-wages-order.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="1",
        document_aliases=("Minimum Wages Order 2024", "P.U. (A) 376"),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                valid_chunk.text: [0.1, 0.9],
                earlier_chunk.text: [0.9, 0.1],
                "What does Section 999 of the Minimum Wages Order 2024 say?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="What does Section 999 of the Minimum Wages Order 2024 say?",
        chunks=[valid_chunk, earlier_chunk],
        top_k=3,
        mode="hybrid",
    )

    assert results == []


def test_hybrid_filtered_rerank_boosts_contract_and_leave_sections_for_broad_employment_agreement_query() -> None:
    contract_chunk = Chunk(
        chunk_id="employment:10:0",
        document_id="employment",
        section_heading="Section 10 Contracts to be in writing and to include provision for termination",
        section_id="10",
        subsection_id=None,
        paragraph_id=None,
        text="Every contract of service for a period exceeding one month shall be in writing and include provisions for termination.",
        source_path="data/raw_law_pdfs/employment.pdf",
        act_title="Employment Act 1955",
        source_file="employment.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="10",
        document_aliases=("Employment Act 1955",),
    )
    leave_chunk = Chunk(
        chunk_id="employment:60e:0",
        document_id="employment",
        section_heading="Section 60E Annual leave",
        section_id="60E",
        subsection_id=None,
        paragraph_id=None,
        text="An employee shall be entitled to paid annual leave after twelve months of continuous service.",
        source_path="data/raw_law_pdfs/employment.pdf",
        act_title="Employment Act 1955",
        source_file="employment.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="60E",
        document_aliases=("Employment Act 1955",),
    )
    foreign_employee_chunk = Chunk(
        chunk_id="employment:60k:0",
        document_id="employment",
        section_heading="Section 60K Employment of foreign employee",
        section_id="60K",
        subsection_id=None,
        paragraph_id=None,
        text="The Director General may approve the employment of a foreign employee subject to conditions.",
        source_path="data/raw_law_pdfs/employment.pdf",
        act_title="Employment Act 1955",
        source_file="employment.pdf",
        chunk_index=2,
        unit_type="section",
        unit_id="60K",
        document_aliases=("Employment Act 1955",),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                contract_chunk.text: [0.2, 0.8],
                leave_chunk.text: [0.3, 0.7],
                foreign_employee_chunk.text: [0.95, 0.05],
                "What do I need to check before signing an employment agreement?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="What do I need to check before signing an employment agreement?",
        chunks=[foreign_employee_chunk, leave_chunk, contract_chunk],
        top_k=3,
        mode="hybrid_filtered_rerank",
    )

    top_ids = [result.chunk.chunk_id for result in results]
    assert top_ids[0] == "employment:10:0"
    assert "employment:60e:0" in top_ids[:3]
    assert top_ids[0] != "employment:60k:0"


def test_hybrid_filtered_rerank_keeps_foreign_employee_section_down_for_broad_employment_contract_query() -> None:
    notice_chunk = Chunk(
        chunk_id="employment:12:0",
        document_id="employment",
        section_heading="Section 12 Notice of termination of contract",
        section_id="12",
        subsection_id=None,
        paragraph_id=None,
        text="Either party may at any time give to the other party notice of his intention to terminate such contract of service.",
        source_path="data/raw_law_pdfs/employment.pdf",
        act_title="Employment Act 1955",
        source_file="employment.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="12",
        document_aliases=("Employment Act 1955",),
    )
    wage_chunk = Chunk(
        chunk_id="employment:24:0",
        document_id="employment",
        section_heading="Section 24 Lawful deductions",
        section_id="24",
        subsection_id=None,
        paragraph_id=None,
        text="No deductions shall be made by an employer from the wages of an employee otherwise than in accordance with this Act.",
        source_path="data/raw_law_pdfs/employment.pdf",
        act_title="Employment Act 1955",
        source_file="employment.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="24",
        document_aliases=("Employment Act 1955",),
    )
    foreign_employee_chunk = Chunk(
        chunk_id="employment:60k:0",
        document_id="employment",
        section_heading="Section 60K Employment of foreign employee",
        section_id="60K",
        subsection_id=None,
        paragraph_id=None,
        text="The Director General may approve the employment of a foreign employee subject to conditions.",
        source_path="data/raw_law_pdfs/employment.pdf",
        act_title="Employment Act 1955",
        source_file="employment.pdf",
        chunk_index=2,
        unit_type="section",
        unit_id="60K",
        document_aliases=("Employment Act 1955",),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                notice_chunk.text: [0.2, 0.8],
                wage_chunk.text: [0.3, 0.7],
                foreign_employee_chunk.text: [0.98, 0.02],
                "How do I protect myself in an employment contract?": [0.99, 0.01],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="How do I protect myself in an employment contract?",
        chunks=[foreign_employee_chunk, wage_chunk, notice_chunk],
        top_k=3,
        mode="hybrid_filtered_rerank",
    )

    top_ids = [result.chunk.chunk_id for result in results]
    assert top_ids[0] == "employment:12:0"
    assert "employment:24:0" in top_ids[:3]
    assert top_ids[0] != "employment:60k:0"


def test_hybrid_filtered_rerank_prefers_exact_appeals_heading_within_same_document() -> None:
    appeals = Chunk(
        chunk_id="employment:4:0",
        document_id="employment",
        section_heading="Section 4 Appeals",
        section_id="4",
        subsection_id=None,
        paragraph_id=None,
        text="Any person aggrieved may appeal under this section.",
        source_path="data/raw_law_pdfs/employment.pdf",
        act_title="Employment Act 1955",
        source_file="employment.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="4",
        document_aliases=("Employment Act 1955",),
    )
    high_court_appeal = Chunk(
        chunk_id="employment:77:0",
        document_id="employment",
        section_heading="Section 77 Appeal against Director General's order to High Court",
        section_id="77",
        subsection_id=None,
        paragraph_id=None,
        text="An appeal against an order of the Director General may be made to the High Court.",
        source_path="data/raw_law_pdfs/employment.pdf",
        act_title="Employment Act 1955",
        source_file="employment.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="77",
        document_aliases=("Employment Act 1955",),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                appeals.text: [0.2, 0.8],
                high_court_appeal.text: [0.95, 0.05],
                "Which section of the Employment Act 1955 deals with appeals?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="Which section of the Employment Act 1955 deals with appeals?",
        chunks=[high_court_appeal, appeals],
        top_k=2,
        mode="hybrid_filtered_rerank",
    )

    assert results[0].chunk.chunk_id == "employment:4:0"


def test_hybrid_filtered_rerank_prefers_exact_application_heading_within_same_document() -> None:
    application = Chunk(
        chunk_id="cpa:2:0",
        document_id="cpa",
        section_heading="Section 2 Application",
        section_id="2",
        subsection_id=None,
        paragraph_id=None,
        text="This Act applies to all goods and services offered to consumers.",
        source_path="data/raw_law_pdfs/cpa.pdf",
        act_title="Consumer Protection Act 1999",
        source_file="cpa.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="2",
        document_aliases=("Consumer Protection Act 1999",),
    )
    other_written_law = Chunk(
        chunk_id="cpa:70:0",
        document_id="cpa",
        section_heading="Section 70 Application of other written law",
        section_id="70",
        subsection_id=None,
        paragraph_id=None,
        text="The provisions of this Act are in addition to any other written law.",
        source_path="data/raw_law_pdfs/cpa.pdf",
        act_title="Consumer Protection Act 1999",
        source_file="cpa.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="70",
        document_aliases=("Consumer Protection Act 1999",),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                application.text: [0.2, 0.8],
                other_written_law.text: [0.95, 0.05],
                "Which section of the Consumer Protection Act 1999 deals with application?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="Which section of the Consumer Protection Act 1999 deals with application?",
        chunks=[other_written_law, application],
        top_k=2,
        mode="hybrid_filtered_rerank",
    )

    assert results[0].chunk.chunk_id == "cpa:2:0"


def test_hybrid_filtered_rerank_prefers_revocation_heading_within_same_document() -> None:
    revocation = Chunk(
        chunk_id="mwo:6:0",
        document_id="mwo",
        section_heading="Section 6 Revocation",
        section_id="6",
        subsection_id=None,
        paragraph_id=None,
        text="The Minimum Wages Order 2022 is revoked.",
        source_path="data/raw_law_pdfs/minimum-wages-order.pdf",
        act_title="Minimum Wages Order 2024",
        source_file="minimum-wages-order.pdf",
        chunk_index=2,
        unit_type="section",
        unit_id="6",
        document_aliases=("Minimum Wages Order 2024", "P.U. (A) 376"),
    )
    rates = Chunk(
        chunk_id="mwo:3:0",
        document_id="mwo",
        section_heading="Section 3 Minimum wage rates with effect from 1 February 2025",
        section_id="3",
        subsection_id=None,
        paragraph_id=None,
        text="The minimum wage rates are as set out in this section.",
        source_path="data/raw_law_pdfs/minimum-wages-order.pdf",
        act_title="Minimum Wages Order 2024",
        source_file="minimum-wages-order.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="3",
        document_aliases=("Minimum Wages Order 2024", "P.U. (A) 376"),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                revocation.text: [0.2, 0.8],
                rates.text: [0.95, 0.05],
                "Which section of the Minimum Wages Order 2024 deals with revocation?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="Which section of the Minimum Wages Order 2024 deals with revocation?",
        chunks=[rates, revocation],
        top_k=2,
        mode="hybrid_filtered_rerank",
    )

    assert results[0].chunk.chunk_id == "mwo:6:0"


def test_hybrid_filtered_rerank_prefers_specific_annual_leave_heading_over_omnibus_heading() -> None:
    annual_leave = Chunk(
        chunk_id="employment:60e:0",
        document_id="employment",
        section_heading="Section 60E Annual leave",
        section_id="60E",
        subsection_id=None,
        paragraph_id=None,
        text="An employee shall be entitled to paid annual leave after twelve months of continuous service.",
        source_path="data/raw_law_pdfs/employment.pdf",
        act_title="Employment Act 1955",
        source_file="employment.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="60E",
        document_aliases=("Employment Act 1955", "Akta Kerja 1955"),
    )
    omnibus = Chunk(
        chunk_id="employment:100:0",
        document_id="employment",
        section_heading="Section 100 overtime, holidays, annual leave, and sick leave",
        section_id="100",
        subsection_id=None,
        paragraph_id=None,
        text="This section addresses overtime, holidays, annual leave, and sick leave in one provision.",
        source_path="data/raw_law_pdfs/employment.pdf",
        act_title="Employment Act 1955",
        source_file="employment.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="100",
        document_aliases=("Employment Act 1955", "Akta Kerja 1955"),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                annual_leave.text: [0.2, 0.8],
                omnibus.text: [0.95, 0.05],
                "Seksyen manakah berkaitan cuti tahunan dalam Akta Kerja 1955?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="Seksyen manakah berkaitan cuti tahunan dalam Akta Kerja 1955?",
        chunks=[omnibus, annual_leave],
        top_k=2,
        mode="hybrid_filtered_rerank",
    )

    assert results[0].chunk.chunk_id == "employment:60e:0"


def test_hybrid_filtered_rerank_prefers_commencement_heading_over_general_amendment() -> None:
    commencement = Chunk(
        chunk_id="a1727:1:0",
        document_id="a1727",
        section_heading="Section 1 Short title and commencement",
        section_id="1",
        subsection_id=None,
        paragraph_id=None,
        text="This Act comes into operation on a date appointed by the Minister.",
        source_path="data/raw_law_pdfs/pdpa-amendment.pdf",
        act_title="Personal Data Protection (Amendment) Act 2024",
        source_file="pdpa-amendment.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="1",
        document_aliases=("Personal Data Protection (Amendment) Act 2024", "Act A1727"),
    )
    general_amendment = Chunk(
        chunk_id="a1727:2:0",
        document_id="a1727",
        section_heading="Section 2 General amendment",
        section_id="2",
        subsection_id=None,
        paragraph_id=None,
        text="The principal Act is amended in the manner set out in this section.",
        source_path="data/raw_law_pdfs/pdpa-amendment.pdf",
        act_title="Personal Data Protection (Amendment) Act 2024",
        source_file="pdpa-amendment.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="2",
        document_aliases=("Personal Data Protection (Amendment) Act 2024", "Act A1727"),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                commencement.text: [0.2, 0.8],
                general_amendment.text: [0.95, 0.05],
                "When does the Personal Data Protection (Amendment) Act 2024 come into force?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="When does the Personal Data Protection (Amendment) Act 2024 come into force?",
        chunks=[general_amendment, commencement],
        top_k=2,
        mode="hybrid_filtered_rerank",
    )

    assert results[0].chunk.chunk_id == "a1727:1:0"


def test_hybrid_filtered_rerank_injects_interpretation_unit_within_requested_hierarchy() -> None:
    hierarchy_anchor = Chunk(
        chunk_id="constitution_ms:131:0",
        document_id="constitution_ms",
        section_heading="Perkara 131 Kuasa Parlimen mengenai perkhidmatan tertentu",
        section_id="131",
        subsection_id=None,
        paragraph_id=None,
        text=(
            "Perkara 131 Kuasa Parlimen mengenai perkhidmatan tertentu\n"
            "Parlimen boleh membuat undang-undang mengenai perkhidmatan tertentu.\n"
            "Bahagian X\n"
            "PERKHIDMATAN AWAM"
        ),
        source_path="data/raw_law_pdfs/Perlembagaan Persekutuan (Cetakan Semula 2020).pdf",
        act_title="Perlembagaan Persekutuan",
        source_file="Perlembagaan Persekutuan (Cetakan Semula 2020).pdf",
        chunk_index=0,
        unit_type="perkara",
        unit_id="131",
        document_aliases=("Perlembagaan Persekutuan", "Federal Constitution"),
    )
    target_interpretation = Chunk(
        chunk_id="constitution_ms:148:0",
        document_id="constitution_ms",
        section_heading="Perkara 148 Tafsiran Bahagian X",
        section_id="148",
        subsection_id=None,
        paragraph_id=None,
        text="Dalam Bahagian ini, tafsiran bagi perkataan yang digunakan hendaklah dipakai bagi Bahagian X.",
        source_path="data/raw_law_pdfs/Perlembagaan Persekutuan (Cetakan Semula 2020).pdf",
        act_title="Perlembagaan Persekutuan",
        source_file="Perlembagaan Persekutuan (Cetakan Semula 2020).pdf",
        chunk_index=1,
        unit_type="perkara",
        unit_id="148",
        document_aliases=("Perlembagaan Persekutuan", "Federal Constitution"),
    )
    in_scope_other = Chunk(
        chunk_id="constitution_ms:147:0",
        document_id="constitution_ms",
        section_heading="Perkara 147 Perlindungan hak pencen",
        section_id="147",
        subsection_id=None,
        paragraph_id=None,
        text="Hak pencen yang terakru hendaklah dilindungi.",
        source_path="data/raw_law_pdfs/Perlembagaan Persekutuan (Cetakan Semula 2020).pdf",
        act_title="Perlembagaan Persekutuan",
        source_file="Perlembagaan Persekutuan (Cetakan Semula 2020).pdf",
        chunk_index=2,
        unit_type="perkara",
        unit_id="147",
        document_aliases=("Perlembagaan Persekutuan", "Federal Constitution"),
    )
    next_hierarchy = Chunk(
        chunk_id="constitution_ms:149:0",
        document_id="constitution_ms",
        section_heading="Perkara 149 Perbuatan menentang negara",
        section_id="149",
        subsection_id=None,
        paragraph_id=None,
        text=(
            "Perkara 149 Perbuatan menentang negara\n"
            "Bahagian XI\n"
            "KUASA KHAS MENENTANG PERBUATAN SUBVERSIF"
        ),
        source_path="data/raw_law_pdfs/Perlembagaan Persekutuan (Cetakan Semula 2020).pdf",
        act_title="Perlembagaan Persekutuan",
        source_file="Perlembagaan Persekutuan (Cetakan Semula 2020).pdf",
        chunk_index=3,
        unit_type="perkara",
        unit_id="149",
        document_aliases=("Perlembagaan Persekutuan", "Federal Constitution"),
    )
    other_document = Chunk(
        chunk_id="employment:2:0",
        document_id="employment",
        section_heading="Section 2 Interpretation",
        section_id="2",
        subsection_id=None,
        paragraph_id=None,
        text="In this Act, employee means a person employed under a contract of service.",
        source_path="data/raw_law_pdfs/Akta Kerja 1955 (Akta 265).pdf",
        act_title="Employment Act 1955",
        source_file="Akta Kerja 1955 (Akta 265).pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="2",
        document_aliases=("Employment Act 1955", "Akta Kerja 1955"),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                hierarchy_anchor.text: [0.95, 0.05],
                target_interpretation.text: [0.2, 0.8],
                in_scope_other.text: [0.85, 0.15],
                next_hierarchy.text: [0.8, 0.2],
                other_document.text: [0.6, 0.4],
                "Apakah yang ditakrifkan di bawah tafsiran Bahagian X dalam Perlembagaan Persekutuan?": [0.9, 0.1],
            }
        ),
    )
    retriever = EmbeddingRetriever(embedder)

    results = retriever.search(
        query="Apakah yang ditakrifkan di bawah tafsiran Bahagian X dalam Perlembagaan Persekutuan?",
        chunks=[other_document, next_hierarchy, in_scope_other, target_interpretation, hierarchy_anchor],
        top_k=3,
        mode="hybrid_filtered_rerank",
    )

    assert results[0].chunk.chunk_id == "constitution_ms:148:0"
    assert {result.chunk.chunk_id for result in results} >= {
        "constitution_ms:148:0",
        "constitution_ms:131:0",
    }
