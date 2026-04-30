import json
from pathlib import Path

from legal_rag.chunking.models import Chunk
from legal_rag.embeddings.embedder import OllamaEmbedder
from legal_rag.retrieval import (
    GoldQuery,
    JsonlVectorStore,
    MISS_CATEGORIES,
    evaluate_retrieval,
    load_gold_queries,
    write_evaluation_summary,
)


class FakeTransport:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed(self, *, texts: list[str], model: str) -> list[list[float]]:
        return [self.vectors[text] for text in texts]


def test_load_gold_queries_and_evaluate_hit_metrics(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "query": "consent requirement",
                        "query_type": "obligation",
                        "expected_act_title": "Personal Data Protection Act 2010",
                        "expected_section_id": "6",
                        "expected_subsection_id": "1",
                    }
                ),
                json.dumps(
                    {
                        "query": "notice purpose",
                        "query_type": "definition",
                        "expected_act_title": "Personal Data Protection Act 2010",
                        "expected_section_id": "7",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    gold_queries = load_gold_queries(gold_path)

    assert gold_queries == [
        GoldQuery(
            query="consent requirement",
            expected_act_title="Personal Data Protection Act 2010",
            expected_section_id="6",
            expected_subsection_id="1",
            query_type="obligation",
        ),
        GoldQuery(
            query="notice purpose",
            expected_act_title="Personal Data Protection Act 2010",
            expected_section_id="7",
            expected_subsection_id=None,
            query_type="definition",
        ),
    ]


def test_retrieval_evaluation_reports_metadata_matches_and_hits(tmp_path: Path) -> None:
    chunks = [
        Chunk(
            chunk_id="pdpa_2010:6:0",
            document_id="pdpa_2010",
            section_heading="Section 6 General Principle",
            section_id="6",
            subsection_id="1",
            paragraph_id=None,
            text="consent lawful basis processing",
            source_path="data/raw_law_pdfs/pdpa.pdf",
            act_title="Personal Data Protection Act 2010",
            act_number="Act 709",
            source_file="pdpa.pdf",
            chunk_index=0,
            unit_type="section",
            unit_id="6",
        ),
        Chunk(
            chunk_id="pdpa_2010:7:1",
            document_id="pdpa_2010",
            section_heading="Section 7 Notice and Choice Principle",
            section_id="7",
            subsection_id="1",
            paragraph_id=None,
            text="notice disclosure purpose statement",
            source_path="data/raw_law_pdfs/pdpa.pdf",
            act_title="Personal Data Protection Act 2010",
            act_number="Act 709",
            source_file="pdpa.pdf",
            chunk_index=1,
            unit_type="section",
            unit_id="7",
        ),
    ]
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                "consent lawful basis processing": [1.0, 0.0],
                "notice disclosure purpose statement": [0.0, 1.0],
                "consent requirement": [1.0, 0.0],
                "notice purpose": [0.0, 1.0],
            }
        ),
    )
    vector_store = JsonlVectorStore(tmp_path / "pdpa.vectors.jsonl")
    vector_store.index_chunks(chunks, embedder)

    summary = evaluate_retrieval(
        gold_queries=[
            GoldQuery(
                query="consent requirement",
                query_type="obligation",
                expected_act_title="Personal Data Protection Act 2010",
                expected_section_id="6",
                expected_subsection_id="1",
            ),
            GoldQuery(
                query="notice purpose",
                query_type="definition",
                expected_act_title="Personal Data Protection Act 2010",
                expected_section_id="7",
                expected_subsection_id=None,
            ),
        ],
        vector_store=vector_store,
        embedder=embedder,
        top_k=3,
        mode="hybrid",
    )

    assert summary.mode == "hybrid"
    assert summary.total_queries == 2
    assert summary.hit_at_1 == 1.0
    assert summary.hit_at_3 == 1.0
    assert summary.wrong_section_rate == 0.0
    assert summary.error_breakdown == {category: 0 for category in MISS_CATEGORIES}
    assert summary.cases[0].query_type == "obligation"
    assert summary.cases[0].matches[0].target_match is True
    assert summary.cases[0].matches[0].subsection_match is True
    assert summary.cases[1].matches[0].section_match is True
    assert summary.cases[1].matches[0].act_match is True

    report_path = tmp_path / "report.json"
    write_evaluation_summary(summary, report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "hybrid"
    assert payload["total_queries"] == 2
    assert payload["hit_at_1"] == 1.0
    assert payload["wrong_section_rate"] == 0.0
    assert payload["error_breakdown"] == {category: 0 for category in MISS_CATEGORIES}
    assert payload["cases"][0]["matches"][0]["chunk_id"] == "pdpa_2010:6:0"


def test_retrieval_evaluation_classifies_miss_categories(tmp_path: Path) -> None:
    chunks = [
        Chunk(
            chunk_id="constitution:4:0",
            document_id="constitution",
            section_heading="Article 4 Supreme Law",
            section_id="4",
            subsection_id="1",
            paragraph_id=None,
            text="supreme law constitution article four",
            source_path="data/raw_law_pdfs/constitution.pdf",
            act_title="Federal Constitution",
            act_number="Act A",
            source_file="constitution.pdf",
            chunk_index=0,
            unit_type="article",
            unit_id="4",
        ),
        Chunk(
            chunk_id="pdpa_2010:7:0",
            document_id="pdpa_2010",
            section_heading="Section 7 Notice and Choice Principle",
            section_id="7",
            subsection_id="1",
            paragraph_id=None,
            text="notice purpose disclosure statement",
            source_path="data/raw_law_pdfs/pdpa.pdf",
            act_title="Personal Data Protection Act 2010",
            act_number="Act 709",
            source_file="pdpa.pdf",
            chunk_index=1,
            unit_type="section",
            unit_id="7",
        ),
        Chunk(
            chunk_id="pdpa_2010:6:1",
            document_id="pdpa_2010",
            section_heading="Section 6 General Principle",
            section_id="6",
            subsection_id="2",
            paragraph_id=None,
            text="consent processing exemption subsection two",
            source_path="data/raw_law_pdfs/pdpa.pdf",
            act_title="Personal Data Protection Act 2010",
            act_number="Act 709",
            source_file="pdpa.pdf",
            chunk_index=2,
            unit_type="section",
            unit_id="6",
        ),
    ]
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                "supreme law constitution article four": [1.0, 0.0, 0.0, 0.0],
                "notice purpose disclosure statement": [0.0, 1.0, 0.0, 0.0],
                "consent processing exemption subsection two": [0.0, 0.0, 1.0, 0.0],
                "data rights query": [1.0, 0.0, 0.0, 0.0],
                "notice principle query": [0.0, 1.0, 0.0, 0.0],
                "consent subsection one query": [0.0, 0.0, 1.0, 0.0],
                "no matching vector query": [0.0, 0.0, 0.0, 1.0],
            }
        ),
    )
    vector_store = JsonlVectorStore(tmp_path / "misses.vectors.jsonl")
    vector_store.index_chunks(chunks, embedder)

    summary = evaluate_retrieval(
        gold_queries=[
            GoldQuery(
                query="data rights query",
                expected_act_title="Personal Data Protection Act 2010",
                expected_section_id="5",
            ),
            GoldQuery(
                query="notice principle query",
                expected_act_title="Personal Data Protection Act 2010",
                expected_section_id="6",
            ),
            GoldQuery(
                query="consent subsection one query",
                expected_act_title="Personal Data Protection Act 2010",
                expected_section_id="6",
                expected_subsection_id="1",
            ),
            GoldQuery(
                query="no matching vector query",
                expected_act_title="Personal Data Protection Act 2010",
                expected_section_id="3",
            ),
        ],
        vector_store=vector_store,
        embedder=embedder,
        top_k=3,
        mode="embedding",
    )

    assert summary.mode == "embedding"
    assert summary.hit_at_1 == 0.0
    assert summary.hit_at_3 == 0.0
    assert summary.wrong_section_rate == 0.5
    assert summary.error_breakdown == {
        "wrong_act": 0,
        "wrong_section": 2,
        "right_section_wrong_subsection": 1,
        "no_hit_in_top_k": 1,
    }
    assert [case.miss_category for case in summary.cases] == [
        "wrong_section",
        "wrong_section",
        "right_section_wrong_subsection",
        "no_hit_in_top_k",
    ]


def test_hybrid_mode_outperforms_embedding_mode_on_direct_lookup_queries(tmp_path: Path) -> None:
    chunks = [
        Chunk(
            chunk_id="constitution:5:0",
            document_id="constitution",
            section_heading="Article 5 Liberty of the person",
            section_id="5",
            subsection_id=None,
            paragraph_id=None,
            text="No person shall be deprived of his life or personal liberty save in accordance with law.",
            source_path="data/raw_law_pdfs/federal-constitution.pdf",
            act_title="Federal Constitution",
            act_number="P.U.",
            source_file="Federal Constitution.pdf",
            chunk_index=0,
            unit_type="article",
            unit_id="5",
        ),
        Chunk(
            chunk_id="pdpa:5:0",
            document_id="pdpa",
            section_heading="Section 5 Personal data protection principles",
            section_id="5",
            subsection_id=None,
            paragraph_id=None,
            text="The following personal data protection principles shall apply.",
            source_path="data/raw_law_pdfs/pdpa.pdf",
            act_title="Personal Data Protection Act 2010",
            act_number="Act 709",
            source_file="pdpa.pdf",
            chunk_index=1,
            unit_type="section",
            unit_id="5",
        ),
    ]
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                chunks[0].text: [0.1, 0.9],
                chunks[1].text: [0.9, 0.1],
                "Article 5 Federal Constitution": [0.95, 0.05],
            }
        ),
    )
    vector_store = JsonlVectorStore(tmp_path / "legal-corpus.vectors.jsonl")
    vector_store.index_chunks(chunks, embedder)
    gold_queries = [
        GoldQuery(
            query="Article 5 Federal Constitution",
            query_type="direct_lookup",
            expected_act_title="Federal Constitution",
            expected_section_id="5",
        )
    ]

    embedding_summary = evaluate_retrieval(
        gold_queries=gold_queries,
        vector_store=vector_store,
        embedder=embedder,
        top_k=3,
        mode="embedding",
    )
    hybrid_summary = evaluate_retrieval(
        gold_queries=gold_queries,
        vector_store=vector_store,
        embedder=embedder,
        top_k=3,
        mode="hybrid",
    )

    assert embedding_summary.hit_at_1 <= hybrid_summary.hit_at_1
    assert hybrid_summary.hit_at_1 == 1.0
    assert hybrid_summary.cases[0].matches[0].act_title == "Federal Constitution"


def test_hybrid_rerank_mode_improves_top_1_for_definition_and_cross_reference_cases(tmp_path: Path) -> None:
    chunks = [
        Chunk(
            chunk_id="pdpa:4:0",
            document_id="pdpa",
            section_heading="Section 4 Interpretation",
            section_id="4",
            subsection_id=None,
            paragraph_id=None,
            text="In this Act, sensitive personal data means personal data consisting of information as to health.",
            source_path="data/raw_law_pdfs/pdpa.pdf",
            act_title="Personal Data Protection Act 2010",
            act_number="Act 709",
            source_file="pdpa.pdf",
            chunk_index=0,
            unit_type="section",
            unit_id="4",
        ),
        Chunk(
            chunk_id="pdpa:40:0",
            document_id="pdpa",
            section_heading="Section 40 Processing of sensitive personal data",
            section_id="40",
            subsection_id=None,
            paragraph_id=None,
            text="Processing of sensitive personal data is prohibited except under this section and section 129.",
            source_path="data/raw_law_pdfs/pdpa.pdf",
            act_title="Personal Data Protection Act 2010",
            act_number="Act 709",
            source_file="pdpa.pdf",
            chunk_index=1,
            unit_type="section",
            unit_id="40",
        ),
        Chunk(
            chunk_id="constitution:8:0",
            document_id="constitution",
            section_heading="Article 8 Equality",
            section_id="8",
            subsection_id=None,
            paragraph_id=None,
            text="All persons are equal before the law and entitled to the equal protection of the law.",
            source_path="data/raw_law_pdfs/constitution.pdf",
            act_title="Federal Constitution",
            act_number="P.U.",
            source_file="constitution.pdf",
            chunk_index=2,
            unit_type="article",
            unit_id="8",
        ),
        Chunk(
            chunk_id="constitution:122:0",
            document_id="constitution",
            section_heading="Article 122 Constitution of Federal Court",
            section_id="122",
            subsection_id=None,
            paragraph_id=None,
            text="Article 8, Article 121 and Article 128 are referenced in this constitutional court provision.",
            source_path="data/raw_law_pdfs/constitution.pdf",
            act_title="Federal Constitution",
            act_number="P.U.",
            source_file="constitution.pdf",
            chunk_index=3,
            unit_type="article",
            unit_id="122",
        ),
    ]
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                chunks[0].text: [0.1, 0.9],
                chunks[1].text: [1.0, 0.0],
                chunks[2].text: [0.1, 0.8],
                chunks[3].text: [0.9, 0.1],
                "How does the PDPA define sensitive personal data?": [1.0, 0.0],
                "Which article is titled Equality in the Federal Constitution?": [1.0, 0.0],
            }
        ),
    )
    vector_store = JsonlVectorStore(tmp_path / "legal-corpus.vectors.jsonl")
    vector_store.index_chunks(chunks, embedder)
    gold_queries = [
        GoldQuery(
            query="How does the PDPA define sensitive personal data?",
            query_type="definition",
            expected_act_title="Personal Data Protection Act 2010",
            expected_section_id="4",
        ),
        GoldQuery(
            query="Which article is titled Equality in the Federal Constitution?",
            query_type="general",
            expected_act_title="Federal Constitution",
            expected_section_id="8",
        ),
    ]

    hybrid_summary = evaluate_retrieval(
        gold_queries=gold_queries,
        vector_store=vector_store,
        embedder=embedder,
        top_k=3,
        mode="hybrid",
    )
    rerank_summary = evaluate_retrieval(
        gold_queries=gold_queries,
        vector_store=vector_store,
        embedder=embedder,
        top_k=3,
        mode="hybrid_rerank",
    )

    assert hybrid_summary.hit_at_1 == 0.5
    assert rerank_summary.hit_at_1 == 1.0
    assert rerank_summary.wrong_section_rate == 0.0


def test_hybrid_filtered_rerank_improves_candidate_pool_quality(tmp_path: Path) -> None:
    query = "Which article is titled Equality in the Federal Constitution?"
    target_chunk = Chunk(
        chunk_id="constitution:8:0",
        document_id="constitution",
        section_heading="Article 8 Equality",
        section_id="8",
        subsection_id=None,
        paragraph_id=None,
        text="All persons are equal before the law and entitled to the equal protection of the law.",
        source_path="data/raw_law_pdfs/constitution.pdf",
        act_title="Federal Constitution",
        act_number="P.U.",
        source_file="constitution.pdf",
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
            source_path="data/raw_law_pdfs/constitution.pdf",
            act_title="Federal Constitution",
            act_number="P.U.",
            source_file="constitution.pdf",
            chunk_index=index,
            unit_type="article",
            unit_id=str(index),
        )
        for index in range(20, 31)
    ]
    chunks = [target_chunk, *distractors]
    vectors = {
        query: [1.0, 0.0],
        target_chunk.text: [0.05, 0.95],
    }
    for distractor in distractors:
        vectors[distractor.text] = [1.0, 0.0]
    embedder = OllamaEmbedder(model="test-model", transport=FakeTransport(vectors))
    vector_store = JsonlVectorStore(tmp_path / "legal-corpus.vectors.jsonl")
    vector_store.index_chunks(chunks, embedder)
    gold_queries = [
        GoldQuery(
            query=query,
            query_type="direct_lookup",
            expected_act_title="Federal Constitution",
            expected_section_id="8",
        )
    ]

    rerank_summary = evaluate_retrieval(
        gold_queries=gold_queries,
        vector_store=vector_store,
        embedder=embedder,
        top_k=3,
        mode="hybrid_rerank",
    )
    filtered_summary = evaluate_retrieval(
        gold_queries=gold_queries,
        vector_store=vector_store,
        embedder=embedder,
        top_k=3,
        mode="hybrid_filtered_rerank",
    )

    assert rerank_summary.hit_at_1 == 0.0
    assert filtered_summary.hit_at_1 == 1.0
    assert filtered_summary.cases[0].matches[0].chunk_id == "constitution:8:0"
    assert filtered_summary.cases[0].matches[0].act_match is True
