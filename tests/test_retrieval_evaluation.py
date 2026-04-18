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
                        "expected_act_title": "Personal Data Protection Act 2010",
                        "expected_section_id": "6",
                        "expected_subsection_id": "1",
                    }
                ),
                json.dumps(
                    {
                        "query": "notice purpose",
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
        ),
        GoldQuery(
            query="notice purpose",
            expected_act_title="Personal Data Protection Act 2010",
            expected_section_id="7",
            expected_subsection_id=None,
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
                expected_act_title="Personal Data Protection Act 2010",
                expected_section_id="6",
                expected_subsection_id="1",
            ),
            GoldQuery(
                query="notice purpose",
                expected_act_title="Personal Data Protection Act 2010",
                expected_section_id="7",
                expected_subsection_id=None,
            ),
        ],
        vector_store=vector_store,
        embedder=embedder,
        top_k=3,
    )

    assert summary.total_queries == 2
    assert summary.hit_at_1 == 1.0
    assert summary.hit_at_3 == 1.0
    assert summary.error_breakdown == {category: 0 for category in MISS_CATEGORIES}
    assert summary.cases[0].matches[0].target_match is True
    assert summary.cases[0].matches[0].subsection_match is True
    assert summary.cases[1].matches[0].section_match is True
    assert summary.cases[1].matches[0].act_match is True

    report_path = tmp_path / "report.json"
    write_evaluation_summary(summary, report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["total_queries"] == 2
    assert payload["hit_at_1"] == 1.0
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
    )

    assert summary.hit_at_1 == 0.0
    assert summary.hit_at_3 == 0.0
    assert summary.error_breakdown == {
        "wrong_act": 1,
        "wrong_section": 1,
        "right_section_wrong_subsection": 1,
        "no_hit_in_top_k": 1,
    }
    assert [case.miss_category for case in summary.cases] == [
        "wrong_act",
        "wrong_section",
        "right_section_wrong_subsection",
        "no_hit_in_top_k",
    ]
