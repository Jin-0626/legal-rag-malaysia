import json
from pathlib import Path

from legal_rag.chunking.models import Chunk
from legal_rag.embeddings.embedder import OllamaEmbedder
from legal_rag.retrieval.vector_store import JsonlVectorStore, load_chunk_records


class FakeTransport:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed(self, *, texts: list[str], model: str) -> list[list[float]]:
        return [self.vectors[text] for text in texts]


def test_load_chunk_records_and_index_into_jsonl_vector_store(tmp_path: Path) -> None:
    chunk_file = tmp_path / "pdpa.jsonl"
    records = [
        {
            "chunk_id": "pdpa_2010:6:0",
            "document_id": "pdpa_2010",
            "act_title": "Personal Data Protection Act 2010",
            "act_number": "Act 709",
            "section_heading": "Section 6 General Principle",
            "section_id": "6",
            "unit_type": "section",
            "unit_id": "6",
            "subsection_id": "1",
            "paragraph_id": "a",
            "source_file": "pdpa.pdf",
            "source_path": "data/raw_law_pdfs/pdpa.pdf",
            "chunk_index": 0,
            "document_aliases": ["Personal Data Protection Act 2010", "PDPA"],
            "text": "consent lawful basis processing",
        },
        {
            "chunk_id": "pdpa_2010:7:1",
            "document_id": "pdpa_2010",
            "act_title": "Personal Data Protection Act 2010",
            "act_number": "Act 709",
            "section_heading": "Section 7 Notice and Choice Principle",
            "section_id": "7",
            "unit_type": "section",
            "unit_id": "7",
            "subsection_id": "1",
            "paragraph_id": "b",
            "source_file": "pdpa.pdf",
            "source_path": "data/raw_law_pdfs/pdpa.pdf",
            "chunk_index": 1,
            "document_aliases": ["Personal Data Protection Act 2010", "PDPA"],
            "text": "notice disclosure purpose statement",
        },
    ]
    chunk_file.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    chunks = load_chunk_records(chunk_file)

    assert len(chunks) == 2
    assert chunks[0].act_title == "Personal Data Protection Act 2010"
    assert chunks[0].paragraph_id == "a"
    assert chunks[0].unit_type == "section"
    assert chunks[0].unit_id == "6"
    assert chunks[0].document_aliases == ("Personal Data Protection Act 2010", "PDPA")

    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                "consent lawful basis processing": [1.0, 0.0],
                "notice disclosure purpose statement": [0.0, 1.0],
                "consent lawful basis": [0.9, 0.1],
            }
        ),
    )
    vector_store = JsonlVectorStore(tmp_path / "pdpa.vectors.jsonl")
    count = vector_store.index_chunks(chunks, embedder)

    assert count == 2
    stored_records = vector_store.load_records()
    assert len(stored_records) == 2
    assert stored_records[0].act_number == "Act 709"
    assert stored_records[0].source_file == "pdpa.pdf"
    assert stored_records[0].unit_type == "section"
    assert stored_records[0].unit_id == "6"
    assert stored_records[0].document_aliases == ["Personal Data Protection Act 2010", "PDPA"]


def test_jsonl_vector_store_search_preserves_metadata_traceability(tmp_path: Path) -> None:
    chunks = [
        Chunk(
            chunk_id="pdpa_2010:6:0",
            document_id="pdpa_2010",
            section_heading="Section 6 General Principle",
            section_id="6",
            subsection_id="1",
            paragraph_id="a",
            text="consent lawful basis processing",
            source_path="data/raw_law_pdfs/pdpa.pdf",
            act_title="Personal Data Protection Act 2010",
            act_number="Act 709",
            source_file="pdpa.pdf",
            chunk_index=0,
            unit_type="section",
            unit_id="6",
            document_aliases=("Personal Data Protection Act 2010", "PDPA"),
        ),
        Chunk(
            chunk_id="pdpa_2010:7:1",
            document_id="pdpa_2010",
            section_heading="Section 7 Notice and Choice Principle",
            section_id="7",
            subsection_id="1",
            paragraph_id="b",
            text="notice disclosure purpose statement",
            source_path="data/raw_law_pdfs/pdpa.pdf",
            act_title="Personal Data Protection Act 2010",
            act_number="Act 709",
            source_file="pdpa.pdf",
            chunk_index=1,
            unit_type="section",
            unit_id="7",
            document_aliases=("Personal Data Protection Act 2010", "PDPA"),
        ),
    ]
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                "consent lawful basis processing": [1.0, 0.0],
                "notice disclosure purpose statement": [0.0, 1.0],
                "consent lawful basis": [0.9, 0.1],
            }
        ),
    )
    vector_store = JsonlVectorStore(tmp_path / "pdpa.vectors.jsonl")
    vector_store.index_chunks(chunks, embedder)

    results = vector_store.search("consent lawful basis", embedder, top_k=1, mode="hybrid")

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "pdpa_2010:6:0"
    assert results[0].chunk.section_id == "6"
    assert results[0].chunk.unit_type == "section"
    assert results[0].chunk.unit_id == "6"
    assert results[0].chunk.subsection_id == "1"
    assert results[0].chunk.paragraph_id == "a"
    assert results[0].chunk.act_title == "Personal Data Protection Act 2010"
    assert "PDPA" in results[0].chunk.document_aliases
    assert results[0].score > 0.0


def test_jsonl_vector_store_exact_lookup_returns_correct_article_chunk(tmp_path: Path) -> None:
    chunks = [
        Chunk(
            chunk_id="constitution:5:0",
            document_id="constitution",
            section_heading="Article 5 Liberty of the person",
            section_id="5",
            subsection_id=None,
            paragraph_id=None,
            text="No person shall be deprived of life or personal liberty save in accordance with law.",
            source_path="data/raw_law_pdfs/federal-constitution.pdf",
            act_title="Federal Constitution",
            act_number="P.U.",
            source_file="federal-constitution.pdf",
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
            text="The following personal data protection principles apply.",
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
                chunks[0].text: [0.2, 0.8],
                chunks[1].text: [0.8, 0.2],
                "Article 5 Federal Constitution": [0.9, 0.1],
            }
        ),
    )
    vector_store = JsonlVectorStore(tmp_path / "legal-corpus.vectors.jsonl")
    vector_store.index_chunks(chunks, embedder)

    results = vector_store.search("Article 5 Federal Constitution", embedder, top_k=2, mode="hybrid")

    assert [result.chunk.chunk_id for result in results] == ["constitution:5:0"]
    assert results[0].chunk.section_heading == "Article 5 Liberty of the person"
