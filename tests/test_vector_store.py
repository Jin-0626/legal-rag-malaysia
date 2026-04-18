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
            "subsection_id": "1",
            "paragraph_id": "a",
            "source_file": "pdpa.pdf",
            "source_path": "data/raw_law_pdfs/pdpa.pdf",
            "chunk_index": 0,
            "text": "consent lawful basis processing",
        },
        {
            "chunk_id": "pdpa_2010:7:1",
            "document_id": "pdpa_2010",
            "act_title": "Personal Data Protection Act 2010",
            "act_number": "Act 709",
            "section_heading": "Section 7 Notice and Choice Principle",
            "section_id": "7",
            "subsection_id": "1",
            "paragraph_id": "b",
            "source_file": "pdpa.pdf",
            "source_path": "data/raw_law_pdfs/pdpa.pdf",
            "chunk_index": 1,
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

    results = vector_store.search("consent lawful basis", embedder, top_k=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "pdpa_2010:6:0"
    assert results[0].chunk.section_id == "6"
    assert results[0].chunk.subsection_id == "1"
    assert results[0].chunk.paragraph_id == "a"
    assert results[0].chunk.act_title == "Personal Data Protection Act 2010"
    assert results[0].score > 0.0
