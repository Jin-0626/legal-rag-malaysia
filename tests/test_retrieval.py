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
