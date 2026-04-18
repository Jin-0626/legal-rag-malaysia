from legal_rag.chunking.models import Chunk
from legal_rag.embeddings.embedder import OllamaEmbedder
from legal_rag.retrieval.in_memory import SimpleVectorIndex


class FakeTransport:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def embed(self, *, texts: list[str], model: str) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.vectors[text] for text in texts]


def test_ollama_embedder_reuses_cached_embeddings_for_same_input() -> None:
    transport = FakeTransport({"same text": [1.0, 2.0, 3.0]})
    embedder = OllamaEmbedder(model="test-model", transport=transport)

    first = embedder.embed(["same text", "same text"])
    second = embedder.embed(["same text"])

    assert first == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    assert second == [[1.0, 2.0, 3.0]]
    assert transport.calls == [["same text"]]


def test_ollama_embedder_returns_consistent_vector_shape() -> None:
    transport = FakeTransport(
        {
            "alpha": [1.0, 2.0, 3.0],
            "beta": [4.0, 5.0, 6.0],
        }
    )
    embedder = OllamaEmbedder(model="test-model", transport=transport)

    vectors = embedder.embed(["alpha", "beta"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 3
    assert len(vectors[1]) == 3


def test_simple_vector_index_ranks_chunks_by_embedding_similarity() -> None:
    chunk_a = Chunk(
        chunk_id="pdpa_2010:6:0",
        document_id="pdpa_2010",
        section_heading="Section 6 General Principle",
        section_id="6",
        subsection_id="1",
        paragraph_id=None,
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
    transport = FakeTransport(
        {
            "consent lawful basis processing": [1.0, 0.0],
            "notice disclosure purpose statement": [0.0, 1.0],
            "consent lawful basis": [0.9, 0.1],
        }
    )
    embedder = OllamaEmbedder(model="test-model", transport=transport)
    index = SimpleVectorIndex(embedder)

    index.add([chunk_a, chunk_b])
    results = index.search("consent lawful basis", top_k=2)

    assert [result.chunk.chunk_id for result in results] == ["pdpa_2010:6:0", "pdpa_2010:7:0"]
    assert results[0].score > results[1].score
