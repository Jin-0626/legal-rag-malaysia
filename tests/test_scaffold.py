from pathlib import Path

from legal_rag.chunking.section_chunker import chunk_section_text
from legal_rag.config.settings import build_settings
from legal_rag.embeddings.base import OllamaEmbedder
from legal_rag.generation.grounded import GroundedAnswerGenerator
from legal_rag.ingestion.pdf_sources import discover_law_pdfs
from legal_rag.retrieval.in_memory import SimpleRetriever


def test_build_settings_prepares_pdf_directories(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)

    assert settings.raw_law_pdfs_dir.exists()
    assert settings.processed_dir.exists()
    assert settings.embeddings_dir.exists()


def test_discover_law_pdfs_returns_stable_metadata(tmp_path: Path) -> None:
    nested_dir = tmp_path / "acts"
    nested_dir.mkdir()
    pdf_path = nested_dir / "Penal Code.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    sources = discover_law_pdfs(tmp_path)

    assert sources == [
        sources[0].__class__(
            document_id="penal_code",
            title="Penal Code",
            path=str(pdf_path),
        )
    ]


def test_chunk_retrieve_and_generate_grounded_answer() -> None:
    class FakeTransport:
        def embed(self, *, texts: list[str], model: str) -> list[list[float]]:
            vectors = {
                "Section 6\nPersonal data shall not be processed without consent unless a lawful basis applies under the Act.": [1.0, 0.0, 0.0],
                "consent lawful basis": [1.0, 0.0, 0.0],
            }
            return [vectors.get(text, [0.0, 1.0, 0.0]) for text in texts]

    section_text = (
        "Personal data shall not be processed without consent unless a lawful "
        "basis applies under the Act."
    )
    chunks = chunk_section_text(
        document_id="pdpa_2010",
        section_heading="Section 6",
        text=section_text,
        source_path="data/raw_law_pdfs/pdpa.pdf",
        max_words=32,
        overlap_words=4,
    )

    retriever = SimpleRetriever(
        OllamaEmbedder(model="test-model", transport=FakeTransport())
    )
    results = retriever.search("consent lawful basis", chunks, top_k=2)
    answer = GroundedAnswerGenerator().answer("What does Section 6 require?", results)

    assert chunks
    assert all(chunk.section_heading == "Section 6" for chunk in chunks)
    assert results
    assert answer.grounded is True
    assert answer.citations == ["pdpa_2010 Section 6"]


def test_ollama_embedder_is_stable_for_same_input() -> None:
    class FakeTransport:
        def __init__(self) -> None:
            self.calls = 0

        def embed(self, *, texts: list[str], model: str) -> list[list[float]]:
            self.calls += 1
            return [[float(len(text)), float(len(text.split()))] for text in texts]

    transport = FakeTransport()
    embedder = OllamaEmbedder(model="test-model", transport=transport)

    first = embedder.embed(["sample legal text"])
    second = embedder.embed(["sample legal text"])

    assert first == second
    assert transport.calls == 1
