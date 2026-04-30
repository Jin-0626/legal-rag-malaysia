from legal_rag.chunking.models import Chunk
from legal_rag.embeddings.embedder import EmbeddedChunk, OllamaEmbedder
from legal_rag.graph import build_legal_graph, search_graph
from legal_rag.workflows import graph_supported_search


class FakeTransport:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed(self, *, texts: list[str], model: str) -> list[list[float]]:
        return [self.vectors[text] for text in texts]


def _entries_for(*chunks: Chunk) -> list[EmbeddedChunk]:
    return [EmbeddedChunk(chunk=chunk, embedding=[1.0, 0.0]) for chunk in chunks]


def test_build_legal_graph_routes_hierarchy_query_to_first_unit_of_part() -> None:
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
        source_file="Federal Constitution.pdf",
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
        text="Article 5 Liberty of the person\nNo person shall be deprived of life save in accordance with law.",
        source_path="data/raw_law_pdfs/federal-constitution.pdf",
        act_title="Federal Constitution",
        source_file="Federal Constitution.pdf",
        chunk_index=1,
        unit_type="article",
        unit_id="5",
        document_aliases=("Federal Constitution", "Constitution of Malaysia"),
    )

    graph = build_legal_graph([article_four_tail, article_five])

    results = search_graph(graph, "Which Article begins Part II in the Federal Constitution?", top_k=1)

    assert results[0].chunk.chunk_id == "constitution:5:0"
    assert "hierarchy" in results[0].reason


def test_build_legal_graph_extracts_amendment_linkage_to_principal_act_section() -> None:
    principal = Chunk(
        chunk_id="pdpa:4:0",
        document_id="pdpa",
        section_heading="Section 4 Interpretation",
        section_id="4",
        subsection_id=None,
        paragraph_id=None,
        text="Interpretation provisions.",
        source_path="data/raw_law_pdfs/pdpa.pdf",
        act_title="Personal Data Protection Act 2010",
        act_number="Act 709",
        source_file="pdpa.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="4",
        document_aliases=("Personal Data Protection Act 2010", "PDPA", "Act 709"),
    )
    amendment = Chunk(
        chunk_id="a1727:3:0",
        document_id="a1727",
        section_heading="Section 3 Amendment of section 4",
        section_id="3",
        subsection_id=None,
        paragraph_id=None,
        text="Section 3 Amendment of section 4\nThe principal Act is amended in section 4.",
        source_path="data/raw_law_pdfs/Act-A1727.pdf",
        act_title="Personal Data Protection (Amendment) Act 2024",
        act_number="Act A1727",
        source_file="Act-A1727.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="3",
        document_aliases=("Personal Data Protection (Amendment) Act 2024", "PDPA Amendment Act 2024", "Act A1727"),
    )

    graph = build_legal_graph([principal, amendment])

    results = search_graph(
        graph,
        "Which section of the PDPA Amendment Act 2024 amends section 4 of the principal Act?",
        top_k=1,
    )

    assert results[0].chunk.chunk_id == "a1727:3:0"
    assert results[0].reason == "graph: amendment linkage"


def test_build_legal_graph_supports_explicit_cross_reference_navigation() -> None:
    target = Chunk(
        chunk_id="pdpa:4:0",
        document_id="pdpa",
        section_heading="Section 4 Interpretation",
        section_id="4",
        subsection_id=None,
        paragraph_id=None,
        text="Interpretation provisions.",
        source_path="data/raw_law_pdfs/pdpa.pdf",
        act_title="Personal Data Protection Act 2010",
        source_file="pdpa.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="4",
        document_aliases=("Personal Data Protection Act 2010", "PDPA"),
    )
    referring = Chunk(
        chunk_id="pdpa:70:0",
        document_id="pdpa",
        section_heading="Section 70 Advisory Committee",
        section_id="70",
        subsection_id=None,
        paragraph_id=None,
        text="The Advisory Committee established under section 70 shall have regard to section 4.",
        source_path="data/raw_law_pdfs/pdpa.pdf",
        act_title="Personal Data Protection Act 2010",
        source_file="pdpa.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="70",
        document_aliases=("Personal Data Protection Act 2010", "PDPA"),
    )

    graph = build_legal_graph([target, referring])

    results = search_graph(graph, "Which section refers to section 4 in the PDPA?", top_k=1)

    assert results[0].chunk.chunk_id == "pdpa:70:0"


def test_graph_supported_search_falls_back_to_hybrid_for_ordinary_direct_lookup() -> None:
    direct = Chunk(
        chunk_id="pdpa:4:0",
        document_id="pdpa",
        section_heading="Section 4 Interpretation",
        section_id="4",
        subsection_id=None,
        paragraph_id=None,
        text="Interpretation provisions.",
        source_path="data/raw_law_pdfs/pdpa.pdf",
        act_title="Personal Data Protection Act 2010",
        source_file="pdpa.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="4",
        document_aliases=("Personal Data Protection Act 2010", "PDPA"),
    )
    distractor = Chunk(
        chunk_id="pdpa:5:0",
        document_id="pdpa",
        section_heading="Section 5 General Principle",
        section_id="5",
        subsection_id=None,
        paragraph_id=None,
        text="General principle provisions.",
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
                "Interpretation provisions.": [0.2, 0.8],
                "General principle provisions.": [0.8, 0.2],
                "Section 4 PDPA": [0.8, 0.2],
            }
        ),
    )
    entries = _entries_for(direct, distractor)
    graph = build_legal_graph([direct, distractor])

    results = graph_supported_search(
        entries=entries,
        embedder=embedder,
        graph=graph,
        query="Section 4 PDPA",
        top_k=1,
        mode="graph_supported",
    )

    assert results[0].chunk.chunk_id == "pdpa:4:0"


def test_graph_supported_search_prefers_commencement_section_for_amendment_query() -> None:
    commencement = Chunk(
        chunk_id="a1727:1:0",
        document_id="a1727",
        section_heading="Section 1 Citation and commencement",
        section_id="1",
        subsection_id=None,
        paragraph_id=None,
        text="This Act may be cited as the Personal Data Protection (Amendment) Act 2024 and comes into operation on a date appointed by the Minister.",
        source_path="data/raw_law_pdfs/Act-A1727.pdf",
        act_title="Personal Data Protection (Amendment) Act 2024",
        source_file="Act-A1727.pdf",
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
        text="The principal Act is amended in accordance with this Act.",
        source_path="data/raw_law_pdfs/Act-A1727.pdf",
        act_title="Personal Data Protection (Amendment) Act 2024",
        source_file="Act-A1727.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="2",
        document_aliases=("Personal Data Protection (Amendment) Act 2024", "Act A1727"),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                commencement.text: [0.1, 0.9],
                general_amendment.text: [0.9, 0.1],
                "When does the Personal Data Protection (Amendment) Act 2024 come into force?": [0.9, 0.1],
            }
        ),
    )
    entries = _entries_for(commencement, general_amendment)
    graph = build_legal_graph([commencement, general_amendment])

    results = graph_supported_search(
        entries=entries,
        embedder=embedder,
        graph=graph,
        query="When does the Personal Data Protection (Amendment) Act 2024 come into force?",
        top_k=1,
        mode="graph_supported",
    )

    assert results[0].chunk.chunk_id == "a1727:1:0"


def test_graph_supported_search_keeps_hybrid_document_when_graph_doc_disagrees() -> None:
    principal = Chunk(
        chunk_id="pdpa:5:0",
        document_id="pdpa",
        section_heading="Section 5 Personal Data Protection Principles",
        section_id="5",
        subsection_id=None,
        paragraph_id=None,
        text="Principal Act provisions.",
        source_path="data/raw_law_pdfs/pdpa.pdf",
        act_title="Personal Data Protection Act 2010",
        source_file="pdpa.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="5",
        document_aliases=("Personal Data Protection Act 2010", "PDPA"),
    )
    amendment = Chunk(
        chunk_id="a1727:9:0",
        document_id="a1727",
        section_heading="Section 9 New section 43a",
        section_id="9",
        subsection_id=None,
        paragraph_id=None,
        text="The principal Act is amended by inserting after section 43 the following section.",
        source_path="data/raw_law_pdfs/Act-A1727.pdf",
        act_title="Personal Data Protection (Amendment) Act 2024",
        source_file="Act-A1727.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="9",
        document_aliases=("Personal Data Protection (Amendment) Act 2024", "PDPA Amendment Act 2024"),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                "Principal Act provisions.": [0.9, 0.1],
                "The principal Act is amended by inserting after section 43 the following section.": [0.1, 0.9],
                "Seksyen manakah memperkenalkan hak pemindahan data dalam Akta Pindaan PDPA 2024?": [0.1, 0.9],
            }
        ),
    )
    entries = _entries_for(principal, amendment)
    graph = build_legal_graph([principal, amendment])

    results = graph_supported_search(
        entries=entries,
        embedder=embedder,
        graph=graph,
        query="Seksyen manakah memperkenalkan hak pemindahan data dalam Akta Pindaan PDPA 2024?",
        top_k=1,
        mode="graph_supported",
    )

    assert results[0].chunk.chunk_id == "a1727:9:0"


def test_graph_rerank_mode_promotes_graph_backed_amendment_candidate() -> None:
    generic = Chunk(
        chunk_id="a1727:2:0",
        document_id="a1727",
        section_heading="Section 2 General amendment",
        section_id="2",
        subsection_id=None,
        paragraph_id=None,
        text="The principal Act is amended in accordance with this Act.",
        source_path="data/raw_law_pdfs/Act-A1727.pdf",
        act_title="Personal Data Protection (Amendment) Act 2024",
        source_file="Act-A1727.pdf",
        chunk_index=0,
        unit_type="section",
        unit_id="2",
        document_aliases=("Personal Data Protection (Amendment) Act 2024", "Act A1727"),
    )
    specific = Chunk(
        chunk_id="a1727:3:0",
        document_id="a1727",
        section_heading="Section 3 Amendment of section 4",
        section_id="3",
        subsection_id=None,
        paragraph_id=None,
        text="Section 3 Amendment of section 4\nThe principal Act is amended in section 4.",
        source_path="data/raw_law_pdfs/Act-A1727.pdf",
        act_title="Personal Data Protection (Amendment) Act 2024",
        source_file="Act-A1727.pdf",
        chunk_index=1,
        unit_type="section",
        unit_id="3",
        document_aliases=("Personal Data Protection (Amendment) Act 2024", "Act A1727"),
    )
    embedder = OllamaEmbedder(
        model="test-model",
        transport=FakeTransport(
            {
                generic.text: [0.95, 0.05],
                specific.text: [0.2, 0.8],
                "Which section of Act A1727 amends section 4 of the PDPA?": [0.95, 0.05],
            }
        ),
    )
    entries = _entries_for(generic, specific)
    graph = build_legal_graph([generic, specific])

    graph_results = graph_supported_search(
        entries=entries,
        embedder=embedder,
        graph=graph,
        query="Which section of Act A1727 amends section 4 of the PDPA?",
        top_k=1,
        mode="graph_supported",
    )
    reranked_results = graph_supported_search(
        entries=entries,
        embedder=embedder,
        graph=graph,
        query="Which section of Act A1727 amends section 4 of the PDPA?",
        top_k=1,
        mode="hybrid_plus_graph_with_graph_rerank",
    )

    assert graph_results[0].chunk.chunk_id == "a1727:3:0"
    assert reranked_results[0].chunk.chunk_id == "a1727:3:0"
