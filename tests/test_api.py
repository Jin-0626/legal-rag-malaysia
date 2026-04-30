from fastapi.testclient import TestClient

from legal_rag.api.app import app, get_service
from legal_rag.api.schemas import ChatResponse, HealthResponse, SourceItem
from legal_rag.api.security import hash_api_key
from legal_rag.api.service import (
    LegalRAGChatService,
    OllamaHealthStatus,
    RequestsOllamaChatTransport,
    aggregate_source_items,
    format_source_item,
    is_employment_agreement_checklist_query,
    preview_text,
    route_mode,
)
from legal_rag.chunking.models import Chunk
from legal_rag.embeddings.embedder import OllamaEmbedder
from legal_rag.retrieval.in_memory import RetrievalResult


class StubService:
    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            ollama_available=True,
            ollama_base_url="http://localhost:11434",
            vector_store_loaded=True,
            indexed_chunks=12562,
            model="llama3.1:8b",
            model_available=True,
            chat_ready=True,
            database_enabled=False,
            database_connected=False,
            database_backend="postgresql+pgvector",
            database_error=None,
            error=None,
        )

    def chat(self, *, query: str, mode: str, top_k: int, request_id: str | None = None, principal_role: str | None = None) -> ChatResponse:
        return ChatResponse(
            answer=f"Mocked answer for: {query}",
            mode_used="hybrid_filtered_rerank" if mode == "hybrid" else "hybrid_plus_graph_with_graph_rerank",
            sources=[
                SourceItem(
                    document="Employment Act 1955",
                    unit_type="section",
                    unit_id="2",
                    heading="Section 2 Interpretation",
                    score=0.9876,
                    chunk_count=1,
                    preview="Employee means a person employed under a contract of service.",
                )
            ],
            graph_path=["graph: amendment linkage -> Personal Data Protection (Amendment) Act 2024 Section 9"],
            warnings=[],
        )


def test_health_endpoint_returns_api_status() -> None:
    app.dependency_overrides[get_service] = lambda: StubService()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["ollama_available"] is True
    assert payload["model_available"] is True
    assert payload["chat_ready"] is True
    assert payload["database_enabled"] is False
    assert payload["indexed_chunks"] == 12562
    app.dependency_overrides.clear()


def test_chat_endpoint_returns_mocked_answer_and_sources() -> None:
    app.dependency_overrides[get_service] = lambda: StubService()
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"query": "Which section introduces data portability in the PDPA Amendment Act 2024?", "mode": "graph", "top_k": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode_used"] == "hybrid_plus_graph_with_graph_rerank"
    assert payload["sources"][0]["document"] == "Employment Act 1955"
    assert payload["graph_path"]
    assert "Mocked answer for:" in payload["answer"]
    assert payload["answer"]
    app.dependency_overrides.clear()


def test_auto_mode_routing_prefers_graph_for_structural_queries() -> None:
    assert route_mode("Which section of Act A1727 amends section 4 of the PDPA?", "auto") == "hybrid_plus_graph_with_graph_rerank"
    assert route_mode("Which Article begins Part III on Citizenship in the Federal Constitution?", "auto") == "hybrid_plus_graph_with_graph_rerank"
    assert route_mode("What does Section 2 of the Employment Act 1955 define?", "auto") == "hybrid_filtered_rerank"


def test_source_formatting_preserves_metadata_and_preview() -> None:
    result = RetrievalResult(
        chunk=Chunk(
            chunk_id="employment:2:0",
            document_id="employment",
            section_heading="Section 2 Interpretation",
            section_id="2",
            subsection_id=None,
            paragraph_id=None,
            text="Employee means a person employed under a contract of service. " * 8,
            source_path="data/raw_law_pdfs/employment.pdf",
            act_title="Employment Act 1955",
            unit_type="section",
            unit_id="2",
        ),
        score=0.99881,
    )

    source = format_source_item(result)

    assert source.document == "Employment Act 1955"
    assert source.unit_id == "2"
    assert source.heading == "Section 2 Interpretation"
    assert source.score == 0.9988
    assert source.chunk_count == 1
    assert source.preview.endswith("...")
    assert len(preview_text(result.chunk.text, limit=80)) <= 83


def test_aggregate_source_items_groups_duplicate_chunks_into_one_section_level_source() -> None:
    first = RetrievalResult(
        chunk=Chunk(
            chunk_id="employment:12:0",
            document_id="employment",
            section_heading="Section 12 Notice of termination of contract",
            section_id="12",
            subsection_id=None,
            paragraph_id=None,
            text="Either party may terminate the contract by giving notice.",
            source_path="data/raw_law_pdfs/employment.pdf",
            act_title="Employment Act 1955",
            unit_type="section",
            unit_id="12",
        ),
        score=1.25,
    )
    second = RetrievalResult(
        chunk=Chunk(
            chunk_id="employment:12:1",
            document_id="employment",
            section_heading="Section 12 Notice of termination of contract",
            section_id="12",
            subsection_id="1",
            paragraph_id=None,
            text="Notice periods may vary depending on the length of service.",
            source_path="data/raw_law_pdfs/employment.pdf",
            act_title="Employment Act 1955",
            unit_type="section",
            unit_id="12",
        ),
        score=1.19,
    )
    third = RetrievalResult(
        chunk=Chunk(
            chunk_id="employment:10:0",
            document_id="employment",
            section_heading="Section 10 Contracts to be in writing and to include provision for termination",
            section_id="10",
            subsection_id=None,
            paragraph_id=None,
            text="Every contract of service exceeding one month shall be in writing.",
            source_path="data/raw_law_pdfs/employment.pdf",
            act_title="Employment Act 1955",
            unit_type="section",
            unit_id="10",
        ),
        score=1.3,
    )

    sources = aggregate_source_items([third, first, second], top_k=5)

    assert [source.unit_id for source in sources] == ["10", "12"]
    assert sources[1].chunk_count == 2
    assert sources[1].score == 1.25
    assert "Either party may terminate the contract by giving notice." in sources[1].preview
    assert "Notice periods may vary depending on the length of service." in sources[1].preview


class FailingOllamaTransport:
    def health(self) -> bool:
        return False

    def health_details(self, *, model: str, api_key: str | None = None, probe_chat: bool = True) -> OllamaHealthStatus:
        return OllamaHealthStatus(
            ollama_available=False,
            model_available=False,
            chat_ready=False,
            error="connection refused",
        )

    def chat(self, *, model: str, messages: list[dict[str, str]], api_key: str | None = None) -> str:
        raise RuntimeError("chat unavailable")

    def stream_chat(self, *, model: str, messages: list[dict[str, str]], api_key: str | None = None):
        raise RuntimeError("chat unavailable")


class DummyResponse:
    def __init__(
        self,
        payload: dict[str, object] | None,
        *,
        status_code: int = 200,
        text: str = "",
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            error = requests.exceptions.HTTPError(f"{self.status_code} error")
            error.response = self
            raise error
        return None

    def json(self) -> dict[str, object]:
        if self._payload is None:
            raise ValueError("invalid json")
        return self._payload

    def iter_content(self, chunk_size=None, decode_unicode=False):
        return iter(())


class DummyStreamingResponse(DummyResponse):
    def __init__(self, chunks: list[bytes], *, status_code: int = 200, text: str = "") -> None:
        super().__init__({}, status_code=status_code, text=text)
        self._chunks = chunks

    def iter_content(self, chunk_size=None, decode_unicode=False):
        return iter(self._chunks)


def test_fallback_answer_is_structured_and_not_raw_dump() -> None:
    service = LegalRAGChatService(
        embedder=OllamaEmbedder(transport=None),
        ollama_transport=FailingOllamaTransport(),
    )
    long_text = (
        "Section 60K Employment of foreign employee. "
        "The Director General may approve the employment of a foreign employee subject to conditions. "
        "The employer shall furnish information, maintain records, and comply with directions. "
    ) * 8
    result = RetrievalResult(
        chunk=Chunk(
            chunk_id="employment:60k:0",
            document_id="employment",
            section_heading="Section 60K Employment of foreign employee",
            section_id="60K",
            subsection_id=None,
            paragraph_id=None,
            text=long_text,
            source_path="data/raw_law_pdfs/employment.pdf",
            act_title="Employment Act 1955",
            unit_type="section",
            unit_id="60K",
        ),
        score=0.91,
    )

    answer = service.generator.answer("How do I protect myself in an employment agreement?", [result]).answer

    assert "Direct Answer:" in answer
    assert "Legal Basis:" in answer
    assert "Practical Meaning:" in answer
    assert "Important Limits:" in answer
    assert "Sources:" in answer
    assert answer.count("Section 60K Employment of foreign employee") <= 1
    assert len(answer) < len(long_text)


def test_requests_ollama_transport_omits_authorization_header_when_api_key_missing(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, *, json: dict[str, object], headers: dict[str, str], timeout: float) -> DummyResponse:
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return DummyResponse({"message": {"content": "Structured answer"}})

    monkeypatch.setattr("legal_rag.api.service.requests.post", fake_post)
    transport = RequestsOllamaChatTransport("http://localhost:11434")

    content = transport.chat(
        model="gpt-oss:120b-cloud",
        messages=[{"role": "user", "content": "Hello"}],
        api_key=None,
    )

    assert content == "Structured answer"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["headers"] == {"Content-Type": "application/json"}


def test_requests_ollama_transport_includes_bearer_authorization_when_api_key_present(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, *, json: dict[str, object], headers: dict[str, str], timeout: float) -> DummyResponse:
        captured["headers"] = headers
        return DummyResponse({"message": {"content": "Structured answer"}})

    monkeypatch.setattr("legal_rag.api.service.requests.post", fake_post)
    transport = RequestsOllamaChatTransport("https://ollama.example.com")

    transport.chat(
        model="gpt-oss:120b-cloud",
        messages=[{"role": "user", "content": "Hello"}],
        api_key="secret-token",
    )

    assert captured["headers"] == {
        "Content-Type": "application/json",
        "Authorization": "Bearer secret-token",
    }


def test_requests_ollama_transport_health_details_reports_model_available_and_chat_ready(monkeypatch) -> None:
    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> DummyResponse:
        return DummyResponse({"models": [{"name": "llama3.1:8b"}]})

    def fake_post(url: str, *, json: dict[str, object], headers: dict[str, str], timeout: float) -> DummyResponse:
        return DummyResponse({"message": {"content": "OK"}})

    monkeypatch.setattr("legal_rag.api.service.requests.get", fake_get)
    monkeypatch.setattr("legal_rag.api.service.requests.post", fake_post)
    transport = RequestsOllamaChatTransport("http://localhost:11434")

    status = transport.health_details(model="llama3.1:8b")

    assert status.ollama_available is True
    assert status.model_available is True
    assert status.chat_ready is True
    assert status.error is None


def test_requests_ollama_transport_health_details_reports_missing_model(monkeypatch) -> None:
    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> DummyResponse:
        return DummyResponse({"models": [{"name": "gemma3"}]})

    monkeypatch.setattr("legal_rag.api.service.requests.get", fake_get)
    transport = RequestsOllamaChatTransport("http://localhost:11434")

    status = transport.health_details(model="llama3.1:8b")

    assert status.ollama_available is True
    assert status.model_available is False
    assert status.chat_ready is False
    assert status.error == "Configured Ollama model is not available. Run: ollama pull llama3.1:8b"


def test_requests_ollama_transport_reports_unauthorized_chat_failure(monkeypatch) -> None:
    def fake_post(url: str, *, json: dict[str, object], headers: dict[str, str], timeout: float) -> DummyResponse:
        return DummyResponse({"error": "unauthorized"}, status_code=401, text="unauthorized")

    monkeypatch.setattr("legal_rag.api.service.requests.post", fake_post)
    transport = RequestsOllamaChatTransport("https://ollama.example.com")

    try:
        transport.chat(
            model="gpt-oss:120b-cloud",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="secret-token",
        )
    except Exception as exc:
        assert str(exc) == "unauthorized"
    else:
        raise AssertionError("Expected unauthorized Ollama chat failure.")


def test_requests_ollama_transport_reports_malformed_response(monkeypatch) -> None:
    def fake_post(url: str, *, json: dict[str, object], headers: dict[str, str], timeout: float) -> DummyResponse:
        return DummyResponse(None)

    monkeypatch.setattr("legal_rag.api.service.requests.post", fake_post)
    transport = RequestsOllamaChatTransport("http://localhost:11434")

    try:
        transport.chat(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": "Hello"}],
        )
    except Exception as exc:
        assert str(exc) == "malformed response"
    else:
        raise AssertionError("Expected malformed response Ollama chat failure.")


def test_requests_ollama_transport_stream_chat_parses_partial_ndjson_chunks(monkeypatch) -> None:
    def fake_post(url: str, *, json: dict[str, object], headers: dict[str, str], timeout: float, stream: bool) -> DummyStreamingResponse:
        assert stream is True
        return DummyStreamingResponse(
            [
                b'{"message":{"content":"Hel',
                b'lo"}}\n{"message":{"content":" world"}}\n{"done":true}\n',
            ]
        )

    monkeypatch.setattr("legal_rag.api.service.requests.post", fake_post)
    transport = RequestsOllamaChatTransport("http://localhost:11434")

    chunks = list(
        transport.stream_chat(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": "Hello"}],
        )
    )

    assert chunks == ["Hello", " world"]


def test_chat_endpoint_preserves_structured_answer_sections() -> None:
    class StructuredStubService(StubService):
        def chat(self, *, query: str, mode: str, top_k: int, request_id: str | None = None, principal_role: str | None = None) -> ChatResponse:
            return ChatResponse(
                answer=(
                    "Direct Answer:\nShort answer.\n\n"
                    "Legal Basis:\nEmployment Act 1955, Section 2 says something relevant.\n\n"
                    "Practical Meaning:\nPlain English explanation.\n\n"
                    "Important Limits:\nNot legal advice.\n\n"
                    "Sources:\n- Employment Act 1955, Section 2"
                ),
                mode_used="hybrid_filtered_rerank",
                sources=[
                    SourceItem(
                        document="Employment Act 1955",
                        unit_type="section",
                        unit_id="2",
                        heading="Section 2 Interpretation",
                        score=0.9876,
                        chunk_count=1,
                        preview="Employee means a person employed under a contract of service.",
                    )
                ],
                graph_path=[],
                warnings=[],
            )

    app.dependency_overrides[get_service] = lambda: StructuredStubService()
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"query": "What does Section 2 of the Employment Act 1955 define?", "mode": "hybrid", "top_k": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Direct Answer:" in payload["answer"]
    assert "Legal Basis:" in payload["answer"]
    assert "Practical Meaning:" in payload["answer"]
    assert "Important Limits:" in payload["answer"]
    assert "Sources:" in payload["answer"]
    assert payload["sources"][0]["heading"] == "Section 2 Interpretation"
    app.dependency_overrides.clear()


def test_employment_agreement_query_detection_matches_broad_checklist_queries() -> None:
    assert is_employment_agreement_checklist_query("What do I need to check before signing an employment agreement?") is True
    assert is_employment_agreement_checklist_query("How do I protect myself in an employment contract?") is True
    assert is_employment_agreement_checklist_query("What does Section 2 of the Employment Act 1955 define?") is False


def test_chat_fallback_produces_grounded_employment_agreement_checklist() -> None:
    class ChecklistService(LegalRAGChatService):
        def _retrieve(self, *, query: str, mode_used: str, top_k: int) -> list[RetrievalResult]:
            return [
                RetrievalResult(
                    chunk=Chunk(
                        chunk_id="employment:10:0",
                        document_id="employment",
                        section_heading="Section 10 Contracts to be in writing and to include provision for termination",
                        section_id="10",
                        subsection_id=None,
                        paragraph_id=None,
                        text="Every contract of service exceeding one month shall be in writing and include a provision for termination.",
                        source_path="data/raw_law_pdfs/employment.pdf",
                        act_title="Employment Act 1955",
                        unit_type="section",
                        unit_id="10",
                    ),
                    score=1.3,
                ),
                RetrievalResult(
                    chunk=Chunk(
                        chunk_id="employment:12:0",
                        document_id="employment",
                        section_heading="Section 12 Notice of termination of contract",
                        section_id="12",
                        subsection_id=None,
                        paragraph_id=None,
                        text="Either party may terminate the contract of service by giving notice.",
                        source_path="data/raw_law_pdfs/employment.pdf",
                        act_title="Employment Act 1955",
                        unit_type="section",
                        unit_id="12",
                    ),
                    score=1.2,
                ),
                RetrievalResult(
                    chunk=Chunk(
                        chunk_id="employment:100:0",
                        document_id="employment",
                        section_heading="Section 100 overtime, holidays, annual leave, and sick leave",
                        section_id="100",
                        subsection_id=None,
                        paragraph_id=None,
                        text="This section addresses overtime, holidays, annual leave, and sick leave obligations.",
                        source_path="data/raw_law_pdfs/employment.pdf",
                        act_title="Employment Act 1955",
                        unit_type="section",
                        unit_id="100",
                    ),
                    score=1.1,
                ),
            ]

    service = ChecklistService(
        embedder=OllamaEmbedder(transport=None),
        ollama_transport=FailingOllamaTransport(),
    )

    response = service.chat(
        query="What do I need to check before signing an employment agreement?",
        mode="hybrid",
        top_k=5,
    )

    assert "Checklist:" in response.answer
    assert "Employment Act 1955, Section 10" in response.answer
    assert "Employment Act 1955, Section 12" in response.answer
    assert "Employment Act 1955, Section 100" in response.answer
    assert "Important Limits:" in response.answer
    assert "Section 10 Contracts to be in writing and to include provision for termination 10." not in response.answer
    assert [source.unit_id for source in response.sources] == ["10", "12", "100"]
    assert "Ollama chat failed: chat unavailable; returned deterministic grounded fallback." in response.warnings


def test_chat_fallback_warning_includes_timeout_reason(monkeypatch) -> None:
    import requests

    class TimeoutTransport:
        def health(self) -> bool:
            return True

        def health_details(self, *, model: str, api_key: str | None = None, probe_chat: bool = True) -> OllamaHealthStatus:
            return OllamaHealthStatus(
                ollama_available=True,
                model_available=True,
                chat_ready=False,
                error="timeout",
            )

        def chat(self, *, model: str, messages: list[dict[str, str]], api_key: str | None = None) -> str:
            raise requests.exceptions.Timeout("timed out")

    class TimeoutService(LegalRAGChatService):
        def _retrieve(self, *, query: str, mode_used: str, top_k: int) -> list[RetrievalResult]:
            return [
                RetrievalResult(
                    chunk=Chunk(
                        chunk_id="employment:10:0",
                        document_id="employment",
                        section_heading="Section 10 Contracts to be in writing and to include provision for termination",
                        section_id="10",
                        subsection_id=None,
                        paragraph_id=None,
                        text="Every contract of service exceeding one month shall be in writing.",
                        source_path="data/raw_law_pdfs/employment.pdf",
                        act_title="Employment Act 1955",
                        unit_type="section",
                        unit_id="10",
                    ),
                    score=1.3,
                )
            ]

    service = TimeoutService(
        embedder=OllamaEmbedder(transport=None),
        ollama_transport=TimeoutTransport(),
    )

    response = service.chat(
        query="What do I need to check before signing an employment agreement?",
        mode="hybrid",
        top_k=3,
    )

    assert any("timeout" in warning for warning in response.warnings)


def test_chat_stream_endpoint_streams_meta_tokens_and_done() -> None:
    class StreamingStubService(StubService):
        def chat_stream(self, *, query: str, mode: str, top_k: int, request_id: str | None = None, principal_role: str | None = None):
            yield '{"type":"meta","mode_used":"hybrid_filtered_rerank","sources":[],"graph_path":[],"warnings":[]}\n'
            yield '{"type":"token","content":"Direct Answer:"}\n'
            yield '{"type":"token","content":"\\nShort answer."}\n'
            yield '{"type":"done","mode_used":"hybrid_filtered_rerank","sources":[],"graph_path":[],"warnings":[]}\n'

    app.dependency_overrides[get_service] = lambda: StreamingStubService()
    client = TestClient(app)

    with client.stream(
        "POST",
        "/chat_stream",
        json={"query": "What does Section 2 define?", "mode": "hybrid", "top_k": 5},
    ) as response:
        body = "".join(chunk for chunk in response.iter_text())

    assert response.status_code == 200
    assert '"type":"meta"' in body
    assert '"type":"token"' in body
    assert 'Short answer.' in body
    assert '"type":"done"' in body
    app.dependency_overrides.clear()


def test_chat_abstains_for_impossible_section_lookup() -> None:
    class ImpossibleSectionService(LegalRAGChatService):
        def _retrieve(self, *, query: str, mode_used: str, top_k: int) -> list[RetrievalResult]:
            return [
                RetrievalResult(
                    chunk=Chunk(
                        chunk_id="employment:10:0",
                        document_id="employment",
                        section_heading="Section 10 Contracts to be in writing and to include provision for termination",
                        section_id="10",
                        subsection_id=None,
                        paragraph_id=None,
                        text="Every contract of service exceeding one month shall be in writing.",
                        source_path="data/raw_law_pdfs/employment.pdf",
                        act_title="Employment Act 1955",
                        unit_type="section",
                        unit_id="10",
                    ),
                    score=1.1,
                )
            ]

    service = ImpossibleSectionService(
        embedder=OllamaEmbedder(transport=None),
        ollama_transport=FailingOllamaTransport(),
    )

    response = service.chat(
        query="What does Section 999 of the Employment Act 1955 say?",
        mode="hybrid",
        top_k=5,
    )

    assert "not enough" in response.answer.lower()
    assert response.sources == []


def test_chat_requires_api_key_when_enabled(monkeypatch, tmp_path) -> None:
    keys_file = tmp_path / "api_keys.json"
    keys_file.write_text(
        '{"keys": [{"name": "viewer", "key_hash": "' + hash_api_key("viewer-secret") + '", "role": "viewer"}]}',
        encoding="utf-8",
    )
    monkeypatch.setenv("LEGAL_RAG_REQUIRE_API_KEY", "true")
    monkeypatch.setenv("LEGAL_RAG_API_KEYS_FILE", str(keys_file))
    app.dependency_overrides[get_service] = lambda: StubService()
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"query": "What does Section 2 define?", "mode": "hybrid", "top_k": 5},
    )

    assert response.status_code == 401
    app.dependency_overrides.clear()


def test_chat_accepts_valid_api_key_when_enabled(monkeypatch, tmp_path) -> None:
    keys_file = tmp_path / "api_keys.json"
    keys_file.write_text(
        '{"keys": [{"name": "viewer", "key_hash": "' + hash_api_key("viewer-secret") + '", "role": "viewer"}]}',
        encoding="utf-8",
    )
    monkeypatch.setenv("LEGAL_RAG_REQUIRE_API_KEY", "true")
    monkeypatch.setenv("LEGAL_RAG_API_KEYS_FILE", str(keys_file))
    app.dependency_overrides[get_service] = lambda: StubService()
    client = TestClient(app)

    response = client.post(
        "/chat",
        headers={"X-API-Key": "viewer-secret"},
        json={"query": "What does Section 2 define?", "mode": "hybrid", "top_k": 5},
    )

    assert response.status_code == 200
    app.dependency_overrides.clear()


def test_admin_endpoint_denies_viewer_role(monkeypatch, tmp_path) -> None:
    keys_file = tmp_path / "api_keys.json"
    keys_file.write_text(
        '{"keys": [{"name": "viewer", "key_hash": "' + hash_api_key("viewer-secret") + '", "role": "viewer"}]}',
        encoding="utf-8",
    )
    monkeypatch.setenv("LEGAL_RAG_REQUIRE_API_KEY", "true")
    monkeypatch.setenv("LEGAL_RAG_API_KEYS_FILE", str(keys_file))
    client = TestClient(app)

    response = client.get("/admin/security", headers={"X-API-Key": "viewer-secret"})

    assert response.status_code == 403


def test_admin_endpoint_allows_admin_role(monkeypatch, tmp_path) -> None:
    keys_file = tmp_path / "api_keys.json"
    keys_file.write_text(
        '{"keys": [{"name": "admin", "key_hash": "' + hash_api_key("admin-secret") + '", "role": "admin"}]}',
        encoding="utf-8",
    )
    monkeypatch.setenv("LEGAL_RAG_REQUIRE_API_KEY", "true")
    monkeypatch.setenv("LEGAL_RAG_API_KEYS_FILE", str(keys_file))
    client = TestClient(app)

    response = client.get("/admin/security", headers={"X-API-Key": "admin-secret"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["api_key_required"] is True
    assert payload["configured_keys"] == 1
