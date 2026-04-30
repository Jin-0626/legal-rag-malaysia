from ragflow_tools.legal_rag_tool import run, tool_definition


class DummyResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


def test_tool_definition_exposes_expected_registration_shape() -> None:
    definition = tool_definition()

    assert definition["name"] == "legal_rag_query"
    assert definition["inputs"][0]["name"] == "query"
    assert definition["inputs"][1]["default"] == "auto"


def test_run_calls_existing_chat_endpoint_and_returns_structured_payload(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, *, json: dict[str, object], headers: dict[str, str] | None, timeout: int) -> DummyResponse:
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return DummyResponse(
            {
                "answer": "Structured answer",
                "sources": [{"document": "Employment Act 1955", "unit_id": "2"}],
                "mode_used": "hybrid_filtered_rerank",
                "graph_path": [],
                "warnings": [],
            }
        )

    monkeypatch.setattr("ragflow_tools.legal_rag_tool.requests.post", fake_post)

    result = run("What does Section 2 of Employment Act 1955 define?", mode="auto")

    assert captured["url"] == "http://127.0.0.1:8000/chat"
    assert captured["json"] == {
        "query": "What does Section 2 of Employment Act 1955 define?",
        "mode": "auto",
        "top_k": 5,
    }
    assert captured["headers"] is None
    assert result["answer"] == "Structured answer"
    assert result["mode_used"] == "hybrid_filtered_rerank"
    assert result["sources"][0]["document"] == "Employment Act 1955"


def test_run_includes_x_api_key_header_when_configured(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, *, json: dict[str, object], headers: dict[str, str] | None, timeout: int) -> DummyResponse:
        captured["headers"] = headers
        return DummyResponse(
            {
                "answer": "Structured answer",
                "sources": [],
                "mode_used": "hybrid_filtered_rerank",
                "graph_path": [],
                "warnings": [],
            }
        )

    monkeypatch.setattr("ragflow_tools.legal_rag_tool.DEFAULT_API_KEY", "ragflow-secret")
    monkeypatch.setattr("ragflow_tools.legal_rag_tool.requests.post", fake_post)

    run("What does Section 2 of Employment Act 1955 define?", mode="auto")

    assert captured["headers"] == {"X-API-Key": "ragflow-secret"}
