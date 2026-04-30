"""Thin backend service layer for the Legal RAG demo API."""

from __future__ import annotations

import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterator, Literal, Protocol

import requests
from requests import Response

from legal_rag.api.schemas import ChatResponse, HealthResponse, SourceItem
from legal_rag.api.logging_utils import build_query_log_fields, log_event
from legal_rag.config.settings import LegalRAGSettings, build_settings
from legal_rag.embeddings.embedder import EmbeddedChunk, OllamaEmbedder
from legal_rag.generation.grounded import GroundedAnswerGenerator
from legal_rag.graph import LegalGraph, build_legal_graph, search_graph
from legal_rag.retrieval.in_memory import RetrievalResult, search_embedded_entries
from legal_rag.retrieval.vector_store import JsonlVectorStore, chunk_from_record
from legal_rag.storage import check_database_health
from legal_rag.workflows import graph_supported_search

ChatMode = Literal["auto", "hybrid", "graph"]
ResolvedMode = Literal["hybrid_filtered_rerank", "hybrid_plus_graph_with_graph_rerank"]
GRAPH_ROUTE_PATTERN = re.compile(
    r"\b(amend|amends|amendment|pindaan|part|chapter|division|bahagian|bab|refers to|reference|rujuk)\b",
    re.IGNORECASE,
)
EMPLOYMENT_AGREEMENT_PATTERN = re.compile(
    r"(employment agreement|employment contract|contract of service).*(before signing|signing|protect myself|protect yourself|need to check|should i check)|"
    r"(before signing|signing|protect myself|protect yourself|need to check|should i check).*(employment agreement|employment contract|contract of service)",
    re.IGNORECASE,
)
EXPLICIT_UNIT_PATTERN = re.compile(r"\b(section|article|perkara)\s+([0-9A-Za-z]+)\b", re.IGNORECASE)
LEGAL_ADVICE_PATTERN = re.compile(r"\b(should i|can i sue|legal advice|what should i do legally)\b", re.IGNORECASE)


class OllamaChatTransport(Protocol):
    """Protocol to support deterministic API tests without a live Ollama server."""

    def health(self) -> bool:
        """Return whether Ollama is reachable."""

    def health_details(self, *, model: str, api_key: str | None = None, probe_chat: bool = True) -> "OllamaHealthStatus":
        """Return detailed Ollama connectivity and readiness information."""

    def chat(self, *, model: str, messages: list[dict[str, str]], api_key: str | None = None) -> str:
        """Return the assistant content from Ollama chat."""

    def stream_chat(self, *, model: str, messages: list[dict[str, str]], api_key: str | None = None) -> Iterator[str]:
        """Yield assistant content fragments from Ollama chat."""


@dataclass(frozen=True)
class OllamaHealthStatus:
    ollama_available: bool
    model_available: bool
    chat_ready: bool
    error: str | None = None


class OllamaChatError(RuntimeError):
    """Raised when Ollama chat fails with a user-actionable reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class RequestsOllamaChatTransport:
    """HTTP transport for local Ollama chat generation."""

    def __init__(self, base_url: str, timeout_seconds: float = 45.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def health(self) -> bool:
        return self.health_details(model="", probe_chat=False).ollama_available

    def health_details(self, *, model: str, api_key: str | None = None, probe_chat: bool = True) -> OllamaHealthStatus:
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                headers=self._headers(api_key),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            return OllamaHealthStatus(
                ollama_available=False,
                model_available=False,
                chat_ready=False,
                error=self._describe_request_error(exc),
            )

        available_models = self._extract_model_names(payload)
        model_available = bool(model and model in available_models)
        if model and not model_available:
            return OllamaHealthStatus(
                ollama_available=True,
                model_available=False,
                chat_ready=False,
                error=f"Configured Ollama model is not available. Run: ollama pull {model}",
            )

        if not probe_chat or not model:
            return OllamaHealthStatus(
                ollama_available=True,
                model_available=model_available,
                chat_ready=model_available,
                error=None,
            )

        try:
            self.chat(
                model=model,
                messages=[{"role": "user", "content": "Reply with OK."}],
                api_key=api_key,
            )
        except OllamaChatError as exc:
            return OllamaHealthStatus(
                ollama_available=True,
                model_available=True,
                chat_ready=False,
                error=exc.reason,
            )
        except Exception as exc:
            return OllamaHealthStatus(
                ollama_available=True,
                model_available=True,
                chat_ready=False,
                error=self._describe_request_error(exc),
            )

        return OllamaHealthStatus(
            ollama_available=True,
            model_available=True,
            chat_ready=True,
            error=None,
        )

    def chat(self, *, model: str, messages: list[dict[str, str]], api_key: str | None = None) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={"model": model, "messages": messages, "stream": False},
                headers=self._headers(api_key),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except Exception as exc:
            raise OllamaChatError(self._describe_request_error(exc)) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise OllamaChatError("malformed response") from exc

        content = self._extract_chat_content(payload)
        if not content:
            raise OllamaChatError("malformed response")
        return content

    def stream_chat(self, *, model: str, messages: list[dict[str, str]], api_key: str | None = None) -> Iterator[str]:
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={"model": model, "messages": messages, "stream": True},
                headers=self._headers(api_key),
                timeout=self.timeout_seconds,
                stream=True,
            )
            response.raise_for_status()
        except Exception as exc:
            raise OllamaChatError(self._describe_request_error(exc)) from exc

        yielded_any = False
        try:
            for fragment in self._iter_ndjson_message_content(response):
                yielded_any = True
                yield fragment
        except OllamaChatError:
            raise
        except Exception as exc:
            raise OllamaChatError(self._describe_request_error(exc)) from exc

        if not yielded_any:
            raise OllamaChatError("malformed response")

    def _headers(self, api_key: str | None = None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _extract_model_names(self, payload: object) -> set[str]:
        if not isinstance(payload, dict):
            return set()
        models = payload.get("models")
        if not isinstance(models, list):
            return set()
        names: set[str] = set()
        for model_entry in models:
            if isinstance(model_entry, dict):
                name = model_entry.get("name")
                if isinstance(name, str) and name.strip():
                    names.add(name.strip())
        return names

    def _extract_chat_content(self, payload: object) -> str | None:
        if not isinstance(payload, dict):
            return None
        message = payload.get("message")
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            return None
        return content.strip()

    def _describe_request_error(self, exc: Exception) -> str:
        if isinstance(exc, requests.exceptions.Timeout):
            return "timeout"
        if isinstance(exc, requests.exceptions.ConnectionError):
            return "connection refused"
        if isinstance(exc, requests.exceptions.HTTPError):
            response = exc.response
            if response is not None:
                return self._describe_http_error(response)
            return "http error"
        if isinstance(exc, ValueError):
            return "malformed response"
        return str(exc).strip() or "unknown error"

    def _describe_http_error(self, response: Response) -> str:
        status_code = response.status_code
        response_text = (response.text or "").strip()
        lowered = response_text.lower()
        if status_code in (401, 403):
            return "unauthorized"
        if status_code == 404:
            return "wrong base URL or /api/chat endpoint not found"
        if "model" in lowered and "not found" in lowered:
            return "model not found"
        if "unauthorized" in lowered:
            return "unauthorized"
        return f"http {status_code}"

    def _iter_ndjson_message_content(self, response: Response) -> Iterator[str]:
        buffer = ""
        for raw_chunk in response.iter_content(chunk_size=None, decode_unicode=False):
            if not raw_chunk:
                continue
            buffer += raw_chunk.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                fragment = self._parse_ndjson_line(line)
                if fragment is not None:
                    yield fragment
        if buffer.strip():
            fragment = self._parse_ndjson_line(buffer)
            if fragment is not None:
                yield fragment

    def _parse_ndjson_line(self, line: str) -> str | None:
        payload_line = line.strip()
        if not payload_line:
            return None
        try:
            payload = json.loads(payload_line)
        except json.JSONDecodeError as exc:
            raise OllamaChatError("malformed response") from exc
        if not isinstance(payload, dict):
            raise OllamaChatError("malformed response")
        if payload.get("done") is True:
            return None
        message = payload.get("message")
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        if not isinstance(content, str):
            return None
        return content


@dataclass(frozen=True)
class LoadedIndex:
    entries: list[EmbeddedChunk]
    graph: LegalGraph


class LegalRAGChatService:
    """Demo-ready orchestration over the existing retrieval and graph workflows."""

    def __init__(
        self,
        settings: LegalRAGSettings | None = None,
        *,
        embedder: OllamaEmbedder | None = None,
        ollama_transport: OllamaChatTransport | None = None,
    ) -> None:
        self.settings = settings or build_settings()
        self.embedder = embedder or OllamaEmbedder()
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")
        self.ollama_api_key = os.getenv("OLLAMA_API_KEY") or None
        self.ollama_timeout_seconds = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))
        self.ollama_transport = ollama_transport or RequestsOllamaChatTransport(
            self.ollama_base_url,
            timeout_seconds=self.ollama_timeout_seconds,
        )
        self.generator = GroundedAnswerGenerator()

    @cached_property
    def loaded_index(self) -> LoadedIndex:
        vector_store = JsonlVectorStore(self._vector_store_path())
        entries: list[EmbeddedChunk] = []
        for record in vector_store.load_records():
            entries.append(
                EmbeddedChunk(
                    chunk=chunk_from_record(
                        {
                            "chunk_id": record.chunk_id,
                            "document_id": record.document_id,
                            "act_title": record.act_title,
                            "act_number": record.act_number,
                            "section_heading": record.section_heading,
                            "section_id": record.section_id,
                            "unit_type": record.unit_type,
                            "unit_id": record.unit_id,
                            "subsection_id": record.subsection_id,
                            "paragraph_id": record.paragraph_id,
                            "source_file": record.source_file,
                            "source_path": record.source_path,
                            "chunk_index": record.chunk_index,
                            "document_aliases": record.document_aliases,
                            "text": record.text,
                        }
                    ),
                    embedding=record.embedding,
                )
            )
        return LoadedIndex(entries=entries, graph=build_legal_graph([entry.chunk for entry in entries]))

    def health(self) -> HealthResponse:
        database_status = check_database_health()
        ollama_status = OllamaHealthStatus(
            ollama_available=False,
            model_available=False,
            chat_ready=False,
            error="Ollama not checked.",
        )
        try:
            ollama_status = self.ollama_transport.health_details(
                model=self.ollama_model,
                api_key=self.ollama_api_key,
                probe_chat=True,
            )
        except Exception as exc:
            ollama_status = OllamaHealthStatus(
                ollama_available=False,
                model_available=False,
                chat_ready=False,
                error=describe_ollama_failure(exc),
            )
        entries = self.loaded_index.entries
        return HealthResponse(
            status=(
                "ok"
                if entries
                and ollama_status.chat_ready
                and (not database_status.enabled or database_status.connected)
                else "degraded"
            ),
            ollama_available=ollama_status.ollama_available,
            ollama_base_url=self.ollama_base_url,
            vector_store_loaded=bool(entries),
            indexed_chunks=len(entries),
            model=self.ollama_model,
            model_available=ollama_status.model_available,
            chat_ready=ollama_status.chat_ready,
            database_enabled=database_status.enabled,
            database_connected=database_status.connected,
            database_backend=database_status.backend,
            database_error=database_status.error,
            error=ollama_status.error,
        )

    def chat(self, *, query: str, mode: ChatMode, top_k: int, request_id: str | None = None, principal_role: str | None = None) -> ChatResponse:
        mode_used = route_mode(query, mode)
        warnings: list[str] = []
        results = self._retrieve(query=query, mode_used=mode_used, top_k=top_k)
        fallback_reason: str | None = None
        if should_abstain_for_explicit_unit_lookup(query, results):
            results = []
        sources = aggregate_source_items(results, top_k=top_k)
        graph_path = self._graph_path(query=query, results=results, mode_used=mode_used, top_k=top_k)

        if not results:
            if _query_demands_abstention(query):
                warnings.append("Not enough information in the retrieved legal sources.")
            else:
                warnings.append("No supporting legal sources were retrieved.")
            fallback = self.generator.answer(query, [])
            self._log_chat_event(
                request_id=request_id,
                query=query,
                mode_used=mode_used,
                principal_role=principal_role,
                result_count=0,
                fallback_reason="no_supporting_sources",
                stream=False,
            )
            return ChatResponse(
                answer=fallback.answer,
                mode_used=mode_used,
                sources=[],
                graph_path=graph_path,
                warnings=warnings,
            )

        try:
            answer = self._generate_grounded_answer(query=query, results=results[:top_k])
        except OllamaChatError as exc:
            fallback_reason = exc.reason
            warnings.append(f"Ollama chat failed: {exc.reason}; returned deterministic grounded fallback.")
            answer = self.generator.answer(query, results[:top_k]).answer
        except Exception as exc:
            reason = describe_ollama_failure(exc)
            fallback_reason = reason
            warnings.append(f"Ollama chat failed: {reason}; returned deterministic grounded fallback.")
            answer = self.generator.answer(query, results[:top_k]).answer

        self._log_chat_event(
            request_id=request_id,
            query=query,
            mode_used=mode_used,
            principal_role=principal_role,
            result_count=len(results),
            fallback_reason=fallback_reason,
            stream=False,
        )

        return ChatResponse(
            answer=answer,
            mode_used=mode_used,
            sources=sources,
            graph_path=graph_path,
            warnings=warnings,
        )

    def chat_stream(self, *, query: str, mode: ChatMode, top_k: int, request_id: str | None = None, principal_role: str | None = None) -> Iterator[str]:
        mode_used = route_mode(query, mode)
        warnings: list[str] = []
        results = self._retrieve(query=query, mode_used=mode_used, top_k=top_k)
        fallback_reason: str | None = None
        if should_abstain_for_explicit_unit_lookup(query, results):
            results = []
        sources = aggregate_source_items(results, top_k=top_k)
        graph_path = self._graph_path(query=query, results=results, mode_used=mode_used, top_k=top_k)

        yield self._stream_event(
            "meta",
            {
                "mode_used": mode_used,
                "sources": [source.model_dump() for source in sources],
                "graph_path": graph_path,
                "warnings": warnings,
            },
        )

        if not results:
            if _query_demands_abstention(query):
                warnings.append("Not enough information in the retrieved legal sources.")
            else:
                warnings.append("No supporting legal sources were retrieved.")
            fallback = self.generator.answer(query, [])
            for chunk in chunk_text_for_streaming(fallback.answer):
                yield self._stream_event("token", {"content": chunk})
            self._log_chat_event(
                request_id=request_id,
                query=query,
                mode_used=mode_used,
                principal_role=principal_role,
                result_count=0,
                fallback_reason="no_supporting_sources",
                stream=True,
            )
            yield self._stream_event(
                "done",
                {
                    "mode_used": mode_used,
                    "sources": [source.model_dump() for source in sources],
                    "graph_path": graph_path,
                    "warnings": warnings,
                },
            )
            return

        messages = self._build_generation_messages(query=query, results=results[:top_k])
        try:
            for chunk in self.ollama_transport.stream_chat(
                model=self.ollama_model,
                messages=messages,
                api_key=self.ollama_api_key,
            ):
                if chunk:
                    yield self._stream_event("token", {"content": chunk})
        except OllamaChatError as exc:
            fallback_reason = exc.reason
            warnings.append(f"Ollama chat failed: {exc.reason}; returned deterministic grounded fallback.")
            fallback = self.generator.answer(query, results[:top_k]).answer
            for chunk in chunk_text_for_streaming(fallback):
                yield self._stream_event("token", {"content": chunk})
        except Exception as exc:
            fallback_reason = describe_ollama_failure(exc)
            warnings.append(f"Ollama chat failed: {fallback_reason}; returned deterministic grounded fallback.")
            fallback = self.generator.answer(query, results[:top_k]).answer
            for chunk in chunk_text_for_streaming(fallback):
                yield self._stream_event("token", {"content": chunk})

        self._log_chat_event(
            request_id=request_id,
            query=query,
            mode_used=mode_used,
            principal_role=principal_role,
            result_count=len(results),
            fallback_reason=fallback_reason,
            stream=True,
        )
        yield self._stream_event(
            "done",
            {
                "mode_used": mode_used,
                "sources": [source.model_dump() for source in sources],
                "graph_path": graph_path,
                "warnings": warnings,
            },
        )

    def _retrieve(self, *, query: str, mode_used: ResolvedMode, top_k: int) -> list[RetrievalResult]:
        entries = self.loaded_index.entries
        if mode_used == "hybrid_plus_graph_with_graph_rerank":
            return graph_supported_search(
                entries=entries,
                embedder=self.embedder,
                graph=self.loaded_index.graph,
                query=query,
                top_k=top_k,
                mode="hybrid_plus_graph_with_graph_rerank",
            )
        return search_embedded_entries(
            entries=entries,
            query=query,
            embedder=self.embedder,
            top_k=top_k,
            mode="hybrid_filtered_rerank",
        )

    def _graph_path(self, *, query: str, results: list[RetrievalResult], mode_used: ResolvedMode, top_k: int) -> list[str]:
        if mode_used != "hybrid_plus_graph_with_graph_rerank" or not results:
            return []
        graph_results = search_graph(self.loaded_index.graph, query, top_k=min(top_k, 3))
        if not graph_results:
            return []
        top_document_id = results[0].chunk.document_id
        return [
            f"{graph_result.reason} -> {graph_result.chunk.act_title} {graph_result.chunk.unit_type.title()} {graph_result.chunk.unit_id or graph_result.chunk.section_id}"
            for graph_result in graph_results
            if graph_result.chunk.document_id == top_document_id
        ]

    def _generate_grounded_answer(self, *, query: str, results: list[RetrievalResult]) -> str:
        messages = self._build_generation_messages(query=query, results=results)
        return self.ollama_transport.chat(
            model=self.ollama_model,
            messages=messages,
            api_key=self.ollama_api_key,
        )

    def _build_generation_messages(self, *, query: str, results: list[RetrievalResult]) -> list[dict[str, str]]:
        checklist_mode = is_employment_agreement_checklist_query(query)
        grouped_sources = aggregate_source_items(results, top_k=len(results))
        return [
            {
                "role": "system",
                "content": (
                    "You are a Malaysian legal RAG assistant. "
                    "Answer only from the retrieved legal sources. "
                    "Do not dump raw statutory text. Summarize the retrieved provisions in plain language. "
                    "Quote only short phrases when absolutely necessary. "
                    "Do not give legal advice. "
                    "If the evidence is narrow or incomplete, say so clearly. "
                    "Keep the answer concise, user-facing, and under 500 words unless the user asks for detail. "
                    + (
                        "For broad employment-agreement questions, use exactly this structure:\n"
                        "Direct Answer:\n"
                        "Checklist:\n"
                        "Legal Basis:\n"
                        "Important Limits:\n"
                        "Sources:"
                        if checklist_mode
                        else "Use exactly this structure:\n"
                        "Direct Answer:\n"
                        "Legal Basis:\n"
                        "Practical Meaning:\n"
                        "Important Limits:\n"
                        "Sources:"
                    )
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\n"
                    f"Retrieved sources:\n{build_grouped_context_block(grouped_sources) if checklist_mode else build_context_block(results)}\n\n"
                    "Write a concise user-facing answer.\n"
                    + (
                        "Requirements:\n"
                        "- Direct Answer: 1 short paragraph saying what the user should check before signing.\n"
                        "- Checklist: grounded bullet points for written terms, wages, working hours, leave, termination, deductions, and employee protections only when supported by the retrieved sources.\n"
                        "- If a point is only a practical check and not clearly supported by the retrieved law text, label it as a practical check rather than a legal conclusion.\n"
                        "- Legal Basis: map checklist items to the retrieved Act and section/article/perkara.\n"
                        "- Important Limits: say this is legal information, not legal advice, and note gaps in the retrieved evidence.\n"
                        "- Sources: list the grouped section-level sources used as bullet points.\n"
                        "- Do not paste long chunk text."
                        if checklist_mode
                        else "Requirements:\n"
                        "- Direct Answer: 1 short paragraph.\n"
                        "- Legal Basis: cite the Act and section/article/perkara and summarize the legal point.\n"
                        "- Practical Meaning: explain what it means in plain English.\n"
                        "- Important Limits: mention factual limits, ambiguity, or incomplete evidence.\n"
                        "- Sources: list the retrieved sources used as bullet points.\n"
                        "- Do not paste long chunk text."
                    )
                ),
            },
        ]

    def _stream_event(self, event_type: str, payload: dict[str, object]) -> str:
        return json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n"

    def _log_chat_event(
        self,
        *,
        request_id: str | None,
        query: str,
        mode_used: ResolvedMode,
        principal_role: str | None,
        result_count: int,
        fallback_reason: str | None,
        stream: bool,
    ) -> None:
        payload = {
            "request_id": request_id,
            "mode_used": mode_used,
            "principal_role": principal_role,
            "result_count": result_count,
            "fallback_reason": fallback_reason,
            "stream": stream,
            **build_query_log_fields(query),
        }
        log_event("chat_request", **payload)

    def _vector_store_path(self) -> Path:
        configured = os.getenv("LEGAL_RAG_VECTOR_STORE")
        if configured:
            return (self.settings.project_root / configured).resolve()
        return self.settings.embeddings_dir / "legal-corpus.vectors.jsonl"


def route_mode(query: str, requested_mode: ChatMode) -> ResolvedMode:
    if requested_mode == "hybrid":
        return "hybrid_filtered_rerank"
    if requested_mode == "graph":
        return "hybrid_plus_graph_with_graph_rerank"
    if GRAPH_ROUTE_PATTERN.search(query):
        return "hybrid_plus_graph_with_graph_rerank"
    return "hybrid_filtered_rerank"


def should_abstain_for_explicit_unit_lookup(query: str, results: list[RetrievalResult]) -> bool:
    match = EXPLICIT_UNIT_PATTERN.search(query)
    if not match:
        return False
    requested_unit = match.group(2).strip().lower()
    if not results:
        return True
    for result in results[:5]:
        candidate = (result.chunk.unit_id or result.chunk.section_id or "").strip().lower()
        if candidate == requested_unit:
            return False
    return True


def format_source_item(result: RetrievalResult) -> SourceItem:
    chunk = result.chunk
    unit_id = chunk.unit_id or chunk.section_id
    return SourceItem(
        document=chunk.act_title or chunk.document_id,
        unit_type=chunk.unit_type,
        unit_id=unit_id,
        heading=chunk.section_heading,
        score=round(float(result.score), 4),
        chunk_count=1,
        preview=preview_text(chunk.text),
    )


def aggregate_source_items(results: list[RetrievalResult], top_k: int) -> list[SourceItem]:
    grouped: OrderedDict[tuple[str, str, str], dict[str, object]] = OrderedDict()
    for result in results:
        chunk = result.chunk
        document = chunk.act_title or chunk.document_id
        unit_type = chunk.unit_type
        unit_id = chunk.unit_id or chunk.section_id
        key = (document, unit_type, unit_id)
        entry = grouped.get(key)
        if entry is None:
            grouped[key] = {
                "document": document,
                "unit_type": unit_type,
                "unit_id": unit_id,
                "headings": [chunk.section_heading] if chunk.section_heading else [],
                "score": float(result.score),
                "previews": [preview_text(chunk.text)] if chunk.text else [],
                "chunk_count": 1,
            }
            continue

        entry["score"] = max(float(entry["score"]), float(result.score))
        entry["chunk_count"] = int(entry["chunk_count"]) + 1
        if chunk.section_heading and chunk.section_heading not in entry["headings"]:
            entry["headings"].append(chunk.section_heading)
        preview = preview_text(chunk.text)
        if preview and preview not in entry["previews"]:
            entry["previews"].append(preview)

    aggregated: list[SourceItem] = []
    for entry in list(grouped.values())[:top_k]:
        headings = [heading for heading in entry["headings"] if heading]
        previews = [preview for preview in entry["previews"] if preview]
        heading = " / ".join(headings[:2]) if headings else ""
        preview = " ".join(previews[:2]).strip()
        aggregated.append(
            SourceItem(
                document=str(entry["document"]),
                unit_type=str(entry["unit_type"]),
                unit_id=str(entry["unit_id"]),
                heading=heading,
                score=round(float(entry["score"]), 4),
                chunk_count=int(entry["chunk_count"]),
                preview=preview,
            )
        )
    return aggregated


def preview_text(text: str, limit: int = 260) -> str:
    compact = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if len(compact) <= limit:
        return compact
    truncated = compact[:limit].rsplit(" ", 1)[0].strip()
    return f"{truncated}..."


def describe_ollama_failure(exc: Exception) -> str:
    if isinstance(exc, OllamaChatError):
        return exc.reason
    if isinstance(exc, requests.exceptions.Timeout):
        return "timeout"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "connection refused"
    if isinstance(exc, requests.exceptions.HTTPError):
        response = exc.response
        if response is not None:
            status_code = response.status_code
            if status_code in (401, 403):
                return "unauthorized"
            if status_code == 404:
                return "wrong base URL or /api/chat endpoint not found"
            text = (response.text or "").lower()
            if "model" in text and "not found" in text:
                return "model not found"
            return f"http {status_code}"
    return str(exc).strip() or "unknown error"


def _query_demands_abstention(query: str) -> bool:
    return bool(EXPLICIT_UNIT_PATTERN.search(query) or LEGAL_ADVICE_PATTERN.search(query))


def chunk_text_for_streaming(text: str, chunk_size: int = 48) -> Iterator[str]:
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        yield text[start:end]
        start = end


def build_grouped_context_block(sources: list[SourceItem]) -> str:
    blocks: list[str] = []
    for index, source in enumerate(sources, start=1):
        blocks.append(
            f"[Source {index}] {source.document} | {source.unit_type.title()} {source.unit_id} | {source.heading} | Chunks: {source.chunk_count}\n"
            f"Summary: {source.preview}"
        )
    return "\n\n".join(blocks)


def build_context_block(results: list[RetrievalResult]) -> str:
    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        unit_id = chunk.unit_id or chunk.section_id
        excerpt = summarize_context_snippet(chunk.text)
        blocks.append(
            f"[Source {index}] {chunk.act_title} | {chunk.unit_type.title()} {unit_id} | {chunk.section_heading}\n"
            f"Summary: {excerpt}"
        )
    return "\n\n".join(blocks)


def summarize_context_snippet(text: str, limit: int = 280) -> str:
    compact = " ".join(line.strip() for line in text.splitlines() if line.strip())
    sentence = compact.split(". ", 1)[0].strip()
    summary = sentence or compact
    if len(summary) > limit:
        summary = summary[:limit].rsplit(" ", 1)[0].strip() + "..."
    return summary


def is_employment_agreement_checklist_query(query: str) -> bool:
    lowered = query.lower()
    return bool(EMPLOYMENT_AGREEMENT_PATTERN.search(lowered)) or (
        "employment" in lowered
        and ("agreement" in lowered or "contract" in lowered)
        and any(token in lowered for token in ("check", "sign", "protect", "rights", "terms"))
    )
