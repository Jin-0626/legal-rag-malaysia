"""FastAPI application for the Legal RAG chatbot demo."""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from legal_rag.api.logging_utils import RequestLoggingMiddleware, configure_logging
from legal_rag.api.schemas import ChatRequest, ChatResponse, HealthResponse
from legal_rag.api.security import Principal, build_security_summary, require_role
from legal_rag.api.service import LegalRAGChatService

configure_logging()


@lru_cache(maxsize=1)
def get_service() -> LegalRAGChatService:
    return LegalRAGChatService()


app = FastAPI(title="Malaysia Legal RAG API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)


@app.get("/health", response_model=HealthResponse)
def health(service: LegalRAGChatService = Depends(get_service)) -> HealthResponse:
    return service.health()


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    http_request: Request,
    principal: Principal = Depends(require_role("admin", "researcher", "viewer")),
    service: LegalRAGChatService = Depends(get_service),
) -> ChatResponse:
    return service.chat(
        query=request.query,
        mode=request.mode,
        top_k=request.top_k,
        request_id=getattr(http_request.state, "request_id", None),
        principal_role=principal.role,
    )


@app.post("/chat_stream")
def chat_stream(
    request: ChatRequest,
    http_request: Request,
    principal: Principal = Depends(require_role("admin", "researcher", "viewer")),
    service: LegalRAGChatService = Depends(get_service),
) -> StreamingResponse:
    return StreamingResponse(
        service.chat_stream(
            query=request.query,
            mode=request.mode,
            top_k=request.top_k,
            request_id=getattr(http_request.state, "request_id", None),
            principal_role=principal.role,
        ),
        media_type="application/x-ndjson",
    )


@app.get("/admin/security")
def admin_security(principal: Principal = Depends(require_role("admin"))) -> dict[str, object]:
    return build_security_summary()
