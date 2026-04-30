"""Structured logging helpers for the Legal RAG API."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

LOGGER_NAME = "legal_rag.api"


def configure_logging() -> None:
    logger = logging.getLogger(LOGGER_NAME)
    if getattr(configure_logging, "_configured", False):
        return

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(message)s")
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    configure_logging._configured = True


def log_event(event: str, **fields: Any) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def build_query_log_fields(query: str) -> dict[str, Any]:
    if os.getenv("APP_ENV", "development").lower() == "production":
        return {
            "query_length": len(query),
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest()[:16],
        }
    preview = query if len(query) <= 140 else query[:137] + "..."
    return {
        "query_length": len(query),
        "query_preview": preview,
    }


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            log_event(
                "http_request",
                request_id=request_id,
                route=request.url.path,
                method=request.method,
                status_code=500,
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
            )
            raise

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        log_event(
            "http_request",
            request_id=request_id,
            route=request.url.path,
            method=request.method,
            status_code=response.status_code,
            latency_ms=latency_ms,
        )
        return response
