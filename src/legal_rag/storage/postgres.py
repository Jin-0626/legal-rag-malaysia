"""PostgreSQL/pgvector production scaffold and health checks."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DatabaseHealthStatus:
    enabled: bool
    connected: bool
    backend: str = "postgresql+pgvector"
    error: str | None = None


def check_database_health() -> DatabaseHealthStatus:
    enabled = _env_flag("LEGAL_RAG_DATABASE_ENABLED", default=False) or bool(os.getenv("DATABASE_URL"))
    if not enabled:
        return DatabaseHealthStatus(enabled=False, connected=False, error=None)

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return DatabaseHealthStatus(
            enabled=True,
            connected=False,
            error="DATABASE_URL is not configured.",
        )

    try:
        import psycopg
    except ImportError:
        return DatabaseHealthStatus(
            enabled=True,
            connected=False,
            error="psycopg is not installed.",
        )

    try:
        with psycopg.connect(database_url, connect_timeout=5) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1;")
                cursor.fetchone()
                cursor.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector');")
                vector_enabled = bool(cursor.fetchone()[0])
    except Exception as exc:
        return DatabaseHealthStatus(
            enabled=True,
            connected=False,
            error=str(exc).strip() or "database connection failed",
        )

    if not vector_enabled:
        return DatabaseHealthStatus(
            enabled=True,
            connected=True,
            error="pgvector extension is not enabled in the target database.",
        )
    return DatabaseHealthStatus(enabled=True, connected=True, error=None)


def _env_flag(name: str, *, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}
