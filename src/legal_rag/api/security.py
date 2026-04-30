"""API key authentication and simple RBAC for the Legal RAG API."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from fastapi import Depends, Header, HTTPException, status

from legal_rag.config.settings import build_settings

Role = Literal["admin", "researcher", "viewer"]


@dataclass(frozen=True)
class Principal:
    name: str
    role: Role
    auth_enabled: bool


@dataclass(frozen=True)
class APIKeyRecord:
    name: str
    key_hash: str
    role: Role


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def is_api_key_required() -> bool:
    return _env_flag("LEGAL_RAG_REQUIRE_API_KEY", default=False)


def build_security_summary() -> dict[str, object]:
    auth_enabled = is_api_key_required()
    file_path = _api_keys_file_path()
    records = _load_api_key_records() if auth_enabled else []
    return {
        "api_key_required": auth_enabled,
        "api_keys_file": str(file_path),
        "configured_keys": len(records),
        "roles": ["admin", "researcher", "viewer"],
    }


def get_current_principal(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> Principal:
    if not is_api_key_required():
        return Principal(name="local-dev", role="admin", auth_enabled=False)

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
        )

    records = _load_api_key_records()
    supplied_hash = hash_api_key(x_api_key)
    for record in records:
        if hmac.compare_digest(record.key_hash, supplied_hash):
            return Principal(name=record.name, role=record.role, auth_enabled=True)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key.",
    )


def require_role(*allowed_roles: Role) -> Callable[[Principal], Principal]:
    def dependency(principal: Principal = Depends(get_current_principal)) -> Principal:
        if principal.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{principal.role}' is not permitted for this endpoint.",
            )
        return principal

    return dependency


def _load_api_key_records() -> list[APIKeyRecord]:
    file_path = _api_keys_file_path()
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"API key configuration file is missing: {file_path}",
        )

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    raw_records = payload.get("keys", [])
    records: list[APIKeyRecord] = []
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            continue
        name = str(raw_record.get("name", "")).strip()
        key_hash = str(raw_record.get("key_hash", "")).strip()
        role = str(raw_record.get("role", "")).strip().lower()
        if not name or not key_hash or role not in {"admin", "researcher", "viewer"}:
            continue
        records.append(APIKeyRecord(name=name, key_hash=key_hash, role=role))

    if not records:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key configuration does not contain any valid key records.",
        )
    return records


def _api_keys_file_path() -> Path:
    configured = os.getenv("LEGAL_RAG_API_KEYS_FILE")
    if configured:
        return Path(configured)
    return build_settings().project_root / "config" / "api_keys.example.json"


def _env_flag(name: str, *, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}
