"""Smoke-check the configured Ollama endpoint and model."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from legal_rag.api.service import RequestsOllamaChatTransport  # noqa: E402


def main() -> int:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")
    api_key = os.getenv("OLLAMA_API_KEY") or None
    timeout_seconds = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))

    transport = RequestsOllamaChatTransport(base_url, timeout_seconds=timeout_seconds)
    status = transport.health_details(model=model, api_key=api_key, probe_chat=True)

    print(f"Ollama base URL: {base_url}")
    print(f"Configured model: {model}")
    print(f"Authorization: {'Bearer token configured' if api_key else 'No API key'}")
    print(f"/api/tags reachable: {'PASS' if status.ollama_available else 'FAIL'}")
    print(f"Model available: {'PASS' if status.model_available else 'FAIL'}")
    print(f"/api/chat ready: {'PASS' if status.chat_ready else 'FAIL'}")
    if status.error:
        print(f"Error: {status.error}")
        if not status.model_available:
            print(f"Hint: run `ollama pull {model}` or update OLLAMA_MODEL.")
        elif status.error == "connection refused":
            print("Hint: start Ollama with `ollama serve` and verify OLLAMA_BASE_URL.")
        elif status.error == "unauthorized":
            print("Hint: verify OLLAMA_API_KEY for your remote or proxied endpoint.")
    return 0 if status.chat_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
