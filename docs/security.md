# Security

## Security Posture

This project is designed to demonstrate production hardening around a working legal RAG stack without claiming full legal-production certification.

Implemented controls:

- API key authentication for the FastAPI service
- RBAC with `admin`, `researcher`, and `viewer`
- optional auth-off local development mode
- structured request logging with request IDs
- no raw API key logging
- no secrets committed in the repository
- explicit legal-answer limits and abstention behavior

## API Key Mode

Environment flags:

```env
LEGAL_RAG_REQUIRE_API_KEY=true
LEGAL_RAG_API_KEYS_FILE=config/api_keys.example.json
```

Header:

```text
X-API-Key: <your-key>
```

Keys are compared using SHA-256 hashes and constant-time comparison.

## RBAC

Roles:

- `admin`: full access, including `/admin/security`
- `researcher`: chat and retrieval-facing operations
- `viewer`: chat-only access

Applied endpoints:

- `POST /chat`
- `POST /chat_stream`
- `GET /admin/security` (admin only)

## Logging

Structured logs include:

- request ID
- route
- method
- status code
- latency
- retrieval mode
- fallback reason
- principal role

Production logging avoids full raw query text by default and logs query length plus a short hash instead.

## Secret Handling

- `.env` is ignored
- `config/api_keys.example.json` is a nonfunctional template only
- do not commit real keys or tokens
- if a key has ever been exposed, rotate it before release

## How To Create API Keys Safely

Generate a SHA-256 hash without storing the raw key in the repo:

```powershell
.\venv_new\Scripts\python.exe scripts/create_api_key_hash.py --name local-admin --role admin
```

Or pass the key explicitly if you are in a safe shell history context:

```powershell
.\venv_new\Scripts\python.exe scripts/create_api_key_hash.py --name local-admin --role admin --key "your-real-key"
```

The helper prints:

- the SHA-256 hash
- an example JSON entry for your local `config/api_keys.json`

Do not commit a real `config/api_keys.json` file.

## Legal Safety

- answers are grounded in retrieved sources only
- unsupported or weak evidence should abstain
- impossible unit lookups should not hallucinate a nearest answer
- outputs are legal information, not legal advice
