"""Safely generate a SHA-256 hash for a Legal RAG API key."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a SHA-256 hash for a Legal RAG API key.")
    parser.add_argument("--name", default="local-admin", help="Descriptive key name for the JSON entry.")
    parser.add_argument(
        "--role",
        default="admin",
        choices=["admin", "researcher", "viewer"],
        help="Role to associate with the API key.",
    )
    parser.add_argument("--key", default=None, help="Raw API key. If omitted, the script will prompt securely.")
    args = parser.parse_args()

    raw_key = args.key or getpass.getpass("Enter the raw API key to hash: ").strip()
    if not raw_key:
        raise SystemExit("A non-empty API key is required.")

    key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    print("SHA-256 hash:")
    print(key_hash)
    print()
    print("Example JSON entry:")
    print(
        json.dumps(
            {
                "name": args.name,
                "key_hash": key_hash,
                "role": args.role,
            },
            indent=2,
        )
    )
    print()
    print("Do not commit config/api_keys.json or any file containing real hashed production keys without review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
