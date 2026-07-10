#!/usr/bin/env python3
"""Register or rotate the Together AI key as OpenRouter BYOK.

OpenRouter BYOK is account/workspace-level: once registered, every completion
key in that workspace can use your Together quota for Together-routed models
(e.g. MiniMax-M3 with provider.order=["together"]).

Requires:
  TOGETHER_API_KEY — the Together provider key to store
  OPENROUTER_MANAGEMENT_API_KEY — an OpenRouter *management* key (not a
    normal completion key). Create one in the OpenRouter dashboard.

Usage:
  uv run python scripts/register_openrouter_together_byok.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

BYOK_URL = "https://openrouter.ai/api/v1/byok"
PROVIDER = "together"


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def _request(
    method: str,
    url: str,
    token: str,
    body: dict | None = None,
) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode() or "{}"
            return resp.status, json.loads(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode() or "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"error": {"message": raw}}
        return exc.code, payload


def main() -> int:
    _load_dotenv()
    together = os.environ.get("TOGETHER_API_KEY", "").strip()
    management = os.environ.get("OPENROUTER_MANAGEMENT_API_KEY", "").strip()
    if not together:
        print("TOGETHER_API_KEY is required", file=sys.stderr)
        return 1
    if not management:
        print(
            "OPENROUTER_MANAGEMENT_API_KEY is required "
            "(a normal OPENROUTER_API_KEY cannot create BYOK).",
            file=sys.stderr,
        )
        return 1

    status, listed = _request("GET", f"{BYOK_URL}?provider={PROVIDER}", management)
    if status == 401:
        print(
            "Invalid management key. Create one in the OpenRouter dashboard "
            "(Management API), then set OPENROUTER_MANAGEMENT_API_KEY.",
            file=sys.stderr,
        )
        return 1
    if status >= 400:
        print(f"Failed to list BYOK credentials ({status}): {listed}", file=sys.stderr)
        return 1

    existing = listed.get("data") or listed.get("keys") or []
    if isinstance(existing, dict):
        existing = existing.get("items") or existing.get("data") or []
    together_creds = [
        item
        for item in existing
        if isinstance(item, dict) and item.get("provider") == PROVIDER
    ]

    if together_creds:
        cred_id = together_creds[0].get("id")
        if not cred_id:
            print(f"Unexpected BYOK list payload: {listed}", file=sys.stderr)
            return 1
        status, updated = _request(
            "PATCH",
            f"{BYOK_URL}/{cred_id}",
            management,
            {"key": together},
        )
        if status >= 400:
            print(
                f"Failed to rotate Together BYOK ({status}): {updated}",
                file=sys.stderr,
            )
            return 1
        print(f"Rotated OpenRouter Together BYOK credential {cred_id}")
        return 0

    status, created = _request(
        "POST",
        BYOK_URL,
        management,
        {"provider": PROVIDER, "key": together},
    )
    if status >= 400:
        print(f"Failed to create Together BYOK ({status}): {created}", file=sys.stderr)
        return 1
    cred_id = (created.get("data") or {}).get("id") or created.get("id")
    print(f"Created OpenRouter Together BYOK credential {cred_id or '(ok)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
