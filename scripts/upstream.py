"""Call official upstream MCPs after the LoRA routes.

Hugging Face: Streamable HTTP at HF_MCP_URL (default https://huggingface.co/mcp).
W&B / arXiv: not wired yet.

  E:/Anaconda/envs/finetuning/python.exe scripts/upstream.py --probe
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

DEFAULT_HF_URL = "https://huggingface.co/mcp"
PLACEHOLDER_TOKENS = frozenset({"", "hf_REPLACE_ME", "${env:HF_TOKEN}"})
MAX_RESULT_CHARS = 8000


def execute(server: str, name: str, arguments: dict) -> Any:
    if server == "huggingface":
        return asyncio.run(_call_hf(name, arguments))
    raise RuntimeError(f"upstream not wired for server {server!r}")


def _hf_token() -> str:
    token = (os.environ.get("HF_TOKEN") or "").strip()
    if token in PLACEHOLDER_TOKENS:
        raise RuntimeError(
            "HF_TOKEN is unset or still hf_REPLACE_ME. "
            "Put a real token from https://huggingface.co/settings/tokens "
            "in ml-router env."
        )
    return token


async def _hf_client():
    from mcp.client.client import Client
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client

    url = os.environ.get("HF_MCP_URL") or DEFAULT_HF_URL
    headers = {
        "Authorization": f"Bearer {_hf_token()}",
        "Accept": "application/json, text/event-stream",
    }
    http = create_mcp_http_client(headers=headers)
    await http.__aenter__()
    try:
        client = Client(streamable_http_client(url, http_client=http))
        await client.__aenter__()
        return http, client
    except Exception:
        await http.__aexit__(*sys.exc_info())
        raise


async def _call_hf(name: str, arguments: dict) -> Any:
    http, client = await _hf_client()
    try:
        result = await client.call_tool(name, arguments)
        return _trim(_result_payload(result))
    finally:
        await client.__aexit__(None, None, None)
        await http.__aexit__(None, None, None)


def _result_payload(result: Any) -> Any:
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return result


def _trim(payload: Any) -> Any:
    text = json.dumps(payload, ensure_ascii=False)
    if len(text) <= MAX_RESULT_CHARS:
        return payload
    return {
        "truncated": True,
        "preview": text[:MAX_RESULT_CHARS],
        "chars": len(text),
    }


async def _probe() -> int:
    http, client = await _hf_client()
    try:
        listed = await client.list_tools()
        names = [t.name for t in listed.tools]
        print(json.dumps({"ok": True, "tools": names}, indent=2))
        return 0
    finally:
        await client.__aexit__(None, None, None)
        await http.__aexit__(None, None, None)


def main() -> int:
    if "--probe" in sys.argv:
        return asyncio.run(_probe())
    raise SystemExit("usage: python scripts/upstream.py --probe")


if __name__ == "__main__":
    raise SystemExit(main())
