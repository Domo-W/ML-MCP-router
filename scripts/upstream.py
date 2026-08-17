"""Call official upstream MCPs after the LoRA routes.

Hugging Face / W&B: Streamable HTTP (HF_MCP_URL, WANDB_MCP_URL).
Sessions stay open on a background loop (reconnect on failure).
arXiv: stdio (ARXIV_CMD + ARXIV_STORAGE), one-shot per call.

  E:/Anaconda/envs/finetuning/python.exe scripts/upstream.py --probe
  E:/Anaconda/envs/finetuning/python.exe scripts/upstream.py --probe wandb
  E:/Anaconda/envs/finetuning/python.exe scripts/upstream.py --probe arxiv
"""

from __future__ import annotations

import atexit
import asyncio
import json
import os
import sys
import threading
from typing import Any

DEFAULT_HF_URL = "https://huggingface.co/mcp"
DEFAULT_WANDB_URL = "https://mcp.withwandb.com/mcp"
DEFAULT_ARXIV_CMD = r"C:\Users\Admin\.local\bin\arxiv-mcp-server.exe"
DEFAULT_ARXIV_STORAGE = "E:/Github/FineTuning/.arxiv-cache"
PLACEHOLDER_TOKENS = frozenset(
    {
        "",
        "hf_REPLACE_ME",
        "${env:HF_TOKEN}",
        "wandb_REPLACE_ME",
        "${env:WANDB_API_KEY}",
    }
)
MAX_RESULT_CHARS = 8000
HTTP_CALL_TIMEOUT_S = 180.0

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()
_loop_ready = threading.Event()
_http_call_lock = threading.Lock()
_sessions: dict[str, tuple[Any, Any]] = {}


def execute(server: str, name: str, arguments: dict) -> Any:
    if server in {"huggingface", "wandb"}:
        with _http_call_lock:
            return _run_on_loop(_call_http_pooled(server, name, arguments))
    if server == "arxiv":
        return asyncio.run(_call_arxiv(name, arguments))
    raise RuntimeError(f"upstream not wired for server {server!r}")


def _ensure_loop() -> asyncio.AbstractEventLoop:
    global _loop
    if _loop is not None and _loop.is_running():
        return _loop
    with _loop_lock:
        if _loop is not None and _loop.is_running():
            return _loop
        _loop_ready.clear()
        loop = asyncio.new_event_loop()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            _loop_ready.set()
            loop.run_forever()

        thread = threading.Thread(target=_run, name="mcp-upstream-http", daemon=True)
        thread.start()
        if not _loop_ready.wait(timeout=5):
            raise RuntimeError("upstream HTTP loop failed to start")
        _loop = loop
        return loop


def _run_on_loop(coro: Any) -> Any:
    loop = _ensure_loop()
    return asyncio.run_coroutine_threadsafe(coro, loop).result(
        timeout=HTTP_CALL_TIMEOUT_S
    )


async def _close_session(server: str) -> None:
    pair = _sessions.pop(server, None)
    if pair is None:
        return
    http, client = pair
    try:
        await client.__aexit__(None, None, None)
    except Exception:
        pass
    try:
        await http.__aexit__(None, None, None)
    except Exception:
        pass


async def _get_client(server: str) -> Any:
    if server in _sessions:
        return _sessions[server][1]
    url, token = _http_conf(server)
    http, client = await _http_connect(url, token)
    _sessions[server] = (http, client)
    print(f"upstream {server} session open", file=sys.stderr, flush=True)
    return client


async def _call_http_pooled(server: str, name: str, arguments: dict) -> Any:
    client = await _get_client(server)
    try:
        result = await client.call_tool(name, arguments)
    except Exception as exc:
        print(
            f"upstream {server} session dead ({exc!r}); reconnecting",
            file=sys.stderr,
            flush=True,
        )
        await _close_session(server)
        client = await _get_client(server)
        result = await client.call_tool(name, arguments)
    return _trim(_result_payload(result))


def _shutdown_http() -> None:
    loop = _loop
    if loop is None or not loop.is_running():
        return

    async def _close_all() -> None:
        for server in list(_sessions):
            await _close_session(server)

    try:
        asyncio.run_coroutine_threadsafe(_close_all(), loop).result(timeout=10)
    except Exception:
        pass
    loop.call_soon_threadsafe(loop.stop)


atexit.register(_shutdown_http)


def _require_token(env_name: str, hint: str) -> str:
    token = (os.environ.get(env_name) or "").strip()
    if token in PLACEHOLDER_TOKENS:
        raise RuntimeError(
            f"{env_name} is unset or still a placeholder. {hint}"
        )
    return token


async def _http_connect(url: str, token: str):
    from mcp.client.client import Client
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client

    headers = {
        "Authorization": f"Bearer {token}",
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


def _http_conf(server: str) -> tuple[str, str]:
    if server == "huggingface":
        url = os.environ.get("HF_MCP_URL") or DEFAULT_HF_URL
        token = _require_token(
            "HF_TOKEN",
            "Put a token from https://huggingface.co/settings/tokens in ml-router env.",
        )
        return url, token
    if server == "wandb":
        url = os.environ.get("WANDB_MCP_URL") or DEFAULT_WANDB_URL
        token = _require_token(
            "WANDB_API_KEY",
            "Put the same W&B key used by the wandb MCP server in ml-router env.",
        )
        return url, token
    raise RuntimeError(f"not an HTTP server: {server}")


async def _call_http(server: str, name: str, arguments: dict) -> Any:
    url, token = _http_conf(server)
    http, client = await _http_connect(url, token)
    try:
        result = await client.call_tool(name, arguments)
        return _trim(_result_payload(result))
    finally:
        await client.__aexit__(None, None, None)
        await http.__aexit__(None, None, None)


def _arxiv_params():
    from mcp.client.stdio import StdioServerParameters

    command = os.environ.get("ARXIV_CMD") or DEFAULT_ARXIV_CMD
    storage = os.environ.get("ARXIV_STORAGE") or DEFAULT_ARXIV_STORAGE
    return StdioServerParameters(
        command=command,
        args=["--storage-path", storage],
    )


async def _arxiv_connect():
    from mcp.client.client import Client
    from mcp.client.stdio import stdio_client

    client = Client(stdio_client(_arxiv_params()))
    await client.__aenter__()
    return client


async def _call_arxiv(name: str, arguments: dict) -> Any:
    client = await _arxiv_connect()
    try:
        result = await client.call_tool(name, arguments)
        return _trim(_result_payload(result))
    finally:
        await client.__aexit__(None, None, None)


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


async def _list_tools(client) -> list[str]:
    listed = await client.list_tools()
    return [t.name for t in listed.tools]


async def _probe_http(server: str) -> dict:
    url, token = _http_conf(server)
    http, client = await _http_connect(url, token)
    try:
        return {"ok": True, "server": server, "url": url, "tools": await _list_tools(client)}
    finally:
        await client.__aexit__(None, None, None)
        await http.__aexit__(None, None, None)


async def _probe_arxiv() -> dict:
    client = await _arxiv_connect()
    try:
        return {"ok": True, "server": "arxiv", "tools": await _list_tools(client)}
    finally:
        await client.__aexit__(None, None, None)


async def _probe(which: str) -> int:
    targets = ("huggingface", "wandb", "arxiv") if which == "all" else (which,)
    rows = []
    failed = 0
    for target in targets:
        try:
            if target == "arxiv":
                rows.append(await _probe_arxiv())
            elif target in {"huggingface", "hf"}:
                rows.append(await _probe_http("huggingface"))
            elif target == "wandb":
                rows.append(await _probe_http("wandb"))
            else:
                raise RuntimeError(f"unknown probe {target!r}")
        except Exception as exc:
            failed += 1
            rows.append({"ok": False, "server": target, "error": str(exc)})
    print(json.dumps(rows if which == "all" else rows[0], indent=2))
    return 1 if failed else 0


def main() -> int:
    if "--probe" not in sys.argv:
        raise SystemExit(
            "usage: python scripts/upstream.py --probe [hf|wandb|arxiv|all]"
        )
    extra = [a for a in sys.argv[1:] if a != "--probe"]
    which = extra[0] if extra else "huggingface"
    if which == "hf":
        which = "huggingface"
    return asyncio.run(_probe(which))


if __name__ == "__main__":
    raise SystemExit(main())
