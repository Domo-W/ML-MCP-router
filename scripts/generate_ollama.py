"""Ollama generate backend. Does not start or own Ollama.

Talks to an already-running daemon (localhost:11434). Short keep_alive
so the 3B is not pinned after idle.

  ML_ROUTER_OLLAMA_HOST       (default http://127.0.0.1:11434)
  ML_ROUTER_OLLAMA_MODEL      (default mcp-slm:q4km)
  ML_ROUTER_OLLAMA_KEEP_ALIVE (default 2m)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from eval_ollama import ping

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "mcp-slm:q4km"
DEFAULT_KEEP_ALIVE = "2m"

_pinged = False


def _conf() -> dict:
    return {
        "host": os.environ.get("ML_ROUTER_OLLAMA_HOST", DEFAULT_HOST).rstrip("/"),
        "model": os.environ.get("ML_ROUTER_OLLAMA_MODEL", DEFAULT_MODEL),
        "keep_alive": os.environ.get("ML_ROUTER_OLLAMA_KEEP_ALIVE", DEFAULT_KEEP_ALIVE),
    }


def generate(system: str, user: str, max_new_tokens: int) -> tuple[str, int]:
    global _pinged
    conf = _conf()
    if not _pinged:
        try:
            ping(conf["host"], conf["model"])
        except SystemExit as exc:
            raise RuntimeError(str(exc)) from exc
        _pinged = True
    body = {
        "model": conf["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,
        "keep_alive": conf["keep_alive"],
        "options": {
            "temperature": 0,
            "num_predict": max_new_tokens,
            "num_ctx": 2048,
        },
    }
    req = urllib.request.Request(
        conf["host"] + "/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    msg = payload.get("message") or {}
    tokens = int(payload.get("prompt_eval_count") or 0)
    return str(msg.get("content") or "").strip(), tokens
