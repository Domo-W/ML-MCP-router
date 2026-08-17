"""Ollama generate backend. Does not start or own Ollama."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

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


def _ping(host: str, model: str) -> None:
    url = host + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"cannot reach Ollama at {host} ({exc}). Is it running?") from exc
    found = [t.get("name") or t.get("model") for t in tags.get("models", [])]
    if model not in found and not any(n.startswith(model) for n in found if n):
        raise RuntimeError(
            f"model {model!r} not in `ollama list` "
            f"({', '.join(n for n in found if n) or 'none'})."
        )


def generate(system: str, user: str, max_new_tokens: int) -> tuple[str, int]:
    global _pinged
    conf = _conf()
    if not _pinged:
        _ping(conf["host"], conf["model"])
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
