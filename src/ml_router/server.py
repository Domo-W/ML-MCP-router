"""Stdio MCP: one tool, route_ml_tools({intent}).

  ml-router
  ml-router --smoke
  ML_ROUTER_BACKEND=llama|ollama
"""

from __future__ import annotations

import json
import os
import sys
import threading

from mcp.server.mcpserver import MCPServer

from ml_router.catalog import load_catalog
from ml_router.generate import make_generate
from ml_router.prompt import system_prompt
from ml_router.route import route
from ml_router.upstream import execute

SMOKE = [
    "Is run k7m2n9p4 in train-lab/rl-benchmark overfitting?",
    "Show me the loss curve for k7m2n9p4 in train-lab/rl-benchmark",
    "Best Qwen 3B instruct GGUF",
    "Abstract of 2106.09685",
]

_lock = threading.Lock()
_loaded: dict | None = None


def _ensure_loaded() -> dict:
    global _loaded
    if _loaded is not None:
        return _loaded
    with _lock:
        if _loaded is not None:
            return _loaded
        backend = os.environ.get("ML_ROUTER_BACKEND", "llama")
        catalog = load_catalog()
        print(f"backend={backend}", file=sys.stderr, flush=True)
        generate_fn = make_generate(backend)
        print("ready", file=sys.stderr, flush=True)
        _loaded = {
            "catalog": catalog,
            "system": system_prompt(catalog),
            "generate": generate_fn,
        }
        return _loaded


mcp = MCPServer("ml-router")


@mcp.tool()
def route_ml_tools(intent: str) -> str:
    """Turn a researcher question into one W&B, Hugging Face, or arXiv tool call.

    Pass the user's request as intent. Returns JSON
    {ok, name, arguments, server} or {ok: false, error}.
    Routed calls are executed on Hugging Face, W&B, and arXiv.
    """
    state = _ensure_loaded()
    rec = route(
        state["generate"],
        state["system"],
        state["catalog"],
        intent,
        max_new_tokens=256,
    )
    rec.pop("raw", None)
    if rec.get("ok") and rec.get("server") in {"huggingface", "wandb", "arxiv"}:
        try:
            rec["result"] = execute(rec["server"], rec["name"], rec["arguments"])
        except Exception as exc:
            rec["result_error"] = str(exc)
    return json.dumps(rec, ensure_ascii=False)


def smoke() -> int:
    state = _ensure_loaded()
    failed = 0
    for intent in SMOKE:
        rec = route(
            state["generate"],
            state["system"],
            state["catalog"],
            intent,
            max_new_tokens=256,
        )
        print(json.dumps({"intent": intent, **rec}, ensure_ascii=False, indent=2))
        if not rec["ok"]:
            failed += 1
    return 1 if failed else 0


def main() -> None:
    if "--smoke" in sys.argv:
        raise SystemExit(smoke())
    mcp.run(transport="stdio")
