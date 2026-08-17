"""Stdio MCP: one tool, route_ml_tools({intent}).

Default generate backend is llama.cpp (GGUF via llama-server).
Hugging Face and W&B HTTP sessions stay open; arXiv is stdio one-shot.

  E:/Anaconda/envs/finetuning/python.exe scripts/mcp_router.py
  ML_ROUTER_BACKEND=llama   (default; owns llama-server)
  ML_ROUTER_BACKEND=ollama  (already-running daemon)
  ML_ROUTER_BACKEND=peft    (old 4-bit Transformers path)
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_ollama import system_prompt  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402
from route_once import route  # noqa: E402
from upstream import execute  # noqa: E402
from validate_canonical import CATALOG, load_json  # noqa: E402

_lock = threading.Lock()
_loaded: dict | None = None


def _make_generate(backend: str):
    if backend == "llama":
        from generate_llama import generate  # noqa: PLC0415

        return generate
    if backend == "ollama":
        from generate_ollama import generate  # noqa: PLC0415

        return generate
    if backend == "peft":
        from eval_adapter import DEFAULT_ADAPTER, DEFAULT_MODEL, generate, load_model  # noqa: PLC0415

        print(
            f"loading {DEFAULT_MODEL} + {DEFAULT_ADAPTER} ...",
            file=sys.stderr,
            flush=True,
        )
        tokenizer, model = load_model(DEFAULT_MODEL, DEFAULT_ADAPTER)

        def _peft(system: str, user: str, max_new_tokens: int):
            return generate(tokenizer, model, system, user, max_new_tokens)

        return _peft
    raise SystemExit(f"unknown ML_ROUTER_BACKEND={backend!r} (llama|ollama|peft)")


def _ensure_loaded() -> dict:
    global _loaded
    if _loaded is not None:
        return _loaded
    with _lock:
        if _loaded is not None:
            return _loaded
        backend = os.environ.get("ML_ROUTER_BACKEND", "llama")
        catalog = load_json(CATALOG)
        print(f"backend={backend}", file=sys.stderr, flush=True)
        generate_fn = _make_generate(backend)
        print("ready", file=sys.stderr, flush=True)
        _loaded = {
            "catalog": catalog,
            "system": system_prompt("slm_no_schema", catalog),
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


if __name__ == "__main__":
    mcp.run(transport="stdio")
