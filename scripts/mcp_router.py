"""Stdio MCP: one tool, route_ml_tools({intent}).

Load the QLoRA adapter once (on first call). Hugging Face calls are
executed against HF_MCP_URL. W&B / arXiv still return the plan only.

  E:/Anaconda/envs/finetuning/python.exe scripts/mcp_router.py
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_adapter import DEFAULT_ADAPTER, DEFAULT_MODEL, load_model  # noqa: E402
from eval_ollama import system_prompt  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402
from route_once import route  # noqa: E402
from upstream import execute  # noqa: E402
from validate_canonical import CATALOG, load_json  # noqa: E402

_lock = threading.Lock()
_loaded: dict | None = None


def _ensure_loaded() -> dict:
    global _loaded
    if _loaded is not None:
        return _loaded
    with _lock:
        if _loaded is not None:
            return _loaded
        catalog = load_json(CATALOG)
        print(
            f"loading {DEFAULT_MODEL} + {DEFAULT_ADAPTER} ...",
            file=sys.stderr,
            flush=True,
        )
        tokenizer, model = load_model(DEFAULT_MODEL, DEFAULT_ADAPTER)
        print("ready", file=sys.stderr, flush=True)
        _loaded = {
            "catalog": catalog,
            "system": system_prompt("slm_no_schema", catalog),
            "tokenizer": tokenizer,
            "model": model,
        }
        return _loaded


mcp = MCPServer("ml-router")


@mcp.tool()
def route_ml_tools(intent: str) -> str:
    """Turn a researcher question into one W&B, Hugging Face, or arXiv tool call.

    Pass the user's request as intent. Returns JSON
    {ok, name, arguments, server} or {ok: false, error}.
    Hugging Face calls are executed; W&B / arXiv return the plan only.
    """
    state = _ensure_loaded()
    rec = route(
        state["tokenizer"],
        state["model"],
        state["system"],
        state["catalog"],
        intent,
        max_new_tokens=256,
    )
    rec.pop("raw", None)
    if rec.get("ok") and rec.get("server") == "huggingface":
        try:
            rec["result"] = execute(rec["server"], rec["name"], rec["arguments"])
        except Exception as exc:
            rec["result_error"] = str(exc)
    return json.dumps(rec, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
