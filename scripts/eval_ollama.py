"""Call local Ollama for phase-3 setups A / B / B'.

Writes prediction JSONL for eval_calls.py. Does not score.

  python scripts/eval_ollama.py --setup slm_no_schema --limit 5
  python scripts/eval_ollama.py --setup slm_v1_schema
  python scripts/eval_ollama.py --setup slm_full_schema
  python scripts/eval_ollama.py --setup slm_full_schema_post
  python scripts/eval_ollama.py --setup slm_gguf --model mcp-slm:q4km
  python scripts/eval_calls.py --pred data/eval/slm_no_schema.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_canonical import (  # noqa: E402
    CANONICAL,
    CATALOG,
    load_json,
    load_jsonl,
    v1_names,
)

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / "data" / "mcp" / "dump"
EVAL_DIR = ROOT / "data" / "eval"

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:3b-instruct"
SERVERS = ("wandb", "huggingface", "arxiv")
SETUPS = (
    "slm_no_schema",
    "slm_v1_schema",
    "slm_full_schema",
    "slm_full_schema_post",
    "slm_gguf",
)

JSON_ONLY = (
    'Reply with a single JSON object {"name":"...","arguments":{...}}. '
    "No markdown, no extra keys, no prose."
)


def compact_tool(tool: dict) -> dict:
    return {
        "name": tool["name"],
        "description": tool.get("description") or "",
        "inputSchema": tool.get("inputSchema") or {},
    }


def load_dump_tools() -> list[dict]:
    tools: list[dict] = []
    for server in SERVERS:
        payload = load_json(DUMP / f"{server}.json")
        tools.extend(payload["tools"])
    return tools


def v1_tools(catalog: dict, names: set[str]) -> list[dict]:
    return [t for t in catalog["tools"] if t["name"] in names]


def system_prompt(setup: str, catalog: dict) -> str:
    names = sorted(v1_names(catalog))
    if setup in ("slm_no_schema", "slm_gguf"):
        listed = ", ".join(names)
        return f"{JSON_ONLY} name must be one of: {listed}."

    if setup == "slm_v1_schema":
        tools = [compact_tool(t) for t in v1_tools(catalog, set(names))]
        post = False
    elif setup == "slm_full_schema":
        tools = [compact_tool(t) for t in load_dump_tools()]
        post = False
    elif setup == "slm_full_schema_post":
        tools = [compact_tool(t) for t in load_dump_tools()]
        post = True
    else:
        raise SystemExit(f"unknown setup {setup!r}")

    blob = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
    instr = f"{JSON_ONLY} name must be one of the tools above." if post else (
        f"{JSON_ONLY} name must be one of the tools below."
    )
    if post:
        return f"Tools:\n{blob}\n\n{instr}"
    return f"{instr}\n\nTools:\n{blob}"


def ping(host: str, model: str) -> None:
    url = host.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"cannot reach Ollama at {host} ({exc}). Is it running?"
        ) from exc
    found = [t.get("name") or t.get("model") for t in tags.get("models", [])]
    if model not in found and not any(n.startswith(model) for n in found if n):
        raise SystemExit(
            f"model {model!r} not in `ollama list` "
            f"({', '.join(n for n in found if n) or 'none'}). "
            f"Try: ollama pull {model}"
        )


def ollama_chat(
    host: str,
    model: str,
    system: str,
    user: str,
    timeout: float,
) -> tuple[str, int]:
    url = host.rstrip("/") + "/api/chat"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_predict": 256,
            "num_ctx": 32768,
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    msg = payload.get("message") or {}
    tokens = int(payload.get("prompt_eval_count") or 0)
    return str(msg.get("content") or ""), tokens


def already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["id"] for _, row in load_jsonl(path) if row.get("id")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", required=True, choices=SETUPS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--gold", type=Path, default=CANONICAL)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="max new calls this run (0 = all)")
    parser.add_argument("--timeout", type=float, default=300, help="seconds per request")
    args = parser.parse_args()

    catalog = load_json(CATALOG)
    gold = [
        row
        for _, row in load_jsonl(args.gold)
        if args.split == "all" or row.get("split") == args.split
    ]
    if not gold:
        raise SystemExit(f"no gold rows for split={args.split!r}")

    out = args.out or (EVAL_DIR / f"{args.setup}.jsonl")
    done = already_done(out)
    todo = [row for row in gold if row["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print(f"nothing to do ({len(done)} already in {out})")
        return 0

    system = system_prompt(args.setup, catalog)
    ping(args.host, args.model)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"{args.setup}  model={args.model}  "
        f"todo={len(todo)}  skipped={len(done)}  -> {out}"
    )

    with out.open("a", encoding="utf-8") as fh:
        for i, row in enumerate(todo, start=1):
            t0 = time.perf_counter()
            raw, tokens = ollama_chat(
                args.host,
                args.model,
                system,
                row["intent"],
                args.timeout,
            )
            rec = {
                "id": row["id"],
                "setup": args.setup,
                "raw": raw,
                "input_tokens": tokens,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(
                f"  {i}/{len(todo)} {row['id']}  "
                f"tokens={tokens}  {rec['latency_ms']}ms"
            )

    print(f"wrote {len(todo)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
