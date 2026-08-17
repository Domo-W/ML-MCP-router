"""Smoke the PEFT adapter: intent -> parse -> schema check.

Load once. Does not call W&B / HF / arXiv.

  python scripts/route_once.py --smoke
  python scripts/route_once.py "Is run k7m2n9p4 in train-lab/rl-benchmark overfitting?"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_adapter import DEFAULT_ADAPTER, DEFAULT_MODEL, generate, load_model  # noqa: E402
from eval_calls import parse_call  # noqa: E402
from eval_ollama import system_prompt  # noqa: E402
from validate_canonical import CATALOG, load_json, v1_names, validate_schema  # noqa: E402

SMOKE = [
    "Is run k7m2n9p4 in train-lab/rl-benchmark overfitting?",
    "Show me the loss curve for k7m2n9p4 in train-lab/rl-benchmark",
    "Best Qwen 3B instruct GGUF",
    "Abstract of 2106.09685",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("intent", nargs="?", help="one user intent")
    parser.add_argument("--smoke", action="store_true", help="run the 4 built-in prompts")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def prune_args(arguments: dict, schema: dict) -> dict:
    props = schema.get("properties") or {}
    if not props:
        return arguments
    return {k: v for k, v in arguments.items() if k in props}


def route(
    tokenizer,
    model,
    system: str,
    catalog: dict,
    intent: str,
    max_new_tokens: int,
) -> dict:
    v1 = v1_names(catalog)
    schema_map = {t["name"]: t.get("inputSchema") or {} for t in catalog["tools"]}
    server_map = {t["name"]: t["server"] for t in catalog["tools"]}

    t0 = time.perf_counter()
    raw, n_in = generate(tokenizer, model, system, intent, max_new_tokens)
    ms = int((time.perf_counter() - t0) * 1000)

    pred = parse_call(raw)
    if pred is None:
        return {
            "ok": False,
            "error": "parse",
            "raw": raw,
            "input_tokens": n_in,
            "latency_ms": ms,
        }

    name = pred["name"]
    if name not in v1:
        return {
            "ok": False,
            "error": f"unknown tool {name!r}",
            "pred": pred,
            "input_tokens": n_in,
            "latency_ms": ms,
        }

    schema = schema_map[name]
    arguments = prune_args(pred["arguments"], schema)
    schema_err = validate_schema(schema, arguments)
    if schema_err:
        return {
            "ok": False,
            "error": schema_err,
            "pred": {"name": name, "arguments": arguments},
            "input_tokens": n_in,
            "latency_ms": ms,
        }

    return {
        "ok": True,
        "name": name,
        "arguments": arguments,
        "server": server_map[name],
        "input_tokens": n_in,
        "latency_ms": ms,
        "raw": raw,
    }


def main() -> int:
    args = parse_args()
    intents = SMOKE if args.smoke else ([args.intent] if args.intent else [])
    if not intents:
        raise SystemExit("pass an intent or --smoke")

    catalog = load_json(CATALOG)
    system = system_prompt("slm_no_schema", catalog)
    print(f"loading {args.model} + {args.adapter} ...", flush=True)
    tokenizer, model = load_model(args.model, args.adapter)

    failed = 0
    for intent in intents:
        rec = route(tokenizer, model, system, catalog, intent, args.max_new_tokens)
        print(json.dumps({"intent": intent, **rec}, ensure_ascii=False, indent=2))
        if not rec["ok"]:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
