"""intent → validated {name, arguments, server}."""

from __future__ import annotations

import time

from ml_router.catalog import v1_names, validate_schema
from ml_router.parse import parse_call


def prune_args(arguments: dict, schema: dict) -> dict:
    props = schema.get("properties") or {}
    if not props:
        return arguments
    return {k: v for k, v in arguments.items() if k in props}


def route(
    generate_fn,
    system: str,
    catalog: dict,
    intent: str,
    max_new_tokens: int,
) -> dict:
    v1 = v1_names(catalog)
    schema_map = {t["name"]: t.get("inputSchema") or {} for t in catalog["tools"]}
    server_map = {t["name"]: t["server"] for t in catalog["tools"]}

    t0 = time.perf_counter()
    raw, n_in = generate_fn(system, intent, max_new_tokens)
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
