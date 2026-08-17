"""Sample schema-valid {tool, arguments} from catalog + slot banks.

MCPTune-style step 2: args first, no intents. Does not call MCP.
Skips the artifact trio (dropped from v1 — no train artifact paths).
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "mcp" / "catalog.v1.json"
BANKS = ROOT / "data" / "sft" / "slot_banks.json"
OUT = ROOT / "data" / "sft" / "arg_sets.jsonl"

RUN_ID = re.compile(r"^[a-z0-9]{8}$")
PAPER_ID = re.compile(r"^(\d{4}\.\d{4,5}|[a-z-]+/\d{7})$")

ARTIFACT_SKIP = {
    "list_artifact_versions_tool",
    "get_artifact_details_tool",
    "compare_artifact_versions_tool",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tools_by_name(catalog: dict) -> dict[str, dict]:
    return {t["name"]: t for t in catalog["tools"]}


def v1_tools(catalog: dict) -> list[dict]:
    return [
        t
        for t in catalog["tools"]
        if t.get("in_v1") and t["name"] not in ARTIFACT_SKIP
    ]


def contexts(banks: dict, split: str | None = None) -> list[dict]:
    rows = banks["wandb"]["contexts"]
    if split:
        return [c for c in rows if c["split"] == split]
    return rows


def lossish(keys: list[str]) -> list[str]:
    return [k for k in keys if "loss" in k.lower()]


def validate(schema: dict, args: dict) -> None:
    if any(v is None for v in args.values()):
        raise ValueError("null values are omitted, not emitted")
    required = schema.get("required") or []
    missing = [k for k in required if k not in args]
    if missing:
        raise ValueError(f"missing required {missing}")
    props = schema.get("properties") or {}
    extra = [k for k in args if k not in props]
    if extra and schema.get("additionalProperties") is False:
        raise ValueError(f"extra keys {extra}")
    for key, value in args.items():
        spec = props.get(key)
        if not spec:
            continue
        expected = spec.get("type")
        if expected == "string" and not isinstance(value, str):
            raise ValueError(f"{key} should be string")
        if expected == "integer" and not isinstance(value, int):
            raise ValueError(f"{key} should be integer")
        if expected == "number" and not isinstance(value, (int, float)):
            raise ValueError(f"{key} should be number")
        if expected == "boolean" and not isinstance(value, bool):
            raise ValueError(f"{key} should be boolean")
        if expected == "array" and not isinstance(value, list):
            raise ValueError(f"{key} should be array")
        enum = spec.get("enum")
        if enum and value not in enum:
            raise ValueError(f"{key}={value!r} not in {enum}")
        items = spec.get("items") or {}
        item_enum = items.get("enum")
        if item_enum and isinstance(value, list):
            bad = [v for v in value if v not in item_enum]
            if bad:
                raise ValueError(f"{key} items {bad} not in {item_enum}")


def emit(rows: list[dict], tool: str, server: str, split: str, arguments: dict) -> None:
    rows.append(
        {
            "tool": tool,
            "arguments": arguments,
            "server": server,
            "split": split,
        }
    )


def sample_list_entities(rows: list[dict]) -> None:
    emit(rows, "list_entities_tool", "wandb", "train", {})


def sample_entity_projects(rows: list[dict], banks: dict) -> None:
    emit(rows, "query_wandb_entity_projects", "wandb", "train", {})
    for ctx in contexts(banks):
        emit(
            rows,
            "query_wandb_entity_projects",
            "wandb",
            ctx["split"],
            {"entity": ctx["entity"]},
        )
        emit(
            rows,
            "query_wandb_entity_projects",
            "wandb",
            ctx["split"],
            {"entity": ctx["entity"], "max_projects": 20},
        )


def sample_probe(rows: list[dict], banks: dict) -> None:
    for ctx in contexts(banks):
        base = {"entity_name": ctx["entity"], "project_name": ctx["project"]}
        emit(rows, "probe_project_tool", "wandb", ctx["split"], dict(base))
        for n in (3, 10):
            emit(
                rows,
                "probe_project_tool",
                "wandb",
                ctx["split"],
                {**base, "sample_runs": n},
            )


def sample_history(rows: list[dict], banks: dict) -> None:
    for ctx in contexts(banks):
        keys = ctx.get("metric_keys") or []
        for run in ctx["runs"]:
            base = {
                "entity_name": ctx["entity"],
                "project_name": ctx["project"],
                "run_id": run["id"],
            }
            emit(rows, "get_run_history_tool", "wandb", ctx["split"], dict(base))
            if keys:
                emit(
                    rows,
                    "get_run_history_tool",
                    "wandb",
                    ctx["split"],
                    {**base, "keys": [keys[0]]},
                )
            if len(keys) >= 2:
                emit(
                    rows,
                    "get_run_history_tool",
                    "wandb",
                    ctx["split"],
                    {**base, "keys": keys[:2]},
                )
            emit(
                rows,
                "get_run_history_tool",
                "wandb",
                ctx["split"],
                {**base, "samples": 100},
            )


def sample_compare(rows: list[dict], banks: dict) -> None:
    for ctx in contexts(banks):
        ids = [r["id"] for r in ctx["runs"]]
        keys = ctx.get("metric_keys") or []
        for i, (a, b) in enumerate(itertools.combinations(ids, 2)):
            base = {
                "entity_name": ctx["entity"],
                "project_name": ctx["project"],
                "run_id_a": a,
                "run_id_b": b,
            }
            emit(rows, "compare_runs_tool", "wandb", ctx["split"], dict(base))
            if i % 3 == 0:
                emit(
                    rows,
                    "compare_runs_tool",
                    "wandb",
                    ctx["split"],
                    {**base, "include_history_overlap": True},
                )
            if i % 3 == 1 and keys:
                emit(
                    rows,
                    "compare_runs_tool",
                    "wandb",
                    ctx["split"],
                    {**base, "history_keys": keys[:2]},
                )


def sample_diagnose(rows: list[dict], banks: dict) -> None:
    for ctx in contexts(banks):
        losses = lossish(ctx.get("metric_keys") or [])
        for run in ctx["runs"]:
            base = {
                "entity_name": ctx["entity"],
                "project_name": ctx["project"],
                "run_id": run["id"],
            }
            emit(rows, "diagnose_run_tool", "wandb", ctx["split"], dict(base))
            if losses:
                extra = {**base, "loss_key": losses[0]}
                emit(rows, "diagnose_run_tool", "wandb", ctx["split"], extra)
                if len(losses) >= 2:
                    emit(
                        rows,
                        "diagnose_run_tool",
                        "wandb",
                        ctx["split"],
                        {**extra, "val_loss_key": losses[1]},
                    )


def sample_docs(rows: list[dict], banks: dict) -> None:
    for item in banks["wandb_docs"]["queries"]:
        emit(
            rows,
            "search_wandb_docs_tool",
            "wandb",
            item["split"],
            {"query": item["query"]},
        )


def sample_hub_search(rows: list[dict], banks: dict) -> None:
    sorts = ["downloads", "likes", "trendingScore"]
    for i, item in enumerate(banks["huggingface"]["queries"]):
        types = [t for t in item["repo_types"] if t in ("model", "dataset")]
        if not types:
            continue
        base = {"query": item["query"], "repo_types": types}
        emit(rows, "hub_repo_search", "huggingface", item["split"], dict(base))
        emit(
            rows,
            "hub_repo_search",
            "huggingface",
            item["split"],
            {**base, "sort": sorts[i % len(sorts)]},
        )
        if i % 2 == 0:
            emit(
                rows,
                "hub_repo_search",
                "huggingface",
                item["split"],
                {**base, "limit": 10},
            )


def sample_hub_details(rows: list[dict], banks: dict) -> None:
    for item in banks["huggingface"]["repo_ids"]:
        base = {"repo_ids": [item["id"]]}
        emit(rows, "hub_repo_details", "huggingface", item["split"], dict(base))
        repo_type = item.get("repo_type")
        if repo_type in ("model", "dataset"):
            emit(
                rows,
                "hub_repo_details",
                "huggingface",
                item["split"],
                {**base, "repo_type": repo_type},
            )


def sample_papers(rows: list[dict], banks: dict) -> None:
    for i, item in enumerate(banks["arxiv"]["queries"]):
        base = {"query": item["query"]}
        emit(rows, "search_papers", "arxiv", item["split"], dict(base))
        cats = item.get("categories") or []
        if cats:
            emit(
                rows,
                "search_papers",
                "arxiv",
                item["split"],
                {**base, "categories": cats},
            )
        if i % 2 == 0:
            extra = dict(base)
            if cats:
                extra["categories"] = cats
            extra["sort_by"] = "date"
            emit(rows, "search_papers", "arxiv", item["split"], extra)


def sample_abstracts(rows: list[dict], banks: dict) -> None:
    for item in banks["arxiv"]["paper_ids"]:
        emit(
            rows,
            "get_abstract",
            "arxiv",
            item["split"],
            {"paper_id": item["id"]},
        )


SAMPLERS = {
    "list_entities_tool": sample_list_entities,
    "query_wandb_entity_projects": sample_entity_projects,
    "probe_project_tool": sample_probe,
    "get_run_history_tool": sample_history,
    "compare_runs_tool": sample_compare,
    "diagnose_run_tool": sample_diagnose,
    "search_wandb_docs_tool": sample_docs,
    "hub_repo_search": sample_hub_search,
    "hub_repo_details": sample_hub_details,
    "search_papers": sample_papers,
    "get_abstract": sample_abstracts,
}


def extra_checks(row: dict) -> None:
    args = row["arguments"]
    for key in ("run_id", "run_id_a", "run_id_b"):
        if key in args and not RUN_ID.match(args[key]):
            raise ValueError(f"{key}={args[key]!r} is not an 8-char run id")
    if "run_id_a" in args and args["run_id_a"] == args.get("run_id_b"):
        raise ValueError("compare_runs_tool needs two different run ids")
    if "paper_id" in args and not PAPER_ID.match(args["paper_id"]):
        raise ValueError(f"paper_id={args['paper_id']!r} looks like a URL or bad id")
    if "repo_ids" in args:
        ids = args["repo_ids"]
        if not ids or any("/" not in x for x in ids):
            raise ValueError(f"repo_ids must be non-empty org/name, got {ids}")


def dedup(rows: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out = []
    for row in rows:
        key = (
            row["tool"],
            row["split"],
            json.dumps(row["arguments"], sort_keys=True, ensure_ascii=False),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def cap_per_tool(rows: list[dict], limit: int, rng: random.Random) -> list[dict]:
    by_tool: dict[str, list[dict]] = {}
    for row in rows:
        by_tool.setdefault(row["tool"], []).append(row)
    out = []
    for tool, group in by_tool.items():
        if len(group) <= limit:
            out.extend(group)
            continue
        train = [r for r in group if r["split"] == "train"]
        hold = [r for r in group if r["split"] != "train"]
        keep_hold = min(len(hold), max(8, limit // 5))
        keep_train = limit - keep_hold
        if len(train) > keep_train:
            train = rng.sample(train, keep_train)
        if len(hold) > keep_hold:
            hold = rng.sample(hold, keep_hold)
        out.extend(train + hold)
    return out


def stamp_ids(rows: list[dict]) -> list[dict]:
    order = list(SAMPLERS)
    split_rank = {"train": 0, "holdout": 1}
    rows.sort(
        key=lambda r: (
            order.index(r["tool"]) if r["tool"] in order else 99,
            split_rank.get(r["split"], 9),
            json.dumps(r["arguments"], sort_keys=True),
        )
    )
    stamped = []
    for i, row in enumerate(rows, start=1):
        stamped.append({"id": f"arg-{i:06d}", **row})
    return stamped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--per-tool", type=int, default=110)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    catalog = load_json(CATALOG)
    banks = load_json(BANKS)
    by_name = tools_by_name(catalog)
    rng = random.Random(args.seed)

    rows: list[dict] = []
    for tool in v1_tools(catalog):
        name = tool["name"]
        sampler = SAMPLERS.get(name)
        if sampler is None:
            raise SystemExit(f"no sampler for v1 tool {name}")
        if name == "list_entities_tool":
            sampler(rows)
        else:
            sampler(rows, banks)

    rows = dedup(rows)
    rows = cap_per_tool(rows, args.per_tool, rng)

    for row in rows:
        schema = by_name[row["tool"]]["inputSchema"]
        validate(schema, row["arguments"])
        extra_checks(row)

    rows = stamp_ids(rows)

    counts = Counter((r["tool"], r["split"]) for r in rows)
    print(f"{len(rows)} arg sets  ({len({r['tool'] for r in rows})} tools)")
    for tool in SAMPLERS:
        train = counts.get((tool, "train"), 0)
        hold = counts.get((tool, "holdout"), 0)
        print(f"  {tool:32} train={train:3}  holdout={hold:3}  total={train + hold:3}")

    if args.dry_run:
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
