"""Append hard-negative collision rows to canonical.jsonl.

MCPTune-style step 4: tool chosen first. Templates only — no Ollama.
Does not overwrite existing easy rows. Preview never writes.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANKS = ROOT / "data" / "sft" / "slot_banks.json"
CATALOG = ROOT / "data" / "mcp" / "catalog.v1.json"
OUT = ROOT / "data" / "sft" / "canonical.jsonl"

SPACE_RE = re.compile(r"\s+")
RUN_ID = re.compile(r"^[a-z0-9]{8}$")
PAPER_ID = re.compile(r"^(\d{4}\.\d{4,5}|[a-z-]+/\d{7})$")

# Short names for bait only. Gold is still paper_id. Do not invent titles.
PAPER_NAMES = {
    "2106.09685": "LoRA",
    "2305.18290": "DPO",
    "2305.14314": "QLoRA",
    "2203.02155": "InstructGPT",
    "2210.11416": "FLAN",
    "2307.09288": "Llama 2",
    "2402.03300": "DeepSeekMath",
    "2402.01306": "WARM",
    "1706.03762": "Transformer",
    "2005.14165": "GPT-3",
    "2212.09796": "reward model overoptimization",
    "2302.13971": "LLaMA",
    "2302.04761": "Toolformer",
    "2210.03629": "ReAct",
    "2310.06825": "Mistral",
    "2401.14196": "DeepSeek-Coder",
    "2405.14734": "SimPO",
    "2401.10774": "Medusa",
    "2211.17192": "speculative decoding",
    "2306.05685": "LLM-as-a-judge",
}

SERVER = {
    "hub_repo_search": "huggingface",
    "hub_repo_details": "huggingface",
    "search_papers": "arxiv",
    "get_abstract": "arxiv",
    "probe_project_tool": "wandb",
    "diagnose_run_tool": "wandb",
    "get_run_history_tool": "wandb",
    "search_wandb_docs_tool": "wandb",
}

# 4 train + 4 holdout frames per type. Cycle i % 4. Do not name the server.
# Bait the wrong tool; one cue decides gold. Do not say "not X".
FRAMES = {
    "hub_vs_wandb": {
        "tool": "hub_repo_search",
        "train": [
            "Best {query}",
            "Top {query} to download",
            "Which {query} should I download?",
            "{query} with the most likes",
        ],
        "holdout": [
            "Top {query} right now",
            "Trending {query}",
            "What's the strongest {query}?",
            "Highest-rated {query}",
        ],
    },
    "probe_vs_hub": {
        "tool": "probe_project_tool",
        "train": [
            "Best run in {entity}/{project}",
            "Which run won in {entity}/{project}?",
            "Top performing run in {entity}/{project}",
            "What's the leader in {entity}/{project}?",
        ],
        "holdout": [
            "Which run is winning in {entity}/{project}?",
            "Who's ahead in {entity}/{project}?",
            "Rank the runs in {entity}/{project}",
            "What's the strongest run in {entity}/{project}?",
        ],
    },
    "paper_vs_hub": {
        "tool": "search_papers",
        "train": [
            "Best {query} paper",
            "Find the {query} paper",
            "Original {query} paper",
            "the original {query} writeup",
        ],
        "holdout": [
            "{query} paper weights",
            "{query} paper checkpoint",
            "Looking for the {query} paper",
            "{query} from the literature",
        ],
    },
    "diagnose_vs_history": {
        "tool": "diagnose_run_tool",
        "train": [
            "Is run {run_id} in {entity}/{project} overfitting?",
            "Is {run_id} in {entity}/{project} diverging?",
            "Did {run_id} in {entity}/{project} blow up?",
            "Health check on {run_id} in {entity}/{project}",
        ],
        "holdout": [
            "Is {run_id} in {entity}/{project} healthy?",
            "Has {run_id} in {entity}/{project} converged?",
            "Any NaNs in {run_id} in {entity}/{project}?",
            "Is training okay for {run_id} in {entity}/{project}?",
        ],
    },
    "history_vs_diagnose": {
        "tool": "get_run_history_tool",
        "train": [
            "Show the loss curve for {run_id} in {entity}/{project}",
            "Plot loss for {run_id} in {entity}/{project}",
            "Training curve of {run_id} in {entity}/{project}",
            "History of {run_id} in {entity}/{project}",
        ],
        "holdout": [
            "Plot metrics over time for {run_id} in {entity}/{project}",
            "Step-by-step metrics for {run_id} in {entity}/{project}",
            "How did {run_id} in {entity}/{project} move over steps?",
            "Time series for {run_id} in {entity}/{project}",
        ],
    },
    "docs_vs_probe": {
        "tool": "search_wandb_docs_tool",
        "train": [
            "{how}",
            "SDK: {query}",
            "How do I do this: {query}",
            "wandb how-to: {query}",
        ],
        "holdout": [
            "Teach me {query}",
            "Can you walk me through {query}?",
            "Walk me through {query}",
            "{how}",
        ],
    },
    "abstract_vs_search": {
        "tool": "get_abstract",
        "train": [
            "Abstract of the {name} paper {paper_id}",
            "Get the abstract for {name} {paper_id}",
            "{name} ({paper_id}) abstract",
            "Pull abstract {paper_id} ({name})",
        ],
        "holdout": [
            "What's the abstract for {name} ({paper_id})?",
            "Summary of {name} {paper_id}",
            "Abstract only: {name} {paper_id}",
            "Metadata for {paper_id} ({name})",
        ],
    },
    "details_vs_search": {
        "tool": "hub_repo_details",
        "train": [
            "What's in {repo_id}?",
            "Files in {repo_id}",
            "Inspect {repo_id}",
            "Overview of {repo_id}",
        ],
        "holdout": [
            "Show the README for {repo_id}",
            "Repo page for {repo_id}",
            "What's in the {repo_id} card?",
            "Structure of {repo_id}",
        ],
    },
}


def norm(text: str) -> str:
    return SPACE_RE.sub(" ", text.casefold()).strip()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def next_ft_id(rows: list[dict]) -> int:
    best = 0
    for row in rows:
        mid = re.fullmatch(r"ft-(\d+)", str(row.get("id", "")))
        if mid:
            best = max(best, int(mid.group(1)))
    return best + 1


def how_docs(query: str) -> str:
    q = query.strip()
    low = q.lower()
    if low.startswith(("how ", "what ", "where ", "why ")):
        text = q if q.endswith("?") else f"{q}?"
        return text[0].upper() + text[1:]
    return f"How do I {q}?"


def licensed(intent: str, arguments: dict) -> str | None:
    """Return a reason if a gold slot is missing from the intent."""
    hay = intent.casefold()
    for key, value in arguments.items():
        if key == "repo_types":
            continue
        if isinstance(value, str) and value.casefold() not in hay:
            return f"missing {key}={value!r}"
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.casefold() not in hay:
                    return f"missing {key} item {item!r}"
    return None


def extra_checks(tool: str, arguments: dict) -> None:
    if tool == "hub_repo_search":
        if arguments.get("repo_types") != ["model"]:
            raise ValueError("hub_vs_wandb must set repo_types=['model']")
        if not arguments.get("query"):
            raise ValueError("hub_repo_search needs query")
    if tool == "hub_repo_details":
        ids = arguments.get("repo_ids") or []
        if not ids or not all("/" in i for i in ids):
            raise ValueError("repo_ids must be org/name")
    if tool == "get_abstract":
        pid = arguments.get("paper_id", "")
        if not PAPER_ID.match(pid):
            raise ValueError(f"bad paper_id {pid!r}")
    for key in ("run_id", "run_id_a", "run_id_b"):
        if key in arguments and not RUN_ID.match(arguments[key]):
            raise ValueError(f"bad {key} {arguments[key]!r}")


def validate_schema(schema: dict, arguments: dict) -> None:
    required = schema.get("required") or []
    missing = [k for k in required if k not in arguments]
    if missing:
        raise ValueError(f"missing required {missing}")
    props = schema.get("properties") or {}
    extra = [k for k in arguments if k not in props]
    if extra and schema.get("additionalProperties") is False:
        raise ValueError(f"extra keys {extra}")


def by_split(items: list[dict], split: str) -> list[dict]:
    return [x for x in items if x.get("split") == split]


def wandb_runs(banks: dict, split: str) -> list[dict]:
    out = []
    for ctx in by_split(banks["wandb"]["contexts"], split):
        for run in ctx["runs"]:
            out.append(
                {
                    "entity": ctx["entity"],
                    "project": ctx["project"],
                    "run_id": run["id"],
                    "name": run["name"],
                }
            )
    return out


def hf_model_queries(banks: dict, split: str) -> list[dict]:
    return [
        q
        for q in by_split(banks["huggingface"]["queries"], split)
        if q.get("repo_types") == ["model"]
    ]


def take(items: list, n: int) -> list:
    return items[:n]


def slots_for(collision: str, banks: dict, split: str, n: int) -> list[dict]:
    """Each item is {arguments, fill} for one row."""
    if collision == "hub_vs_wandb":
        return [
            {
                "arguments": {"query": q["query"], "repo_types": ["model"]},
                "fill": {"query": q["query"]},
            }
            for q in take(hf_model_queries(banks, split), n)
        ]
    if collision == "probe_vs_hub":
        ctxs = by_split(banks["wandb"]["contexts"], split)
        return [
            {
                "arguments": {
                    "entity_name": c["entity"],
                    "project_name": c["project"],
                },
                "fill": {"entity": c["entity"], "project": c["project"]},
            }
            for c in take(ctxs, n)
        ]
    if collision == "paper_vs_hub":
        return [
            {
                "arguments": {"query": q["query"]},
                "fill": {"query": q["query"]},
            }
            for q in take(by_split(banks["arxiv"]["queries"], split), n)
        ]
    if collision in ("diagnose_vs_history", "history_vs_diagnose"):
        rows = []
        for run in take(wandb_runs(banks, split), n):
            args = {
                "entity_name": run["entity"],
                "project_name": run["project"],
                "run_id": run["run_id"],
            }
            rows.append(
                {
                    "arguments": args,
                    "fill": {
                        "entity": run["entity"],
                        "project": run["project"],
                        "run_id": run["run_id"],
                    },
                }
            )
        return rows
    if collision == "docs_vs_probe":
        rows = []
        for q in take(by_split(banks["wandb_docs"]["queries"], split), n):
            query = q["query"]
            rows.append(
                {
                    "arguments": {"query": query},
                    "fill": {"query": query, "how": how_docs(query)},
                }
            )
        return rows
    if collision == "abstract_vs_search":
        rows = []
        for p in take(by_split(banks["arxiv"]["paper_ids"], split), n):
            name = PAPER_NAMES.get(p["id"])
            if not name:
                raise KeyError(f"no short name for paper_id {p['id']}")
            rows.append(
                {
                    "arguments": {"paper_id": p["id"]},
                    "fill": {"paper_id": p["id"], "name": name},
                }
            )
        return rows
    if collision == "details_vs_search":
        return [
            {
                "arguments": {"repo_ids": [r["id"]]},
                "fill": {"repo_id": r["id"]},
            }
            for r in take(by_split(banks["huggingface"]["repo_ids"], split), n)
        ]
    raise KeyError(collision)


def build_row(
    collision: str,
    split: str,
    slot: dict,
    frame: str,
    schemas: dict[str, dict],
    row_id: str,
) -> dict:
    spec = FRAMES[collision]
    tool = spec["tool"]
    intent = frame.format(**slot["fill"])
    arguments = slot["arguments"]
    reason = licensed(intent, arguments)
    if reason:
        raise ValueError(f"{collision} {split}: {reason} | {intent}")
    extra_checks(tool, arguments)
    validate_schema(schemas[tool], arguments)
    return {
        "id": row_id,
        "split": split,
        "intent": intent,
        "tool": tool,
        "arguments": arguments,
        "server": SERVER[tool],
        "collision": collision,
        "source": "template",
    }


def generate(
    banks: dict,
    schemas: dict[str, dict],
    per_split: int,
    id_start: int,
    prefix: str = "ft-",
) -> list[dict]:
    rows: list[dict] = []
    seen = set()
    n = id_start
    for collision, spec in FRAMES.items():
        for split in ("train", "holdout"):
            frames = spec[split]
            available = slots_for(collision, banks, split, per_split)
            if not available:
                continue
            for i in range(per_split):
                slot = available[i % len(available)]
                frame = frames[i % len(frames)]
                row_id = f"{prefix}{n:06d}"
                row = build_row(collision, split, slot, frame, schemas, row_id)
                key = norm(row["intent"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
                n += 1
    return rows


def preview_examples(per_split: int = 2) -> list[dict]:
    """Build a few rows per type. Never writes canonical.jsonl."""
    banks = load_json(BANKS)
    catalog = load_json(CATALOG)
    schemas = {t["name"]: t["inputSchema"] for t in catalog["tools"] if t.get("in_v1")}
    return generate(banks, schemas, per_split, id_start=1, prefix="ft-preview-")


def print_preview(rows: list[dict]) -> None:
    by_type: dict[str, list[dict]] = {}
    for row in rows:
        by_type.setdefault(row["collision"], []).append(row)
    for collision in FRAMES:
        print(f"\n=== {collision}  gold={FRAMES[collision]['tool']} ===")
        for row in by_type.get(collision, []):
            print(json.dumps(row, ensure_ascii=False, indent=2))
    print(f"\n{len(rows)} preview rows (not written)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="print a few examples per type; do not write canonical.jsonl",
    )
    parser.add_argument(
        "--preview-n",
        type=int,
        default=2,
        help="preview rows per type per split (default 2 train + 2 holdout)",
    )
    parser.add_argument("--per-split", type=int, default=5, help="rows per type per split when writing")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    if args.preview:
        print_preview(preview_examples(args.preview_n))
        return

    banks = load_json(BANKS)
    catalog = load_json(CATALOG)
    schemas = {t["name"]: t["inputSchema"] for t in catalog["tools"] if t.get("in_v1")}
    existing = load_jsonl(args.out)
    seen = {norm(r["intent"]) for r in existing if r.get("intent")}
    start = next_ft_id(existing)
    rows = generate(banks, schemas, args.per_split, start)
    kept = []
    for row in rows:
        if norm(row["intent"]) in seen:
            continue
        seen.add(norm(row["intent"]))
        kept.append(row)

    counts = Counter((r["collision"], r["split"]) for r in kept)
    print(f"{len(kept)} collision rows -> {args.out}")
    for collision in FRAMES:
        train = counts.get((collision, "train"), 0)
        hold = counts.get((collision, "holdout"), 0)
        print(f"  {collision:24} train={train}  holdout={hold}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
