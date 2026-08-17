"""Score predicted tool calls against canonical gold.

Model-blind. Does not call Ollama. Valid = parseable {name, arguments}
for an in_v1 tool whose args pass that tool's inputSchema.
correct_args checks gold required/slot keys only; extra optionals are
ignored. exact_match is arguments == gold.arguments.

  python scripts/eval_calls.py --self-check
  python scripts/eval_calls.py --pred data/eval/slm_no_schema.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_canonical import (  # noqa: E402
    CANONICAL,
    CATALOG,
    load_json,
    load_jsonl,
    v1_names,
    validate_schema,
)

THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
FENCE_RE = re.compile(r"^```(?:\w+)?\s*|\s*```$", re.S)

SLOT_KEYS = {
    "entity",
    "entity_name",
    "project_name",
    "run_id",
    "run_id_a",
    "run_id_b",
    "query",
    "repo_types",
    "repo_ids",
    "paper_id",
}

FLAG_KEYS = ("valid", "correct_tool", "correct_args", "exact_match")


def parse_call(raw: str) -> dict | None:
    text = THINK_RE.sub("", raw or "")
    text = FENCE_RE.sub("", text.strip()).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or set(obj) != {"name", "arguments"}:
        return None
    if not isinstance(obj["name"], str) or not isinstance(obj["arguments"], dict):
        return None
    return obj


def slot_keys(gold: dict, schema_map: dict) -> list[str]:
    schema = schema_map.get(gold["tool"]) or {}
    required = set(schema.get("required") or [])
    return [
        key
        for key in gold["arguments"]
        if key in required or key in SLOT_KEYS
    ]


def required_slots_match(pred_args: dict, gold: dict, schema_map: dict) -> bool:
    gold_args = gold["arguments"]
    return all(pred_args.get(key) == gold_args[key] for key in slot_keys(gold, schema_map))


def score_raw(
    raw: str,
    gold: dict,
    *,
    v1: set[str],
    schema_map: dict,
) -> dict:
    pred = parse_call(raw)
    valid = False
    if pred is not None and pred["name"] in v1:
        valid = validate_schema(schema_map[pred["name"]], pred["arguments"]) is None
    return {
        "pred": pred,
        "valid": valid,
        "correct_tool": bool(valid and pred["name"] == gold["tool"]),
        "correct_args": bool(valid and required_slots_match(pred["arguments"], gold, schema_map)),
        "exact_match": bool(valid and pred["arguments"] == gold["arguments"]),
    }


def gold_raw(row: dict) -> str:
    return json.dumps({"name": row["tool"], "arguments": row["arguments"]}, ensure_ascii=False)


def load_gold(path: Path, split: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for _, row in load_jsonl(path):
        if split != "all" and row.get("split") != split:
            continue
        rid = row["id"]
        if rid in out:
            raise SystemExit(f"duplicate gold id {rid}")
        out[rid] = row
    if not out:
        raise SystemExit(f"no gold rows for split={split!r} in {path}")
    return out


def score_records(
    records: list[dict],
    gold_by_id: dict[str, dict],
    *,
    v1: set[str],
    schema_map: dict,
) -> list[dict]:
    scored: list[dict] = []
    for rec in records:
        rid = rec.get("id")
        gold = gold_by_id.get(rid)
        if gold is None:
            raise SystemExit(f"pred id {rid!r} not in gold split")
        raw = rec.get("raw")
        if raw is None and rec.get("pred") is not None:
            raw = json.dumps(rec["pred"], ensure_ascii=False)
        flags = score_raw(str(raw or ""), gold, v1=v1, schema_map=schema_map)
        row = {
            "id": rid,
            "setup": rec.get("setup") or "unknown",
            "raw": raw or "",
            "pred": flags["pred"],
            "valid": flags["valid"],
            "correct_tool": flags["correct_tool"],
            "correct_args": flags["correct_args"],
            "exact_match": flags["exact_match"],
        }
        if "input_tokens" in rec:
            row["input_tokens"] = rec["input_tokens"]
        if "latency_ms" in rec:
            row["latency_ms"] = rec["latency_ms"]
        scored.append(row)
    return scored


def _pct(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return 100.0 * sum(1 for r in rows if r[key]) / len(rows)


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return mean(vals) if vals else None


def format_block(label: str, rows: list[dict]) -> str:
    parts = [
        f"{label:22} n={len(rows):<4}",
        f"valid={_pct(rows, 'valid'):5.1f}%",
        f"correct_tool={_pct(rows, 'correct_tool'):5.1f}%",
        f"correct_args={_pct(rows, 'correct_args'):5.1f}%",
        f"exact={_pct(rows, 'exact_match'):5.1f}%",
    ]
    tokens = _mean(rows, "input_tokens")
    latency = _mean(rows, "latency_ms")
    if tokens is not None:
        parts.append(f"tokens/req={tokens:.0f}")
    if latency is not None:
        parts.append(f"latency_ms={latency:.0f}")
    return "  ".join(parts)


def print_report(scored: list[dict], gold_by_id: dict[str, dict]) -> None:
    print(format_block("overall", scored))
    if not scored:
        return

    slices: dict[str, dict[str, list[dict]]] = {
        "collision": defaultdict(list),
        "server": defaultdict(list),
        "tool": defaultdict(list),
    }
    for row in scored:
        gold = gold_by_id[row["id"]]
        kind = "collision" if gold.get("collision") else "easy"
        slices["collision"][kind].append(row)
        slices["server"][gold.get("server") or "?"].append(row)
        slices["tool"][gold["tool"]].append(row)

    for title, groups in slices.items():
        print(f"\n{title}:")
        for name in sorted(groups):
            print(format_block(name, groups[name]))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="score gold as predictions")
    parser.add_argument("--pred", type=Path, help="prediction JSONL to score")
    parser.add_argument("--gold", type=Path, default=CANONICAL)
    parser.add_argument("--split", default="test", help="gold split: test, train, or all")
    parser.add_argument("--out", type=Path, help="write scored JSONL here")
    args = parser.parse_args()

    if args.self_check == bool(args.pred):
        parser.error("exactly one of --self-check or --pred is required")

    catalog = load_json(CATALOG)
    v1 = v1_names(catalog)
    schema_map = {t["name"]: t["inputSchema"] for t in catalog["tools"]}
    gold_by_id = load_gold(args.gold, args.split)

    if args.self_check:
        records = [
            {"id": rid, "setup": "gold", "raw": gold_raw(row)}
            for rid, row in gold_by_id.items()
        ]
    else:
        records = [row for _, row in load_jsonl(args.pred)]
        if not records:
            raise SystemExit(f"no rows in {args.pred}")

    scored = score_records(records, gold_by_id, v1=v1, schema_map=schema_map)
    print_report(scored, gold_by_id)

    out = args.out
    if out is None and args.pred is not None:
        out = args.pred.with_name(args.pred.stem + ".scored.jsonl")
    if out is not None:
        write_jsonl(out, scored)
        print(f"\nwrote {len(scored)} rows -> {out}")

    if args.self_check:
        bad = [r["id"] for r in scored if not all(r[k] for k in FLAG_KEYS)]
        if bad:
            print(f"\nself-check failed: {len(bad)} gold rows not 100% ({bad[:8]})")
            return 1
        print("\nself-check passed: 100% valid / correct_tool / correct_args / exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
