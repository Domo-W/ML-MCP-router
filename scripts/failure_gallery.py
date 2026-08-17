"""Join scored preds to gold and write a QLoRA failure gallery.

Model-blind. Does not call the adapter. Uses flags already on the
scored JSONL; gold supplies intent / tool / collision.

  python scripts/failure_gallery.py
  python scripts/failure_gallery.py --pred data/eval/slm_qlora.scored.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_calls import (  # noqa: E402
    FENCE_RE,
    THINK_RE,
    load_gold,
    parse_call,
    slot_keys,
)
from validate_canonical import (  # noqa: E402
    CATALOG,
    CANONICAL,
    load_json,
    load_jsonl,
    v1_names,
    validate_schema,
)

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"
DEFAULT_PRED = EVAL_DIR / "slm_qlora.scored.jsonl"
DEFAULT_OUT = EVAL_DIR / "failure_gallery.md"

RUN_ID_KEYS = ("run_id", "run_id_a", "run_id_b")
PRIMARY_ORDER = (
    "invalid",
    "wrong_server",
    "wrong_tool",
    "bad_run_id",
    "bad_slots",
    "inexact_optional",
)
PRIMARY_TITLE = {
    "invalid": "Invalid / extra prose",
    "wrong_server": "Wrong server",
    "wrong_tool": "Wrong tool",
    "bad_run_id": "Bad run_id",
    "bad_slots": "Bad slots",
    "inexact_optional": "Inexact optionals",
}
COLLISION_HINT = {
    "hub_vs_wandb": "Hub model search vs W&B runs",
    "probe_vs_hub": "best run in a project vs Hub likes",
    "paper_vs_hub": "paper search vs Hub weights",
    "diagnose_vs_history": "health/overfit vs loss curve",
    "history_vs_diagnose": "loss curve vs health/overfit",
    "docs_vs_probe": "how-do-I docs vs project probe",
    "abstract_vs_search": "known paper id vs paper search",
    "details_vs_search": "specific repo vs Hub search",
}
INEXACT_EXAMPLES = 2


def compact(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def pred_name(row: dict) -> str | None:
    pred = row.get("pred")
    if isinstance(pred, dict):
        name = pred.get("name")
        if isinstance(name, str):
            return name
    return None


def pred_args(row: dict) -> dict:
    pred = row.get("pred")
    if isinstance(pred, dict) and isinstance(pred.get("arguments"), dict):
        return pred["arguments"]
    return {}


def prose_notes(raw: str) -> list[str]:
    notes = []
    text = raw or ""
    if THINK_RE.search(text):
        notes.append("think-block")
    if "```" in text:
        notes.append("fence")
    stripped = FENCE_RE.sub("", THINK_RE.sub("", text).strip()).strip()
    if parse_call(text) is None:
        notes.append("unparseable" if stripped else "empty")
    elif stripped != text.strip():
        notes.append("stripped-wrapper")
    return notes


def slot_diffs(gold: dict, row: dict, schema_map: dict) -> list[str]:
    keys = slot_keys(gold, schema_map)
    got = pred_args(row)
    want = gold["arguments"]
    diffs = []
    for key in keys:
        if got.get(key) != want.get(key):
            diffs.append(f"{key} gold={compact(want.get(key))} pred={compact(got.get(key))}")
    return diffs


def schema_note(name: str | None, args: dict, *, v1: set[str], schema_map: dict) -> str | None:
    if not name:
        return None
    if name not in v1:
        return f"`{name}` is not an in_v1 tool"
    schema = schema_map.get(name)
    if not schema:
        return f"no schema for `{name}`"
    return validate_schema(schema, args)


def classify(
    gold: dict,
    row: dict,
    *,
    server_map: dict[str, str],
    schema_map: dict,
    v1: set[str],
) -> dict:
    name = pred_name(row)
    gold_server = gold.get("server") or server_map.get(gold["tool"], "?")
    pred_server = server_map.get(name, "?") if name else None
    diffs = slot_diffs(gold, row, schema_map) if name else []
    run_diffs = [d for d in diffs if d.split(" ", 1)[0] in RUN_ID_KEYS]
    notes = prose_notes(str(row.get("raw") or ""))
    schema_err = schema_note(name, pred_args(row), v1=v1, schema_map=schema_map)

    if not row.get("valid"):
        primary = "invalid"
    elif name and pred_server != gold_server:
        primary = "wrong_server"
    elif not row.get("correct_tool"):
        primary = "wrong_tool"
    elif run_diffs:
        primary = "bad_run_id"
    elif not row.get("correct_args"):
        primary = "bad_slots"
    elif not row.get("exact_match"):
        primary = "inexact_optional"
    else:
        primary = "ok"

    tags = []
    if gold.get("collision"):
        tags.append(f"collision={gold['collision']}")
    if gold.get("tool") == "search_wandb_docs_tool":
        tags.append("docs")
    if gold.get("tool") == "get_abstract":
        tags.append("abstract")
    if notes:
        tags.extend(notes)

    return {
        "primary": primary,
        "gold_server": gold_server,
        "pred_server": pred_server,
        "diffs": diffs,
        "run_diffs": run_diffs,
        "notes": notes,
        "schema_err": schema_err,
        "tags": tags,
    }


def why(gold: dict, row: dict, info: dict) -> str:
    primary = info["primary"]
    name = pred_name(row) or "?"
    collision = gold.get("collision")
    hint = COLLISION_HINT.get(collision, "") if collision else ""

    if primary == "invalid":
        bits = []
        if info["notes"]:
            bits.append("raw is not a clean {name, arguments} object (" + ", ".join(info["notes"]) + ")")
        if info.get("schema_err"):
            bits.append(info["schema_err"])
        if name not in ("?",) and name != gold["tool"]:
            bits.append(f"parsed `{name}`")
        return "; ".join(bits) or "parsed JSON failed schema or is not an in_v1 tool"

    if primary == "wrong_server":
        base = (
            f"gold {info['gold_server']} `{gold['tool']}` vs "
            f"pred {info['pred_server']} `{name}`"
        )
        return f"{base} ({hint})" if hint else base

    if primary == "wrong_tool":
        base = f"gold `{gold['tool']}` vs pred `{name}`"
        return f"{base} ({hint})" if hint else base

    if primary == "bad_run_id":
        return "; ".join(info["run_diffs"]) or "run id slot mismatch"

    if primary == "bad_slots":
        return "; ".join(info["diffs"]) or "required/slot keys mismatch"

    extra = pred_args(row)
    gold_args = gold["arguments"]
    added = sorted(set(extra) - set(gold_args))
    missing = sorted(set(gold_args) - set(extra))
    bits = []
    if added:
        bits.append("extra " + ", ".join(f"`{k}`" for k in added))
    if missing:
        bits.append("missing " + ", ".join(f"`{k}`" for k in missing))
    for key in sorted(set(extra) & set(gold_args)):
        if extra[key] != gold_args[key]:
            bits.append(f"{key} gold={compact(gold_args[key])} pred={compact(extra[key])}")
    return "; ".join(bits) or "arguments differ on optionals"


def is_weak(gold: dict, row: dict) -> bool:
    if gold.get("collision"):
        return True
    if gold["tool"] == "search_wandb_docs_tool" and not row.get("correct_args"):
        return True
    if gold["tool"] == "get_abstract" and not row.get("correct_tool"):
        return True
    return False


def load_scored(path: Path) -> list[dict]:
    rows = [row for _, row in load_jsonl(path)]
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return rows


def join_rows(
    scored: list[dict],
    gold_by_id: dict[str, dict],
    *,
    server_map: dict[str, str],
    schema_map: dict,
    v1: set[str],
) -> list[dict]:
    joined = []
    for row in scored:
        rid = row.get("id")
        gold = gold_by_id.get(rid)
        if gold is None:
            raise SystemExit(f"pred id {rid!r} not in gold split")
        info = classify(gold, row, server_map=server_map, schema_map=schema_map, v1=v1)
        joined.append(
            {
                "id": rid,
                "gold": gold,
                "row": row,
                "info": info,
                "why": why(gold, row, info),
            }
        )
    return joined


def pct(n: int, d: int) -> str:
    if not d:
        return "0.0%"
    return f"{100.0 * n / d:.1f}%"


def card(item: dict) -> str:
    gold = item["gold"]
    row = item["row"]
    info = item["info"]
    name = pred_name(row)
    pred_obj = {"name": name, "arguments": pred_args(row)} if name else row.get("raw")
    bits = [f"`{item['id']}` · `{info['primary']}`"]
    if gold.get("collision"):
        bits.append(f"collision=`{gold['collision']}`")
    lines = [
        " · ".join(bits),
        "",
        f"intent: {gold['intent']}",
        f"gold: `{gold['tool']}` `{compact(gold['arguments'])}`",
        f"pred: `{compact(pred_obj)}`" if name else f"pred raw: {row.get('raw')!r}",
        f"why: {item['why']}",
    ]
    return "\n".join(lines)


def render(joined: list[dict], *, setup: str, pred_path: Path) -> str:
    n = len(joined)
    flags = ("valid", "correct_tool", "correct_args", "exact_match")
    flag_ok = {k: sum(1 for item in joined if item["row"].get(k)) for k in flags}
    by_primary = defaultdict(list)
    for item in joined:
        by_primary[item["info"]["primary"]].append(item)
    real = [
        item
        for item in joined
        if item["info"]["primary"] not in {"ok", "inexact_optional"}
    ]

    collisions = [item for item in joined if item["gold"].get("collision")]
    docs = [item for item in joined if item["gold"]["tool"] == "search_wandb_docs_tool"]
    abstracts = [item for item in joined if item["gold"]["tool"] == "get_abstract"]

    lines = [
        "# QLoRA failure gallery",
        "",
        f"Setup `{setup}` on the {n} test rows. Source: `{rel(pred_path)}` joined to `data/sft/canonical.jsonl`.",
        "Primary class is mutually exclusive (first match: invalid → wrong server → wrong tool → bad run_id → bad slots → inexact).",
        "Headline misses are everything except exact-only optional drift.",
        "",
        "## Overall",
        "",
        "| Bar | n | Rate |",
        "|---|---:|---:|",
        f"| Valid | {flag_ok['valid']} / {n} | {pct(flag_ok['valid'], n)} |",
        f"| Correct tool | {flag_ok['correct_tool']} / {n} | {pct(flag_ok['correct_tool'], n)} |",
        f"| Correct args | {flag_ok['correct_args']} / {n} | {pct(flag_ok['correct_args'], n)} |",
        f"| Exact | {flag_ok['exact_match']} / {n} | {pct(flag_ok['exact_match'], n)} |",
        f"| Headline misses | {len(real)} | |",
        f"| Inexact optionals | {len(by_primary['inexact_optional'])} | |",
        "",
        "## Primary class counts",
        "",
        "| Class | n |",
        "|---|---:|",
    ]
    for key in PRIMARY_ORDER:
        lines.append(f"| {PRIMARY_TITLE[key]} | {len(by_primary[key])} |")
    lines.append("| Exact / ok | " + str(len(by_primary["ok"])) + " |")

    def slice_line(label: str, items: list[dict], key: str) -> str:
        ok = sum(1 for item in items if item["row"].get(key))
        return f"| {label} | {len(items)} | {pct(ok, len(items))} |"

    lines += [
        "",
        "## Weak slices",
        "",
        "The three slices called out after the bake-off. Rates are on that slice only.",
        "",
        "| Slice | n | Rate |",
        "|---|---:|---:|",
        slice_line("Collisions — correct tool", collisions, "correct_tool"),
        slice_line("Collisions — correct args", collisions, "correct_args"),
        slice_line("`search_wandb_docs_tool` — correct args", docs, "correct_args"),
        slice_line("`get_abstract` — correct tool", abstracts, "correct_tool"),
        "",
        "## Takeaways",
        "",
    ]
    lines += takeaways(by_primary, real)
    lines.append("")

    for key in PRIMARY_ORDER:
        items = by_primary[key]
        title = PRIMARY_TITLE[key]
        lines += [f"## {title}", ""]
        if not items:
            lines += ["None.", ""]
            continue
        blurb = class_blurb(key, items)
        if blurb:
            lines += [blurb, ""]
        shown = items
        if key == "inexact_optional":
            shown = items[:INEXACT_EXAMPLES]
        for item in shown:
            lines += ["```", card(item), "```", ""]

    weak_items = [item for item in real if is_weak(item["gold"], item["row"])]
    lines += [
        "## Weak-slice appendix",
        "",
        f"Every headline miss in a collision row, a docs-args miss, or a `get_abstract` tool miss ({len(weak_items)}). Cards are in the class sections above.",
        "",
        "| id | Class | Slice | Why |",
        "|---|---|---|---|",
    ]
    if not weak_items:
        lines += ["| — | — | — | none |", ""]
    else:
        for item in weak_items:
            gold = item["gold"]
            slice_label = gold.get("collision") or gold["tool"]
            why_text = item["why"].replace("|", "\\|")
            lines.append(
                f"| `{item['id']}` | {PRIMARY_TITLE[item['info']['primary']]} | `{slice_label}` | {why_text} |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def takeaways(by_primary: dict, real: list[dict]) -> list[str]:
    invalid = by_primary["invalid"]
    prose = sum(1 for item in invalid if item["info"]["notes"])
    query_slots = sum(
        1
        for item in by_primary["bad_slots"]
        if any(d.startswith("query ") for d in item["info"]["diffs"])
    )
    wrong_server = by_primary["wrong_server"]
    ws_coll = sum(1 for item in wrong_server if item["gold"].get("collision"))
    if not wrong_server:
        server_line = "- No wrong-server misses."
    elif ws_coll == len(wrong_server):
        server_line = f"- Wrong server is rare ({len(wrong_server)}), all collision rows."
    else:
        server_line = f"- Wrong server is rare ({len(wrong_server)}); {ws_coll} are collision rows."
    if not invalid:
        invalid_line = "- No invalid / extra-prose misses."
    elif prose == 0:
        invalid_line = (
            f"- {len(invalid)} invalids, none extra prose — all parsed as JSON and failed `inputSchema`."
        )
    else:
        invalid_line = (
            f"- {len(invalid)} invalids; {prose} have extra prose / unparseable wrappers. "
            "The rest parsed as JSON and failed `inputSchema`."
        )
    return [
        invalid_line,
        f"- {len(real)} headline misses. {query_slots} of {len(by_primary['bad_slots'])} bad-slot rows are `query` drift (docs / Hub / paper frame words).",
        server_line,
    ]


def class_blurb(key: str, items: list[dict]) -> str:
    if key == "invalid":
        prose = sum(1 for item in items if item["info"]["notes"])
        if prose:
            return f"{len(items)} rows. {prose} are extra prose; the others failed schema."
        return f"{len(items)} rows. None are extra prose — each parsed as JSON and failed that tool's `inputSchema`."
    if key == "wrong_server":
        return f"{len(items)} rows. Gold server and predicted server disagree."
    if key == "wrong_tool":
        return f"{len(items)} rows. Same server, wrong name."
    if key == "bad_run_id":
        return f"{len(items)} rows. Right tool; `run_id` / `run_id_a` / `run_id_b` does not match gold."
    if key == "bad_slots":
        return f"{len(items)} rows. Right tool; a required/slot key other than run id is off."
    if key == "inexact_optional":
        return (
            f"{len(items)} rows have the right tool and required slots but differ on optionals "
            f"(`sample_runs`, `max_projects`, extra `keys`, …). Not counted as headline misses. "
            f"Showing {min(INEXACT_EXAMPLES, len(items))}."
        )
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--gold", type=Path, default=CANONICAL)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = load_json(CATALOG)
    v1 = v1_names(catalog)
    server_map = {t["name"]: t["server"] for t in catalog["tools"]}
    schema_map = {t["name"]: t["inputSchema"] for t in catalog["tools"]}
    gold_by_id = load_gold(args.gold, args.split)
    scored = load_scored(args.pred)
    joined = join_rows(
        scored,
        gold_by_id,
        server_map=server_map,
        schema_map=schema_map,
        v1=v1,
    )
    setup = scored[0].get("setup") or args.pred.stem
    text = render(joined, setup=setup, pred_path=args.pred)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")

    counts = Counter(item["info"]["primary"] for item in joined)
    print(f"n={len(joined)} -> {args.out}")
    for key in PRIMARY_ORDER:
        print(f"  {key:18} {counts[key]}")
    print(f"  {'ok':18} {counts['ok']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
