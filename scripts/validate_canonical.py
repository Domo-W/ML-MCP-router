"""Validate canonical SFT rows before split/train.

Plan step 5: drop or repair; do not train on invalid gold.
Read-only by default. --write-clean emits passing rows only.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

try:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import ValidationError
except ImportError:  # pragma: no cover - optional dep
    Draft202012Validator = None
    ValidationError = Exception

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "mcp" / "catalog.v1.json"
BANKS = ROOT / "data" / "sft" / "slot_banks.json"
ARG_SETS = ROOT / "data" / "sft" / "arg_sets.jsonl"
CANONICAL = ROOT / "data" / "sft" / "canonical.jsonl"

ID_RE = re.compile(r"^ft-\d{6}$")
RUN_ID = re.compile(r"^[a-z0-9]{8}$")
PAPER_ID = re.compile(r"^(\d{4}\.\d{4,5}|[a-z-]+/\d{7})$")
REPO_ID = re.compile(r"^[^/\s]+/[^/\s]+$")
ARXIV_CAT = re.compile(r"^[a-z-]+(\.[A-Z]{2})?$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SPACE_RE = re.compile(r"\s+")

REQUIRED_FIELDS = ("id", "split", "intent", "tool", "arguments")
SPLITS = {"train", "holdout", "val", "test"}
SOURCES = {"hand", "template", "llm"}
COLLISIONS = {
    "hub_vs_wandb",
    "probe_vs_hub",
    "paper_vs_hub",
    "diagnose_vs_history",
    "history_vs_diagnose",
    "docs_vs_probe",
    "abstract_vs_search",
    "details_vs_search",
}
V1_SKIP = {
    "list_artifact_versions_tool",
    "get_artifact_details_tool",
    "compare_artifact_versions_tool",
}
ID_KEYS = {
    "entity",
    "entity_name",
    "project_name",
    "run_id",
    "run_id_a",
    "run_id_b",
    "paper_id",
}
QUERY_KEYS = {"query"}
LIST_EXACT_KEYS = {"keys", "history_keys", "repo_ids", "categories"}
ENUM_KEYS = {"sort", "sort_by", "repo_type"}
NUM_KEYS = {"max_projects", "sample_runs", "samples", "limit", "max_results"}
BANNED = {
    "list_entities_tool",
    "query_wandb_entity_projects",
    "probe_project_tool",
    "get_run_history_tool",
    "compare_runs_tool",
    "diagnose_run_tool",
    "search_wandb_docs_tool",
    "hub_repo_search",
    "hub_repo_details",
    "search_papers",
    "get_abstract",
    "list_entities",
    "probe_project",
    "get_run_history",
    "compare_runs",
    "diagnose_run",
    "search_wandb_docs",
    "hub_repo",
    "inputschema",
    "tool call",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[tuple[int, dict]]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        rows.append((i, json.loads(line)))
    return rows


def norm(text: str) -> str:
    return SPACE_RE.sub(" ", text.casefold()).strip()


def v1_names(catalog: dict) -> set[str]:
    return {
        t["name"]
        for t in catalog["tools"]
        if t.get("in_v1") and t["name"] not in V1_SKIP
    }


def bank_index(banks: dict) -> dict[str, dict[str, set[str]]]:
    """split -> kind -> values. Catches invented ids and train/holdout leaks."""
    out = {
        split: {k: set() for k in ("entity", "project", "run_id", "repo_id", "paper_id")}
        for split in ("train", "holdout")
    }
    for ctx in banks["wandb"]["contexts"]:
        split = ctx["split"]
        out[split]["entity"].add(ctx["entity"])
        out[split]["project"].add(ctx["project"])
        for run in ctx["runs"]:
            out[split]["run_id"].add(run["id"])
    for item in banks["huggingface"]["repo_ids"]:
        out[item["split"]]["repo_id"].add(item["id"])
    for item in banks["arxiv"]["paper_ids"]:
        out[item["split"]]["paper_id"].add(item["id"])
    return out


def display_names(banks: dict) -> set[str]:
    names = set()
    for ctx in banks["wandb"]["contexts"]:
        for run in ctx["runs"]:
            names.add(run["name"])
    return names


def slots_from(arguments: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for key, value in arguments.items():
        if value is None or key == "repo_types":
            continue
        if key in ID_KEYS and isinstance(value, str):
            out.append((value, "exact"))
        elif key in QUERY_KEYS and isinstance(value, str):
            out.append((value, "query"))
        elif key in LIST_EXACT_KEYS and isinstance(value, list):
            out.extend((str(x), "exact") for x in value if x)
        elif key in ENUM_KEYS and isinstance(value, str):
            out.append((value, "word"))
        elif key in NUM_KEYS and isinstance(value, (int, float)):
            out.append((str(value), "num"))
        elif key == "include_history_overlap" and value is True:
            out.append(("overlap", "word"))
        elif isinstance(value, str) and value:
            out.append((value, "exact"))
    return out


def has_word(hay: str, needle: str) -> bool:
    return re.search(rf"(?i)(?<![A-Za-z0-9_]){re.escape(needle)}(?![A-Za-z0-9_])", hay) is not None


def slot_ok(intent: str, value: str, kind: str) -> bool:
    if kind == "exact":
        return value in intent
    if kind == "query":
        n_int, n_val = norm(intent), norm(value)
        if n_val in n_int:
            return True
        words = [w for w in n_val.split() if len(w) > 1]
        return bool(words) and all(w in n_int for w in words)
    if kind in {"word", "num"}:
        return has_word(intent, value)
    return value in intent


def parse_ymd(value: str) -> date | None:
    if not DATE_RE.match(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def validate_schema(schema: dict, arguments: dict) -> str | None:
    if Draft202012Validator is not None:
        try:
            Draft202012Validator(schema).validate(arguments)
        except ValidationError as exc:
            return f"schema: {exc.message}"
        return None
    if any(v is None for v in arguments.values()):
        return "null values must be omitted"
    required = schema.get("required") or []
    missing = [k for k in required if k not in arguments]
    if missing:
        return f"missing required {missing}"
    props = schema.get("properties") or {}
    extra = [k for k in arguments if k not in props]
    if extra and schema.get("additionalProperties") is False:
        return f"extra keys {extra}"
    for key, value in arguments.items():
        spec = props.get(key)
        if not spec:
            continue
        expected = spec.get("type")
        if expected == "string" and not isinstance(value, str):
            return f"{key} should be string"
        if expected == "integer" and not isinstance(value, int):
            return f"{key} should be integer"
        if expected == "number" and not isinstance(value, (int, float)):
            return f"{key} should be number"
        if expected == "boolean" and not isinstance(value, bool):
            return f"{key} should be boolean"
        if expected == "array" and not isinstance(value, list):
            return f"{key} should be array"
        enum = spec.get("enum")
        if enum and value not in enum:
            return f"{key}={value!r} not in {enum}"
        items = spec.get("items") or {}
        item_enum = items.get("enum")
        if item_enum and isinstance(value, list):
            bad = [v for v in value if v not in item_enum]
            if bad:
                return f"{key} items {bad} not in {item_enum}"
    return None


def bank_pairs(arguments: dict) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if "entity" in arguments:
        pairs.append(("entity", arguments["entity"]))
    if "entity_name" in arguments:
        pairs.append(("entity", arguments["entity_name"]))
    if "project_name" in arguments:
        pairs.append(("project", arguments["project_name"]))
    for key in ("run_id", "run_id_a", "run_id_b"):
        if key in arguments:
            pairs.append(("run_id", arguments[key]))
    if "paper_id" in arguments:
        pairs.append(("paper_id", arguments["paper_id"]))
    for repo in arguments.get("repo_ids") or []:
        pairs.append(("repo_id", repo))
    return pairs


def check_row(
    row: dict,
    *,
    v1: set[str],
    schema_map: dict[str, dict],
    server_map: dict[str, str],
    slots: dict[str, dict[str, set[str]]],
    names: set[str],
    arg_ids: set[str],
) -> list[str]:
    errors: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in row:
            errors.append(f"missing field {field}")
    if errors:
        return errors

    if not isinstance(row["id"], str) or not ID_RE.match(row["id"]):
        errors.append(f"bad id {row['id']!r}")
    if row["split"] not in SPLITS:
        errors.append(f"bad split {row['split']!r}")
    if not isinstance(row["intent"], str) or len(row["intent"].strip()) < 8:
        errors.append("intent empty or too short")
    if not isinstance(row["arguments"], dict):
        errors.append("arguments must be an object")
        return errors

    tool = row["tool"]
    arguments = row["arguments"]
    intent = row["intent"]

    if tool not in v1:
        errors.append(f"tool {tool!r} not in v1")
        return errors

    if any(v is None for v in arguments.values()):
        errors.append("null values must be omitted")
    if any(v == "" for v in arguments.values() if isinstance(v, str)):
        errors.append("empty string argument")

    schema_err = validate_schema(schema_map[tool], arguments)
    if schema_err:
        errors.append(schema_err)

    extra = [k for k in arguments if k not in (schema_map[tool].get("properties") or {})]
    if extra and schema_map[tool].get("additionalProperties") is False:
        errors.append(f"extra keys {extra}")

    for key in ("run_id", "run_id_a", "run_id_b"):
        if key not in arguments:
            continue
        value = arguments[key]
        if not isinstance(value, str) or not RUN_ID.match(value):
            errors.append(f"bad {key} {value!r}")
        elif value in names:
            errors.append(f"{key} is a display name {value!r}")
    if "paper_id" in arguments and not PAPER_ID.match(str(arguments["paper_id"])):
        errors.append(f"bad paper_id {arguments['paper_id']!r}")
    if "repo_ids" in arguments:
        ids = arguments["repo_ids"]
        if not isinstance(ids, list) or not ids:
            errors.append("repo_ids must be a non-empty org/name array")
        elif not all(isinstance(i, str) and REPO_ID.match(i) for i in ids):
            errors.append("repo_ids must be a non-empty org/name array")

    if tool == "list_entities_tool" and arguments != {}:
        errors.append("list_entities_tool arguments must be {}")
    if tool == "compare_runs_tool" and arguments.get("run_id_a") == arguments.get("run_id_b"):
        errors.append("run_id_a == run_id_b")
    if tool == "get_run_history_tool":
        lo, hi = arguments.get("min_step"), arguments.get("max_step")
        if isinstance(lo, int) and isinstance(hi, int) and lo > hi:
            errors.append("min_step > max_step")
    if tool == "hub_repo_search":
        if not arguments.get("query"):
            errors.append("hub_repo_search needs query")
        types = arguments.get("repo_types")
        if types not in (["model"], ["dataset"]):
            errors.append("hub_repo_search repo_types must be [model] or [dataset]")
    if tool == "search_papers":
        for key in ("date_from", "date_to"):
            if key in arguments and parse_ymd(str(arguments[key])) is None:
                errors.append(f"bad {key} {arguments[key]!r}")
        start, end = arguments.get("date_from"), arguments.get("date_to")
        if start and end:
            a, b = parse_ymd(str(start)), parse_ymd(str(end))
            if a and b and a > b:
                errors.append("date_from > date_to")
        for cat in arguments.get("categories") or []:
            if not ARXIV_CAT.match(str(cat)):
                errors.append(f"bad category {cat!r}")
    if tool == "get_abstract" and set(arguments) - {"paper_id"}:
        errors.append("get_abstract gold args must be only paper_id")
    if tool == "search_wandb_docs_tool":
        if any(k in arguments for k in ("run_id", "entity_name", "project_name")):
            errors.append("docs tool must not carry run/project slots")

    if "server" in row and row["server"] != server_map[tool]:
        errors.append(f"server {row['server']!r} != catalog {server_map[tool]!r}")
    if "source" in row and row["source"] not in SOURCES:
        errors.append(f"bad source {row['source']!r}")
    if "collision" in row and row["collision"] not in COLLISIONS:
        errors.append(f"bad collision {row['collision']!r}")
    if "arg_id" in row and row["arg_id"] not in arg_ids:
        errors.append(f"unknown arg_id {row['arg_id']!r}")

    bank_split = "holdout" if row["split"] in {"holdout", "val", "test"} else "train"
    other = "train" if bank_split == "holdout" else "holdout"
    for kind, value in bank_pairs(arguments):
        if not isinstance(value, str):
            continue
        if value in slots[other][kind]:
            errors.append(f"{kind} {value!r} leaks from {other} bank")
        elif value not in slots[bank_split][kind]:
            errors.append(f"{kind} {value!r} not in {bank_split} bank")

    folded = norm(intent)
    if "{" in intent or "}" in intent:
        errors.append("intent looks like json")
    for ban in BANNED:
        if ban.casefold() in folded:
            errors.append(f"intent names {ban}")
    for value, kind in slots_from(arguments):
        if not slot_ok(intent, value, kind):
            errors.append(f"intent missing slot {value!r}")

    exported = json.dumps({"name": tool, "arguments": arguments}, ensure_ascii=False)
    parsed = json.loads(exported)
    if set(parsed) != {"name", "arguments"}:
        errors.append("export is not {name, arguments}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="src", type=Path, default=CANONICAL)
    parser.add_argument("--write-clean", type=Path, help="write passing rows here")
    args = parser.parse_args()

    catalog = load_json(CATALOG)
    banks = load_json(BANKS)
    arg_ids = {row["id"] for _, row in load_jsonl(ARG_SETS)} if ARG_SETS.exists() else set()
    rows = load_jsonl(args.src)

    v1 = v1_names(catalog)
    schema_map = {t["name"]: t["inputSchema"] for t in catalog["tools"]}
    server_map = {t["name"]: t["server"] for t in catalog["tools"]}
    slots = bank_index(banks)
    names = display_names(banks)

    seen_ids: set[str] = set()
    seen_intents: dict[str, str] = {}
    failed = 0
    kept: list[dict] = []
    reasons: Counter[str] = Counter()

    for line_no, row in rows:
        errs = check_row(
            row,
            v1=v1,
            schema_map=schema_map,
            server_map=server_map,
            slots=slots,
            names=names,
            arg_ids=arg_ids,
        )
        rid = row.get("id", f"line-{line_no}")
        if rid in seen_ids:
            errs.append("duplicate id")
        seen_ids.add(rid)
        key = norm(row.get("intent") or "")
        if key and key in seen_intents:
            errs.append(f"duplicate intent (also {seen_intents[key]})")
        elif key:
            seen_intents[key] = rid

        if errs:
            failed += 1
            for err in errs:
                reasons[err.split(" ", 1)[0]] += 1
                print(f"{rid} L{line_no}: {err}")
        else:
            kept.append(row)

    engine = "jsonschema" if Draft202012Validator is not None else "stdlib-fallback"
    print(f"\n{len(rows)} rows  pass={len(kept)}  fail={failed}  ({engine})")
    if reasons:
        print("by prefix:", dict(reasons))
    if args.write_clean:
        args.write_clean.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
            encoding="utf-8",
        )
        print(f"wrote {len(kept)} clean rows -> {args.write_clean}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
