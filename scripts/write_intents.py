"""Rewrite licensed drafts into researcher intents via a local Ollama model.

MCPTune-style step 3: args already sampled. The one-line draft is a
constraint only — never stored as intent. The slot checker must pass.

search_wandb_docs_tool skips the rewriter and uses a How-do-I template.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARG_SETS = ROOT / "data" / "sft" / "arg_sets.jsonl"
OUT = ROOT / "data" / "sft" / "canonical.jsonl"

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:14b"

THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
FENCE_RE = re.compile(r"^```(?:\w+)?\s*|\s*```$", re.S)
QUOTE_RE = re.compile(r'^[\s"\'`“”]+|[\s"\'`“”]+$')
SPACE_RE = re.compile(r"\s+")

TOOL_NAMES = {
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
}

# Substrings that mean the model named the API instead of asking a question.
BANNED = TOOL_NAMES | {
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

SYSTEM = (
    "You rewrite a draft into one short request a researcher would type in Cursor. "
    "Keep every listed slot unchanged. Do not name tools, APIs, or function names. "
    "No quotes, no preamble, no bullet list. Output only the rewritten request."
)

STYLE = {
    "list_entities_tool": "underspecified discovery (what teams / who am I)",
    "query_wandb_entity_projects": "listing projects under an entity",
    "probe_project_tool": "first look at a project, not a single-run curve",
    "get_run_history_tool": "curves / time series for one run, not a health check",
    "compare_runs_tool": "diff two run ids",
    "diagnose_run_tool": "health / overfit / NaN language, not 'show the curve'",
    "search_wandb_docs_tool": "how-do-I product/SDK docs, not run data",
    "hub_repo_search": "find a Hugging Face model or dataset, not a W&B run",
    "hub_repo_details": "inspect a known repo id",
    "search_papers": "find a paper by topic, not a Hub repo",
    "get_abstract": "known arXiv id, abstract only",
}


def norm(text: str) -> str:
    return SPACE_RE.sub(" ", text.casefold()).strip()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def join_and(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def licensed_draft(tool: str, arguments: dict) -> str:
    a = arguments
    if tool == "list_entities_tool":
        return "What W&B teams or usernames can I see?"
    if tool == "query_wandb_entity_projects":
        entity = a.get("entity")
        n = a.get("max_projects")
        if entity and n is not None:
            return f"List up to {n} W&B projects under entity {entity}."
        if entity:
            return f"List W&B projects under entity {entity}."
        return "What W&B projects do I have?"
    if tool == "probe_project_tool":
        loc = f"{a['entity_name']}/{a['project_name']}"
        n = a.get("sample_runs")
        if n is not None:
            return f"What's in the {loc} W&B project? Sample {n} runs."
        return f"What's in the {loc} W&B project?"
    if tool == "get_run_history_tool":
        loc = f"{a['entity_name']}/{a['project_name']}"
        rid = a["run_id"]
        keys = a.get("keys") or []
        samples = a.get("samples")
        if keys and samples is not None:
            return f"Show {join_and(keys)} for run {rid} in {loc}, {samples} sampled points."
        if keys:
            return f"Show {join_and(keys)} for run {rid} in {loc}."
        if samples is not None:
            return f"Show {samples} sampled metric points for run {rid} in {loc}."
        return f"Show the metric history for run {rid} in {loc}."
    if tool == "compare_runs_tool":
        loc = f"{a['entity_name']}/{a['project_name']}"
        bits = [f"Compare runs {a['run_id_a']} and {a['run_id_b']} in {loc}."]
        if a.get("include_history_overlap"):
            bits.append("Include history overlap.")
        keys = a.get("history_keys") or []
        if keys:
            bits.append(f"Compare {join_and(keys)}.")
        return " ".join(bits)
    if tool == "diagnose_run_tool":
        loc = f"{a['entity_name']}/{a['project_name']}"
        bits = [f"Is run {a['run_id']} in {loc} overfitting?"]
        if a.get("loss_key"):
            bits.append(f"Use {a['loss_key']} as the loss.")
        if a.get("val_loss_key"):
            bits.append(f"Use {a['val_loss_key']} as the val loss.")
        return " ".join(bits)
    if tool == "search_wandb_docs_tool":
        return f"W&B docs lookup: {a['query']}"
    if tool == "hub_repo_search":
        types = a.get("repo_types") or ["model"]
        kind = join_and(types)
        bits = [f"Find a Hugging Face {kind} for {a['query']}."]
        if a.get("sort"):
            bits.append(f"Sort by {a['sort']}.")
        if a.get("limit") is not None:
            bits.append(f"Limit {a['limit']}.")
        return " ".join(bits)
    if tool == "hub_repo_details":
        rid = (a.get("repo_ids") or ["?"])[0]
        rtype = a.get("repo_type")
        if rtype:
            return f"What's in the Hugging Face {rtype} {rid}?"
        return f"What's in {rid}?"
    if tool == "search_papers":
        bits = [f"Find papers on {a['query']}."]
        cats = a.get("categories") or []
        if cats:
            bits.append(f"Categories {join_and(cats)}.")
        if a.get("sort_by") == "date":
            bits.append("Sort by date.")
        if a.get("max_results") is not None:
            bits.append(f"Max {a['max_results']} results.")
        return " ".join(bits)
    if tool == "get_abstract":
        return f"Abstract of {a['paper_id']}."
    raise ValueError(f"no draft for {tool}")


def slots_from(arguments: dict) -> list[tuple[str, str]]:
    """(value, kind) pairs the rewrite must keep."""
    out: list[tuple[str, str]] = []
    for key, value in arguments.items():
        if value is None:
            continue
        if key in ID_KEYS and isinstance(value, str):
            out.append((value, "exact"))
        elif key in QUERY_KEYS and isinstance(value, str):
            out.append((value, "query"))
        elif key in LIST_EXACT_KEYS and isinstance(value, list):
            out.extend((str(x), "exact") for x in value if x)
        elif key == "repo_types" and isinstance(value, list):
            out.extend((str(x), "word") for x in value if x)
        elif key in ENUM_KEYS and isinstance(value, str):
            out.append((value, "word"))
        elif key in NUM_KEYS and isinstance(value, (int, float)):
            out.append((str(value), "num"))
        elif key == "include_history_overlap" and value is True:
            out.append(("overlap", "word"))
        elif isinstance(value, str) and value:
            out.append((value, "exact"))
    return out


def clean_reply(text: str) -> str:
    text = THINK_RE.sub("", text)
    text = FENCE_RE.sub("", text).strip()
    text = text.splitlines()[0].strip() if text.strip() else ""
    text = QUOTE_RE.sub("", text).strip()
    return SPACE_RE.sub(" ", text).strip()


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
    if kind == "word":
        return has_word(intent, value)
    if kind == "num":
        return has_word(intent, value)
    return value in intent


def check_intent(intent: str, draft: str, arguments: dict) -> str | None:
    if not intent or len(intent) < 8:
        return "empty or too short"
    if len(intent) > 280:
        return "too long"
    folded = norm(intent)
    if folded == norm(draft):
        return "identical to draft"
    for ban in BANNED:
        if ban.casefold() in folded:
            return f"names {ban}"
    if "{" in intent or "}" in intent:
        return "looks like json"
    for value, kind in slots_from(arguments):
        if not slot_ok(intent, value, kind):
            return f"missing slot {value!r}"
    return None


def rewrite_prompt(tool: str, draft: str, arguments: dict, variant: int) -> str:
    slots = [v for v, _ in slots_from(arguments)]
    slot_lines = "\n".join(f"- {s}" for s in slots) if slots else "- (none — vary the wording)"
    tones = ("terse", "a full question", "casual")
    tone = tones[variant % len(tones)]
    return (
        f"Style: {STYLE.get(tool, 'short researcher request')}. Tone: {tone}.\n"
        f"Slots that must appear unchanged:\n{slot_lines}\n\n"
        f"Draft:\n{draft}\n\n"
        "Rewrite the draft. One sentence only."
    )


def ollama_chat(host: str, model: str, prompt: str, timeout: float) -> str:
    url = host.rstrip("/") + "/api/chat"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.9, "num_predict": 128},
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
    return clean_reply(str(msg.get("content") or ""))


def ping(host: str, model: str) -> None:
    url = host.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"cannot reach Ollama at {host} ({exc}). Is it running?"
        ) from exc
    names = [t.get("name") or t.get("model") for t in tags.get("models", [])]
    if model not in names and not any(n.startswith(model) for n in names if n):
        raise SystemExit(
            f"model {model!r} not in `ollama list` ({', '.join(n for n in names if n) or 'none'})."
        )


def next_ft_id(existing: list[dict]) -> int:
    best = 0
    for row in existing:
        m = re.fullmatch(r"ft-(\d+)", str(row.get("id", "")))
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def already_kept(existing: list[dict]) -> dict[str, list[str]]:
    by_arg: dict[str, list[str]] = {}
    for row in existing:
        arg_id = row.get("arg_id")
        intent = row.get("intent")
        if arg_id and intent:
            by_arg.setdefault(arg_id, []).append(intent)
    return by_arg


def unique_enough(intent: str, seen: list[str]) -> bool:
    n = norm(intent)
    return all(n != norm(s) for s in seen)


def attempts_for(tool: str, default_attempts: int) -> int:
    # Extra tries for the empty-args tool unless the user lowered --attempts.
    if tool == "list_entities_tool" and default_attempts >= 3:
        return max(default_attempts, 8)
    return default_attempts


def looks_like_api(query: str) -> bool:
    return "wandb." in query.lower() or "WANDB_" in query or "." in query


def docs_intent(query: str) -> str:
    q = query.strip()
    low = q.lower()
    if low.startswith(("how ", "what ", "where ", "why ")):
        text = q if q.endswith("?") else f"{q}?"
        return text[0].upper() + text[1:]
    if looks_like_api(q):
        return f"How do I use {q}?"
    return f"How do I {q}?"


def keep_for(tool: str, default_keep: int) -> int:
    if tool == "list_entities_tool" and default_keep >= 2:
        return max(default_keep, 5)
    if tool == "search_wandb_docs_tool":
        return 1
    return default_keep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--in", dest="inp", type=Path, default=ARG_SETS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--keep", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="process only N arg sets (0 = all)")
    parser.add_argument("--tool", action="append", default=[], help="only these tool names (repeatable)")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--dry-run", action="store_true", help="print drafts, do not call Ollama")
    parser.add_argument("--force", action="store_true", help="ignore existing canonical.jsonl")
    args = parser.parse_args()

    arg_sets = load_jsonl(args.inp)
    if not arg_sets:
        raise SystemExit(f"no arg sets in {args.inp}")
    if args.tool:
        unknown = set(args.tool) - TOOL_NAMES
        if unknown:
            raise SystemExit(f"unknown --tool {sorted(unknown)}")
        arg_sets = [r for r in arg_sets if r["tool"] in args.tool]
    if args.limit:
        arg_sets = arg_sets[: args.limit]

    existing = [] if args.force else load_jsonl(args.out)
    kept_by_arg = already_kept(existing)
    seen_intents = [r["intent"] for r in existing if r.get("intent")]
    next_id = next_ft_id(existing)
    written = 0
    skipped = 0
    failed = 0
    per_tool: Counter[str] = Counter()

    if args.dry_run:
        for row in arg_sets:
            draft = licensed_draft(row["tool"], row["arguments"])
            slots = slots_from(row["arguments"])
            print(f"{row['id']}  {row['tool']}")
            print(f"  draft: {draft}")
            if row["tool"] == "search_wandb_docs_tool":
                print(f"  template: {docs_intent(row['arguments']['query'])}")
            print(f"  slots: {[v for v, _ in slots]}")
        print(f"{len(arg_sets)} drafts (dry-run, no Ollama)")
        return

    docs_only = bool(arg_sets) and all(r["tool"] == "search_wandb_docs_tool" for r in arg_sets)
    if not docs_only:
        ping(args.host, args.model)
        print(f"ollama {args.model} at {args.host}  in={len(arg_sets)}  existing={len(existing)}")
    else:
        print(f"docs templates  in={len(arg_sets)}  existing={len(existing)}")

    for i, row in enumerate(arg_sets, start=1):
        tool = row["tool"]
        arguments = row["arguments"]
        want = keep_for(tool, args.keep)
        have = list(kept_by_arg.get(row["id"], []))
        if len(have) >= want:
            skipped += 1
            continue

        if tool == "search_wandb_docs_tool":
            intent = docs_intent(arguments["query"])
            if not unique_enough(intent, have + seen_intents):
                failed += 1
                print(f"FAIL {row['id']} {tool}  template={intent} (duplicate)")
                continue
            rec = {
                "id": f"ft-{next_id:06d}",
                "split": row["split"],
                "intent": intent,
                "tool": tool,
                "arguments": arguments,
                "server": row["server"],
                "source": "template",
                "arg_id": row["id"],
            }
            next_id += 1
            written += 1
            per_tool[tool] += 1
            seen_intents.append(intent)
            have.append(intent)
            kept_by_arg.setdefault(row["id"], []).append(intent)
            append_jsonl(args.out, rec)
            print(f"[{i}/{len(arg_sets)}] {row['id']} {tool} kept={len(have)} (template)", flush=True)
            continue

        draft = licensed_draft(tool, arguments)
        n_try = attempts_for(tool, args.attempts)
        accepted: list[str] = []
        for attempt in range(n_try):
            if len(have) + len(accepted) >= want:
                break
            try:
                raw = ollama_chat(
                    args.host,
                    args.model,
                    rewrite_prompt(tool, draft, arguments, attempt),
                    args.timeout,
                )
            except RuntimeError as exc:
                print(f"  reject {row['id']} #{attempt + 1}: {exc}")
                continue
            reason = check_intent(raw, draft, arguments)
            if reason:
                print(f"  reject {row['id']} #{attempt + 1}: {reason} | {raw[:80]}")
                continue
            if not unique_enough(raw, have + accepted + seen_intents):
                print(f"  reject {row['id']} #{attempt + 1}: duplicate")
                continue
            accepted.append(raw)

        if not accepted and not have:
            failed += 1
            print(f"FAIL {row['id']} {tool}  draft={draft}")
            continue

        for intent in accepted:
            rec = {
                "id": f"ft-{next_id:06d}",
                "split": row["split"],
                "intent": intent,
                "tool": tool,
                "arguments": arguments,
                "server": row["server"],
                "source": "llm",
                "arg_id": row["id"],
            }
            next_id += 1
            written += 1
            per_tool[tool] += 1
            seen_intents.append(intent)
            have.append(intent)
            kept_by_arg.setdefault(row["id"], []).append(intent)
            append_jsonl(args.out, rec)

        print(
            f"[{i}/{len(arg_sets)}] {row['id']} {tool} "
            f"kept={len(have)} (+{len(accepted)})",
            flush=True,
        )

    print(
        f"wrote {written} intents -> {args.out}  "
        f"skipped_done={skipped}  failed={failed}"
    )
    for tool, n in per_tool.most_common():
        print(f"  {tool:32} +{n}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
