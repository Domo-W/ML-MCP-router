"""Export canonical rows to TRL chat `messages` JSONL.

Projects intent/tool/arguments into setup A (names-only system prompt).
Canonical stays the source of truth; re-run after any rewrite.

  python scripts/export_sft.py --dry-run
  python scripts/export_sft.py
  python scripts/export_sft.py --split test -o data/sft/test.messages.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_calls import parse_call  # noqa: E402
from eval_ollama import system_prompt  # noqa: E402
from validate_canonical import (  # noqa: E402
    CANONICAL,
    CATALOG,
    load_json,
    load_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "data" / "sft"
SPLITS = ("train", "test", "all")


def assistant_json(row: dict) -> str:
    return json.dumps(
        {"name": row["tool"], "arguments": row["arguments"]},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def to_messages(row: dict, system: str) -> dict:
    content = assistant_json(row)
    parsed = parse_call(content)
    if parsed is None:
        raise ValueError(f"{row.get('id')}: assistant JSON failed parse_call")
    if parsed["name"] != row["tool"] or parsed["arguments"] != row["arguments"]:
        raise ValueError(f"{row.get('id')}: assistant JSON drifted from gold")
    return {
        "id": row["id"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": row["intent"]},
            {"role": "assistant", "content": content},
        ],
    }


def default_out(split: str) -> Path:
    name = "all" if split == "all" else split
    return SFT / f"{name}.messages.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=CANONICAL)
    parser.add_argument("--split", default="train", choices=SPLITS)
    parser.add_argument("-o", "--out", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print one mapped row and write nothing",
    )
    args = parser.parse_args()

    catalog = load_json(CATALOG)
    system = system_prompt("slm_no_schema", catalog)
    gold = [
        row
        for _, row in load_jsonl(args.gold)
        if args.split == "all" or row.get("split") == args.split
    ]
    if not gold:
        raise SystemExit(f"no gold rows for split={args.split!r}")

    mapped = [to_messages(row, system) for row in gold]
    out = args.out or default_out(args.split)

    if args.dry_run:
        print(json.dumps(mapped[0], ensure_ascii=False, indent=2))
        print(f"would write {len(mapped)} rows -> {out}")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in mapped:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"wrote {len(mapped)} rows split={args.split} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
