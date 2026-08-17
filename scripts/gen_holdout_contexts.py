"""Invent holdout W&B contexts. Never reuse train or live-account strings.

Offline only: writes fake entity/project/run_id strings into
data/sft/slot_banks.json. Does not call W&B or create projects.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANKS = ROOT / "data" / "sft" / "slot_banks.json"
RUN_ID = re.compile(r"^[a-z0-9]{8}$")

# Live account — never emit these even if train is still empty.
LIVE_RESERVE = {
    "train-lab",
    "rl-benchmark",
    "rl-benchmark-b",
    "rl-benchmark-c",
    "a1b2c3d4",
    "k7m2n9p4",
    "v2x5y9z3",
    "h8j3k6m1",
    "e5f6g7h8",
    "i9j0k1l2",
    "b3x8w1q6",
    "f9h4j2k8",
    "m1p5r7t3",
    "c6d8e2f4",
    "n4q8s1w7",
}

TEMPLATES = [
    {
        "entity": "heldout-lab",
        "project": "lora-ablation",
        "names": ["ppo-clip-wide", "dpo-beta-0.1", "sft-qwen-3b", "grpo-math-v2"],
        "artifacts": ["reward-model", "sft-dataset"],
        "metric_keys": ["reward", "kl", "loss"],
    },
    {
        "entity": "eval-sandbox",
        "project": "adapter-sweep",
        "names": ["qwen-lora-r16", "qwen-lora-r64", "freeze-embed", "full-ft-smoke"],
        "artifacts": ["adapter", "eval-split"],
        "metric_keys": ["val_loss", "exact_match"],
    },
    {
        "entity": "paper-repl",
        "project": "dpo-reproduce",
        "names": ["beta-0.05", "beta-0.5", "ipo-tau", "kto-unpaired"],
        "artifacts": ["pref-pairs"],
        "metric_keys": ["train/loss", "eval/reward_acc"],
    },
]


def reserved_from(banks: dict) -> set[str]:
    out = set(LIVE_RESERVE)
    for ctx in banks.get("wandb", {}).get("contexts", []):
        out.add(ctx["entity"])
        out.add(ctx["project"])
        for run in ctx.get("runs", []):
            out.add(run["id"])
            out.add(run["name"])
        for art in ctx.get("artifacts", []):
            out.add(art)
            for part in str(art).split("/"):
                out.add(part.split(":")[0])
    return {s.lower() for s in out if s}


def new_run_id(rng: random.Random, taken: set[str]) -> str:
    alphabet = string.ascii_lowercase + string.digits
    while True:
        rid = "".join(rng.choice(alphabet) for _ in range(8))
        if RUN_ID.match(rid) and rid not in taken:
            taken.add(rid)
            return rid


def make_context(tmpl: dict, rng: random.Random, taken: set[str], n_runs: int) -> dict:
    entity, project = tmpl["entity"], tmpl["project"]
    if entity.lower() in taken or project.lower() in taken:
        raise SystemExit(f"template collides with reserved names: {entity}/{project}")
    taken.update({entity.lower(), project.lower()})

    runs = [{"id": new_run_id(rng, taken), "name": name} for name in tmpl["names"][:n_runs]]

    first = tmpl["artifacts"][0]
    artifacts = [
        f"{entity}/{project}/{first}:v1",
        f"{entity}/{project}/{first}:v3",
    ]
    if len(tmpl["artifacts"]) > 1:
        artifacts.append(f"{entity}/{project}/{tmpl['artifacts'][1]}:v0")

    return {
        "split": "holdout",
        "entity": entity,
        "project": project,
        "runs": runs,
        "artifacts": artifacts,
        "metric_keys": list(tmpl["metric_keys"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=3, help="holdout contexts (2–3)")
    parser.add_argument("--runs", type=int, default=3, help="fake runs per context (2–4)")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.count <= 3:
        raise SystemExit("--count must be 2 or 3")
    if not 2 <= args.runs <= 4:
        raise SystemExit("--runs must be 2–4")

    banks = json.loads(BANKS.read_text(encoding="utf-8"))
    contexts = banks.setdefault("wandb", {}).setdefault("contexts", [])
    contexts[:] = [c for c in contexts if c.get("split") != "holdout"]

    taken = reserved_from(banks)
    rng = random.Random(args.seed)
    new = [make_context(t, rng, taken, args.runs) for t in TEMPLATES[: args.count]]
    contexts.extend(new)

    if args.dry_run:
        print(json.dumps(new, indent=2))
        return
    BANKS.write_text(json.dumps(banks, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(new)} holdout contexts -> {BANKS}")


if __name__ == "__main__":
    main()
