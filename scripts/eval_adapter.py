"""Run the saved PEFT adapter on the test split.

Setup A prompt (names only). Writes prediction JSONL for eval_calls.py.

  python scripts/eval_adapter.py --limit 5
  python scripts/eval_adapter.py
  python scripts/eval_calls.py --pred data/eval/slm_qlora.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_ollama import already_done, system_prompt  # noqa: E402
from validate_canonical import CANONICAL, CATALOG, load_json, load_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_ADAPTER = ROOT / "outputs" / "qwen25-3b-qlora"
DEFAULT_SETUP = "slm_qlora"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--setup", default=DEFAULT_SETUP)
    parser.add_argument("--gold", type=Path, default=CANONICAL)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="max new calls (0 = all)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def load_model(model_id: str, adapter: Path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    if not adapter.is_dir():
        raise SystemExit(f"missing adapter {adapter}")

    tokenizer = AutoTokenizer.from_pretrained(adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, str(adapter))
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, system: str, user: str, max_new_tokens: int) -> tuple[str, int]:
    import torch

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    n_in = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(out[0, n_in:], skip_special_tokens=True)
    return raw.strip(), n_in


def main() -> int:
    args = parse_args()
    catalog = load_json(CATALOG)
    gold = [
        row
        for _, row in load_jsonl(args.gold)
        if args.split == "all" or row.get("split") == args.split
    ]
    if not gold:
        raise SystemExit(f"no gold rows for split={args.split!r}")

    out = args.out or (EVAL_DIR / f"{args.setup}.jsonl")
    done = already_done(out)
    todo = [row for row in gold if row["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print(f"nothing to do ({len(done)} already in {out})")
        return 0

    system = system_prompt("slm_no_schema", catalog)
    print(
        f"{args.setup}  adapter={args.adapter}  "
        f"todo={len(todo)}  skipped={len(done)}  -> {out}"
    )
    tokenizer, model = load_model(args.model, args.adapter)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("a", encoding="utf-8") as fh:
        for i, row in enumerate(todo, start=1):
            t0 = time.perf_counter()
            raw, tokens = generate(
                tokenizer, model, system, row["intent"], args.max_new_tokens
            )
            rec = {
                "id": row["id"],
                "setup": args.setup,
                "raw": raw,
                "input_tokens": tokens,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(
                f"  {i}/{len(todo)} {row['id']}  "
                f"tokens={tokens}  {rec['latency_ms']}ms"
            )

    print(f"wrote {len(todo)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
