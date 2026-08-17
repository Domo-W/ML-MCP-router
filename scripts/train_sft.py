"""SFT Qwen2.5-3B-Instruct on the exported messages JSONL.

One recipe, two runs. --qlora loads the base in 4-bit; omit it for 16-bit LoRA.
Both log to W&B. Loss is assistant tokens only (the tool-call JSON).

  python scripts/train_sft.py --dry-run
  python scripts/train_sft.py --qlora
  python scripts/train_sft.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "sft" / "train.messages.jsonl"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_PROJECT = "mcp-slm"

LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qlora",
        action="store_true",
        help="4-bit base (QLoRA). Default is 16-bit LoRA.",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, help="adapter dir (default outputs/qwen25-3b-{tag})")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="W&B project")
    parser.add_argument("--run-name", help="W&B run name (default sft-qwen25-3b-{tag})")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print config and dataset size; do not load the model",
    )
    return parser.parse_args()


def tag(qlora: bool) -> str:
    return "qlora" if qlora else "lora"


def main() -> int:
    args = parse_args()
    kind = tag(args.qlora)
    out = args.out or (ROOT / "outputs" / f"qwen25-3b-{kind}")
    run_name = args.run_name or f"sft-qwen25-3b-{kind}"

    if not args.data.is_file():
        raise SystemExit(
            f"missing {args.data}. Run: python scripts/export_sft.py"
        )

    print(
        f"{kind}  model={args.model}  data={args.data}  "
        f"epochs={args.epochs}  lr={args.lr}  -> {out}"
    )
    print(f"wandb project={args.project}  run={run_name}")

    if args.dry_run:
        from datasets import load_dataset

        ds = load_dataset("json", data_files=str(args.data), split="train")
        print(f"rows={len(ds)}  columns={ds.column_names}")
        if "messages" not in ds.column_names:
            raise SystemExit("dataset has no messages column")
        print("dry-run ok")
        return 0

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required. This script will not train on CPU.")

    ds = load_dataset("json", data_files=str(args.data), split="train")
    if "messages" not in ds.column_names:
        raise SystemExit("dataset has no messages column")
    print(f"rows={len(ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )

    bnb = None
    if args.qlora:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "QLoRA needs bitsandbytes. Install it, or drop --qlora for 16-bit LoRA. "
                "On Windows this often means WSL2."
            ) from exc
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    os.environ.setdefault("WANDB_PROJECT", args.project)
    out.mkdir(parents=True, exist_ok=True)

    config = SFTConfig(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to="wandb",
        run_name=run_name,
        assistant_only_loss=True,
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        seed=args.seed,
    )

    trainer_kwargs = {
        "model": args.model,
        "args": config,
        "train_dataset": ds,
        "peft_config": peft_config,
        "processing_class": tokenizer,
    }
    if bnb is not None:
        trainer_kwargs["quantization_config"] = bnb

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(out))
    print(f"saved adapter -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
