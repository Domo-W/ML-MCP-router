"""Bake the QLoRA adapter into a full bf16 copy of the base.

Loads the original Qwen weights (not 4-bit), applies the adapter, writes
one merged folder. Does not overwrite the adapter. CPU by default so the
sidecar can keep the 3080.

  python scripts/merge_adapter.py
  python scripts/merge_adapter.py --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_ADAPTER = ROOT / "outputs" / "qwen25-3b-qlora"
DEFAULT_OUT = ROOT / "outputs" / "qwen25-3b-merged"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="cpu is safer on this box (sidecar holds VRAM)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    adapter = args.adapter.resolve()
    out = args.out.resolve()

    if not adapter.is_dir():
        raise SystemExit(f"missing adapter {adapter}")
    if not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"not a PEFT adapter: {adapter}")
    if out == adapter:
        raise SystemExit("refusing to overwrite the adapter dir")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for --device cuda")

    print(f"base={args.model}  adapter={adapter}  device={args.device}  -> {out}")

    tokenizer = AutoTokenizer.from_pretrained(adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = {"": 0} if args.device == "cuda" else "cpu"
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter))
    merged = model.merge_and_unload()

    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    tokenizer.save_pretrained(out)

    leftover = out / "adapter_config.json"
    if leftover.is_file():
        raise SystemExit(f"merge wrote an adapter file: {leftover}")

    weights = list(out.glob("*.safetensors")) + list(out.glob("pytorch_model*.bin"))
    if not weights:
        raise SystemExit(f"no weight files in {out}")
    size_gb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e9
    print(f"saved merged model -> {out}  ({size_gb:.1f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
