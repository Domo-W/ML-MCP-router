"""HF merged folder -> F16 GGUF -> Q4_K_M.

Uses the local llama.cpp clone (converter + already-built llama-quantize).

  python scripts/convert_gguf.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MERGED = ROOT / "outputs" / "qwen25-3b-merged"
DEFAULT_F16 = ROOT / "outputs" / "qwen25-3b-f16.gguf"
DEFAULT_Q4 = ROOT / "outputs" / "qwen25-3b-q4_k_m.gguf"
DEFAULT_LLAMA = Path(r"E:\Github\llama.cpp")
DEFAULT_QUANTIZE = (
    DEFAULT_LLAMA / "build_vs2022" / "bin" / "Release" / "llama-quantize.exe"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--f16", type=Path, default=DEFAULT_F16)
    parser.add_argument("--out", type=Path, default=DEFAULT_Q4)
    parser.add_argument("--llama-cpp", type=Path, default=DEFAULT_LLAMA)
    parser.add_argument("--quantize", type=Path, default=DEFAULT_QUANTIZE)
    parser.add_argument(
        "--skip-f16",
        action="store_true",
        help="reuse an existing --f16 file",
    )
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    merged = args.merged.resolve()
    f16 = args.f16.resolve()
    out = args.out.resolve()
    converter = args.llama_cpp.resolve() / "convert_hf_to_gguf.py"
    quantize = args.quantize.resolve()

    if not merged.is_dir():
        raise SystemExit(f"missing merged model {merged}")
    if not converter.is_file():
        raise SystemExit(f"missing converter {converter}")
    if not quantize.is_file():
        raise SystemExit(f"missing llama-quantize {quantize}")

    f16.parent.mkdir(parents=True, exist_ok=True)
    if args.skip_f16:
        if not f16.is_file():
            raise SystemExit(f"--skip-f16 but missing {f16}")
    else:
        run(
            [
                sys.executable,
                str(converter),
                str(merged),
                "--outfile",
                str(f16),
                "--outtype",
                "f16",
                "--model-name",
                "qwen25-3b-mcp",
            ]
        )

    run([str(quantize), str(f16), str(out), "Q4_K_M"])
    size_gb = out.stat().st_size / 1e9
    print(f"saved Q4_K_M -> {out}  ({size_gb:.1f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
