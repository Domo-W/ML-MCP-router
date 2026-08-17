"""Pick a generate backend from ML_ROUTER_BACKEND."""

from __future__ import annotations

import os


def make_generate(backend: str | None = None):
    kind = backend or os.environ.get("ML_ROUTER_BACKEND", "llama")
    if kind == "llama":
        from ml_router.generate_llama import generate

        return generate
    if kind == "ollama":
        from ml_router.generate_ollama import generate

        return generate
    raise RuntimeError(f"unknown ML_ROUTER_BACKEND={kind!r} (llama|ollama)")
