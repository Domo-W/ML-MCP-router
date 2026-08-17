"""One-line names-only system prompt (same as eval setup A / QLoRA / GGUF)."""

from __future__ import annotations

from ml_router.catalog import v1_names

JSON_ONLY = (
    'Reply with a single JSON object {"name":"...","arguments":{...}}. '
    "No markdown, no extra keys, no prose."
)


def system_prompt(catalog: dict) -> str:
    listed = ", ".join(sorted(v1_names(catalog)))
    return f"{JSON_ONLY} name must be one of: {listed}."
