"""Parse a model reply into {name, arguments}."""

from __future__ import annotations

import json
import re

THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
FENCE_RE = re.compile(r"^```(?:\w+)?\s*|\s*```$", re.S)


def parse_call(raw: str) -> dict | None:
    text = THINK_RE.sub("", raw or "")
    text = FENCE_RE.sub("", text.strip()).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or set(obj) != {"name", "arguments"}:
        return None
    if not isinstance(obj["name"], str) or not isinstance(obj["arguments"], dict):
        return None
    return obj
