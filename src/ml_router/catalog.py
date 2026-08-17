"""Frozen v1 catalog (11 tools) and JSON Schema checks."""

from __future__ import annotations

import json
from importlib.resources import files

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

V1_SKIP = {
    "list_artifact_versions_tool",
    "get_artifact_details_tool",
    "compare_artifact_versions_tool",
}


def load_catalog() -> dict:
    path = files("ml_router").joinpath("data/catalog.v1.json")
    return json.loads(path.read_text(encoding="utf-8"))


def v1_names(catalog: dict) -> set[str]:
    return {
        t["name"]
        for t in catalog["tools"]
        if t.get("in_v1") and t["name"] not in V1_SKIP
    }


def validate_schema(schema: dict, arguments: dict) -> str | None:
    try:
        Draft202012Validator(schema).validate(arguments)
    except ValidationError as exc:
        return f"schema: {exc.message}"
    return None
