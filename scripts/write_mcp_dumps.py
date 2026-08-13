"""Rebuild catalog.v1.json and schema_tax.md from data/mcp/dump/*.json."""

from __future__ import annotations

import json
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "mcp"
DUMP = OUT / "dump"

V1 = {
    "wandb": [
        "list_entities_tool",
        "query_wandb_entity_projects",
        "probe_project_tool",
        "get_run_history_tool",
        "compare_runs_tool",
        "diagnose_run_tool",
        "search_wandb_docs_tool",
        "list_artifact_versions_tool",
        "get_artifact_details_tool",
        "compare_artifact_versions_tool",
    ],
    "huggingface": [
        "hub_repo_search",
        "hub_repo_details",
    ],
    "arxiv": [
        "search_papers",
        "get_abstract",
    ],
}

ENC = tiktoken.get_encoding("cl100k_base")


def schema_text(tool: dict) -> str:
    payload = {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "inputSchema": tool.get("inputSchema", {}),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def tokens_for(tool: dict) -> int:
    return len(ENC.encode(schema_text(tool)))


def md_table(headers: list[str], body: list[list]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in body:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def main() -> None:
    dumps: dict[str, dict] = {}
    for server in V1:
        path = DUMP / f"{server}.json"
        dumps[server] = json.loads(path.read_text(encoding="utf-8"))
        print(f"read {path.relative_to(ROOT)} ({dumps[server]['tool_count']} tools)")

    catalog_tools = []
    missing = []
    for server, names in V1.items():
        by_name = {t["name"]: t for t in dumps[server]["tools"]}
        for name in names:
            if name not in by_name:
                missing.append(f"{server}:{name}")
                continue
            tool = by_name[name]
            catalog_tools.append(
                {
                    **tool,
                    "in_v1": True,
                    "schema_tokens_cl100k": tokens_for(tool),
                }
            )

    catalog = {
        "version": "v1",
        "frozen_at": dumps["wandb"].get("dumped_at", ""),
        "notes": [
            "Live HF tools replaced plan names model_search/dataset_search/hf_doc_search with hub_repo_search + hub_repo_details.",
            "hf_fs exists but is excluded from v1 (obese filesystem grammar; docs search lives there).",
            "Train only these tools. Full dumps in dump/ are for schema-tax measurement.",
        ],
        "tool_count": len(catalog_tools),
        "missing": missing,
        "tools": catalog_tools,
    }
    catalog_path = OUT / "catalog.v1.json"
    catalog_path.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {catalog_path.relative_to(ROOT)} ({len(catalog_tools)} tools) missing={missing}")

    rows = []
    for server, dump in dumps.items():
        v1_names = set(V1[server])
        for tool in dump["tools"]:
            rows.append(
                {
                    "server": server,
                    "name": tool["name"],
                    "in_v1": tool["name"] in v1_names,
                    "tokens": tokens_for(tool),
                }
            )
    rows.sort(key=lambda r: (-r["tokens"], r["server"], r["name"]))

    all_tokens = sum(r["tokens"] for r in rows)
    v1_tokens = sum(r["tokens"] for r in rows if r["in_v1"])
    by_server_all: dict[str, int] = {}
    by_server_v1: dict[str, int] = {}
    for row in rows:
        by_server_all[row["server"]] = by_server_all.get(row["server"], 0) + row["tokens"]
        if row["in_v1"]:
            by_server_v1[row["server"]] = by_server_v1.get(row["server"], 0) + row["tokens"]

    dumped_at = dumps["wandb"].get("dumped_at", "")
    lines = [
        "# MCP schema tax",
        "",
        "Counted with tiktoken `cl100k_base` on `name + description + inputSchema` (compact JSON).",
        f"Dumped {dumped_at} from live Cursor MCP `tools/list`. `mcp_auth` omitted (Cursor wrapper).",
        "",
        "## Totals",
        "",
        md_table(
            ["Scope", "Tools", "Tokens"],
            [
                ["All tools (3 servers)", len(rows), all_tokens],
                ["Frozen v1 catalog", sum(1 for r in rows if r["in_v1"]), v1_tokens],
                [
                    "Dropped (not trained)",
                    sum(1 for r in rows if not r["in_v1"]),
                    all_tokens - v1_tokens,
                ],
            ],
        ),
        "",
        (
            f"v1 is **{100 * v1_tokens / all_tokens:.1f}%** of full schema tokens "
            f"({all_tokens - v1_tokens} tokens saved vs stuffing every tool)."
        ),
        "",
        "## Per server",
        "",
        md_table(
            ["Server", "All tools", "All tokens", "v1 tools", "v1 tokens"],
            [
                [
                    server,
                    sum(1 for r in rows if r["server"] == server),
                    by_server_all[server],
                    sum(1 for r in rows if r["server"] == server and r["in_v1"]),
                    by_server_v1.get(server, 0),
                ]
                for server in ["wandb", "huggingface", "arxiv"]
            ],
        ),
        "",
        "## Per tool (all, by tokens desc)",
        "",
        md_table(
            ["Server", "Tool", "v1", "Tokens"],
            [
                [
                    r["server"],
                    f"`{r['name']}`",
                    "yes" if r["in_v1"] else "",
                    r["tokens"],
                ]
                for r in rows
            ],
        ),
        "",
        "## Notes",
        "",
        "- Hugging Face live names: `hub_repo_search` + `hub_repo_details`. Plan names `model_search`, `dataset_search`, `hf_doc_search` are gone.",
        "- `hf_fs` is the obese HF schema (filesystem grammar + docs/papers). Excluded from v1.",
        "- W&B Weave / GraphQL / reports / automations / registry are excluded from v1.",
        "- arXiv v1 is `search_papers` + `get_abstract` only.",
        "",
    ]
    tax_path = OUT / "schema_tax.md"
    tax_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {tax_path.relative_to(ROOT)}")
    print(f"all_tokens={all_tokens} v1_tokens={v1_tokens}")


if __name__ == "__main__":
    main()
