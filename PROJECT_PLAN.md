# ML MCP Tool-Calling SLM

Fine-tune a small language model to route natural-language requests to W&B, Hugging Face, and arXiv MCP tools. Use it in Cursor as a sidecar, not as a replacement for Agent.

## Goal

A 1.5B–3B LoRA that, given a researcher-style question, emits one schema-valid tool call (name + arguments). Cursor Agent (Claude/GPT) stays the planner; the SLM is one MCP tool (`route_ml_tools`) that hides the v1 catalog (~15 schemas) from the frontier prompt.

Prove it with numbers: valid-call rate, correct-tool rate, input tokens vs stuffing all MCP schemas into Claude.

## Why this, not a general chatbot

Tool calling on a **fixed, small menu** is a real fine-tuning job (format + routing). Experiment facts stay in W&B/HF APIs (not weights). Schema tax on W&B+HF+arXiv is large; a specialist with a short prompt is the cost/reliability story.

## Scope

No Weave. Experiment tracking only on W&B Models (runs, history, artifacts, docs). Hub search on Hugging Face. Papers on arXiv (not HF `paper_search`, to avoid two “find a paper” tools).

Catalog frozen 2026-08-13 against live `tools/list` in `data/mcp/`. HF shifted: `hub_repo_search` / `hub_repo_details` (no `model_search` / `hf_doc_search`).

| Server | Train on (v1) | Do not train on |
|---|---|---|
| **W&B** | 8–10 run/artifact/docs tools below | GraphQL, reports, automations, **all Weave / eval-trace / agent-span tools** |
| **Hugging Face** | `hub_repo_search`, `hub_repo_details` | `hf_fs`, Spaces, image gen, `hf_whoami` |
| **arXiv** | `search_papers`, `get_abstract` | download/read/LaTeX/citations/watches (add `download_paper` later if needed) |

**Out of project:** GitHub MCP, MLflow, GPU clouds, Google Workspace, Figma.

### Frozen v1 catalog (14 tools)

**W&B — discovery and runs**

| Tool | Use when |
|---|---|
| `list_entities_tool` | “What teams/usernames can I see?” |
| `query_wandb_entity_projects` | “Projects under this entity” |
| `probe_project_tool` | First look at a project: metric/config keys, scale |
| `get_run_history_tool` | Loss/accuracy curves, time series for one run |
| `compare_runs_tool` | Two run IDs: config diff, which is better |
| `diagnose_run_tool` | Overfitting, NaNs, “is this run healthy?” |
| `search_wandb_docs_tool` | How to use the W&B SDK / product, not run data |

**W&B — artifacts (keep if you log models/datasets; drop if you never do)**

| Tool | Use when |
|---|---|
| `list_artifact_versions_tool` | Versions of a model/dataset collection |
| `get_artifact_details_tool` | One version: metadata, lineage |
| `compare_artifact_versions_tool` | What changed between `:v3` and `:v7` |

**W&B — skip:** `query_wandb_tool` (GraphQL), `create_wandb_report_tool`, `log_analysis_to_wandb`, registry/automation/integration tools, and every `*weave*` / `*agent*` / `summarize_evaluation_tool`.

**Hugging Face** (frozen against live `tools/list` 2026-08-13)

| Tool | Use when |
|---|---|
| `hub_repo_search` | Find a base model or dataset (`repo_types: ["model"]` or `["dataset"]`) |
| `hub_repo_details` | README / files / dataset structure for a specific repo id |

**Hugging Face — skip:** `hf_fs` (obese filesystem grammar; also covers docs/papers), `hf_whoami`, `dynamic_space`, `gr1_z_image_turbo_generate`. Live Hub has no `model_search` / `dataset_search` / `hf_doc_search`.

**arXiv**

| Tool | Use when |
|---|---|
| `search_papers` | “LoRA paper”, “DPO 2023”, cs.LG queries |
| `get_abstract` | Known arXiv id, metadata + abstract only |

If the installed arXiv server has no `get_abstract`, use `search_papers` only in v1. Do not train `download_paper` / `read_paper` until you actually page full text in Cursor.

**Routing collisions to put in the eval set**

- “Best Qwen 3B” → HF `hub_repo_search` (`repo_types: ["model"]`), not W&B runs
- “Best run in my project” → W&B `probe` / history / compare, not Hub likes
- “LoRA paper” → arXiv `search_papers`, not `hf_fs` or `search_wandb_docs_tool`
- “Is training overfitting?” → `diagnose_run_tool`, not `get_run_history_tool`
- “How do I log metrics?” → `search_wandb_docs_tool`, not `probe_project_tool`

## Architecture

```
User (Cursor Agent / Claude)
        │  one short tool: route_ml_tools({intent})
        ▼
Local MCP server
        │  LoRA: intent → {name, arguments}
        │  validate against JSON Schema
        ▼
Official W&B / HF / arXiv MCP  →  result back to Agent
```

Fallback: if parse/schema fails, return an error (optional repair retry, then Claude).

Do **not** set this LoRA as Cursor’s Agent model. Distribution shift (Cursor’s huge prompt + 40 tools) will break it.

## Data (~800–2,000 traces)

Each example: user intent → gold `{tool, arguments}`.

- Synthesize from the real MCP JSON Schemas (MCPTune-style).
- Paraphrase like a researcher: “is PPO overfitting?”, “Qwen 3B instruct GGUF”, “LoRA paper 2021”.
- Hard negatives across servers: Hub search vs W&B best-run vs arXiv paper vs W&B docs.
- Hold out 100–200 prompts (never in train). Include run-id vs display-name traps for W&B.

Log real FineTuning W&B runs so demos hit live data.

## Training

- Base: Qwen 1.7B–3B Instruct (or similar), QLoRA/LoRA (Unsloth or TRL+PEFT).
- Loss on assistant tokens only (the tool call).
- Chat template native to the base model.
- 1–3 epochs; watch overfitting on entity/project names.
- Export adapter; optional GGUF/Ollama for local serve.

## Evaluation (the project’s evidence)

Same held-out set for every row:

| Setup | Valid % | Correct tool % | Correct args % | Input tokens/req | Latency |
|---|---|---|---|---|---|
| SLM, no schemas | | | | | |
| SLM + all W&B/HF/arXiv schemas in prompt | | | | | |
| SLM + LoRA, 1-line prompt | | | | | |
| Claude/GPT, full schemas, uncached | | | | | |
| Claude/GPT, cached prefix | | | | | |

**Valid** = parseable JSON, known tool, required fields, types OK.  
**Correct tool** = right name for the intent.  
**Correct args** = required slots filled (entity, project, run_id, query, …).

If cached Claude wins on quality and dollars, say so. The win can still be: shorter context, local routing, fewer tools in Agent’s menu.

## Cursor integration

1. Serve the adapter (vLLM or Ollama).
2. Thin stdio MCP: `route_ml_tools` → LoRA → validate → call upstream MCP → return result.
3. Register in `.cursor/mcp.json`. Keep Agent on a frontier model.
4. Disable or hide the raw W&B/HF/arXiv tool flood when the router is on (stay under Cursor’s ~40-tool cap).

## Future extension: MCP measurement (no training)

The LoRA is optional. A more widely useful follow-on is **measurement of the user’s MCP config**, not a model:

- Dump schema tax: tokens per tool / per server (tiktoken or provider `count_tokens`).
- Recommend a diet: drop unused and obese schemas (e.g. W&B GraphQL + report builder).
- A/B **all tools vs trimmed** on a small prompt set: tool-choice accuracy and input tokens (uncached vs cached).
- Ship as `npx mcp-cost` and/or a tiny Cursor-oriented CLI. Optional later: train a LoRA on the surviving catalog.

This extension does not depend on W&B/HF/arXiv. The v1 catalog above is just the first config it could score.

Do **not** build this before the LoRA bake-off unless the project stalls; it is the plausible productization path, not v1.

## Suggested phases

1. Freeze the tool list; dump schemas; count tokens per server (schema tax).
2. Build train/val/test JSONL; write a schema validator.
3. Baseline: prompted small model + prompted Claude on the test set.
4. QLoRA; log to W&B (dogfood the domain).
5. Eval table + failure gallery (wrong server, bad run_id, extra prose).
6. MCP sidecar in Cursor; 20 live prompts on a real W&B project.

## Resume line (target)

Fine-tuned a 1.5B–3B LoRA to route ML MCP tools (W&B, Hugging Face, arXiv); raised valid/correct-call rates vs the prompted base model and compared token cost to Claude with full schemas. Served as a Cursor MCP sidecar so Agent delegates experiment/Hub/paper lookups without loading every schema.

## Non-goals

- Replacing Cursor Agent or Tab.
- Putting experiment facts or paper text into weights.
- Training on GraphQL, report layout, GPU provisioning, or Weave traces.
- Claiming the architecture is novel (TinyAgent, MCPTune, tool-smith). Novelty is **this toolchain + honest bake-off**.

## References to cite

TinyAgent; MCPTune; tool-smith; W&B MCP; Hugging Face MCP; arXiv MCP.
