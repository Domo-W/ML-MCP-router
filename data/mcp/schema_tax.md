# MCP schema tax

Counted with tiktoken `cl100k_base` on `name + description + inputSchema` (compact JSON).
Dumped 2026-08-13 from live Cursor MCP `tools/list`. `mcp_auth` omitted (Cursor wrapper).

## Totals

| Scope | Tools | Tokens |
|---|---|---|
| All tools (3 servers) | 50 | 27537 |
| Frozen v1 catalog | 14 | 5842 |
| Dropped (not trained) | 36 | 21695 |

v1 is **21.2%** of full schema tokens (21695 tokens saved vs stuffing every tool).

## Per server

| Server | All tools | All tokens | v1 tools | v1 tokens |
|---|---|---|---|---|
| wandb | 30 | 22198 | 10 | 3907 |
| huggingface | 6 | 2440 | 2 | 731 |
| arxiv | 14 | 2899 | 2 | 1204 |

## Per tool (all, by tokens desc)

| Server | Tool | v1 | Tokens |
|---|---|---|---|
| wandb | `query_wandb_tool` |  | 4289 |
| wandb | `query_weave_traces_tool` |  | 3640 |
| wandb | `create_wandb_report_tool` |  | 2229 |
| arxiv | `search_papers` | yes | 1068 |
| huggingface | `hf_fs` |  | 791 |
| wandb | `log_analysis_to_wandb` |  | 713 |
| huggingface | `gr1_z_image_turbo_generate` |  | 710 |
| wandb | `search_weave_agents_tool` |  | 687 |
| wandb | `list_artifact_versions_tool` | yes | 666 |
| wandb | `count_weave_traces_tool` |  | 653 |
| wandb | `query_weave_agent_spans_tool` |  | 647 |
| wandb | `get_run_history_tool` | yes | 645 |
| wandb | `list_wandb_automations_tool` |  | 633 |
| wandb | `get_weave_agent_span_stats_tool` |  | 604 |
| wandb | `compare_artifact_versions_tool` | yes | 512 |
| wandb | `list_weave_agents_tool` |  | 497 |
| wandb | `list_wandb_integrations_tool` |  | 471 |
| wandb | `get_artifact_details_tool` | yes | 426 |
| wandb | `resolve_trace_roots_tool` |  | 424 |
| huggingface | `hub_repo_details` | yes | 409 |
| wandb | `list_registries_tool` |  | 401 |
| wandb | `compare_runs_tool` | yes | 391 |
| wandb | `list_weave_agent_versions_tool` |  | 379 |
| wandb | `infer_trace_schema_tool` |  | 367 |
| wandb | `list_registry_collections_tool` |  | 367 |
| wandb | `summarize_evaluation_tool` |  | 343 |
| wandb | `diagnose_run_tool` | yes | 336 |
| wandb | `list_weave_agent_custom_attributes_tool` |  | 329 |
| huggingface | `hub_repo_search` | yes | 322 |
| wandb | `get_weave_agent_conversation_tool` |  | 322 |
| wandb | `probe_project_tool` | yes | 312 |
| wandb | `get_weave_agent_trace_tool` |  | 296 |
| wandb | `query_wandb_entity_projects` | yes | 293 |
| arxiv | `watch_topic` |  | 274 |
| wandb | `list_entities_tool` | yes | 205 |
| arxiv | `semantic_search` |  | 189 |
| arxiv | `export_citations` |  | 176 |
| arxiv | `read_paper` |  | 161 |
| arxiv | `download_paper` |  | 157 |
| huggingface | `dynamic_space` |  | 157 |
| arxiv | `get_paper_latex_section` |  | 147 |
| arxiv | `check_alerts` |  | 144 |
| arxiv | `get_abstract` | yes | 136 |
| arxiv | `get_paper_latex` |  | 127 |
| wandb | `search_wandb_docs_tool` | yes | 121 |
| arxiv | `list_paper_latex_sections` |  | 115 |
| arxiv | `list_papers` |  | 76 |
| arxiv | `citation_graph` |  | 73 |
| arxiv | `reindex` |  | 56 |
| huggingface | `hf_whoami` |  | 51 |

## Notes

- Hugging Face live names: `hub_repo_search` + `hub_repo_details`. Plan names `model_search`, `dataset_search`, `hf_doc_search` are gone.
- `hf_fs` is the obese HF schema (filesystem grammar + docs/papers). Excluded from v1.
- W&B Weave / GraphQL / reports / automations / registry are excluded from v1.
- arXiv v1 is `search_papers` + `get_abstract` only.
