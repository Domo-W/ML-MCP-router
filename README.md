# ML MCP router

A fine-tuned 3B that maps a researcher question to one W&B, Hugging Face, or arXiv tool call, then runs that call from a Cursor MCP sidecar.

**Purpose.** Keep the frontier model as the planner. Give it one tool (`route_ml_tools`) instead of 11 MCP schemas. A local SLM chooses `{name, arguments}`, validates against JSON Schema, and executes the official server.

**Intended benefit.** Match prompted-3B quality without stuffing catalogs into the prompt (~117 tokens/request vs ~28k), and hide unused tools from Cursor Agent. In this host the win is smaller than the original 4k–27k claim: about 3k when the 11 schemas would have been inspected, plus a one-tool menu.

**Stack.** `Qwen2.5-3B-Instruct` · TRL SFT · PEFT QLoRA · schema-sampled synthetic data · held-out slot split · GGUF Q4_K_M · Ollama / llama.cpp · Cursor MCP

## Results

Same 265 held-out intents. Base is `Qwen2.5-3B-Instruct`.

| Setup | Valid | Tool | Args | Exact | Tokens/req |
|---|---:|---:|---:|---:|---:|
| Prompted, names only | 33.2% | 14.0% | 7.9% | 7.2% | 117 |
| Prompted + all 50 schemas (instr last) | 56.2% | 50.6% | 33.6% | 22.3% | 28,101 |
| **QLoRA, 1-line prompt** | **98.5%** | **95.5%** | **92.1%** | **82.6%** | **117** |
| GGUF Q4_K_M (Ollama) | 97.7% | 91.3% | 87.5% | 75.5% | 117 |

QLoRA beat a schema-stuffed 3B at the cheap prompt’s token cost. GGUF keeps most of that and serves in ~400 ms.

In Cursor, quality tied Claude + the 11 live tools on a small licensed prompt set. Cursor already refuses to keep the 27k all-tool dump in context (short index; oversized dumps leave the prompt).

## How it was built

- **Catalog.** Frozen 11 tools from live `tools/list` (7 W&B run/docs, 2 Hub, 2 arXiv). No Weave, GraphQL, `hf_fs`, or paper download.
- **Data.** MCPTune-style synthesis: sample arguments from each tool’s JSON Schema, write an intent that licenses those slots, then add collision (hard-negative) rows. 746 validated examples (481 train / 265 test), split by held-out entity, run, and repo ids — not a random shuffle.
- **Train.** TRL + PEFT QLoRA, 2 epochs, assistant-only loss on the tool-call JSON. Logged to W&B. 16-bit LoRA did not fit a 10 GB RTX 3080.
- **Eval.** Valid JSON, correct tool, correct arguments, exact match. Failure gallery in `data/eval/failure_gallery.md`.
- **Serve.** Merge adapter → Q4_K_M GGUF. The sidecar validates and executes (HF/W&B HTTP MCP, arXiv stdio) via Ollama or `llama-server`.

## Limitations

- Not a general MCP router. The menu is this 11-tool catalog.
- Weak slices: collisions, `query` drift, `get_abstract`, and bare run ids with no entity/project (the model invents slots).
- GGUF drops a bit vs the PEFT adapter (expected).
- Upstream servers differ in auth and transport, so this is hard to customize for other people.
- Cursor token accounting mixes index, inspect, file-spill, and conversation. Do not quote the 27k figure as a Cursor win.

## Why this is not a product

The original claim was that a specialist sidecar would save thousands of tokens by hiding full MCP catalogs from the host. Mid-project measurements showed Cursor already avoids most of that cost: it keeps a short tool index, fetches schemas on inspect, and writes oversized dumps to a file instead of keeping them in the prompt. The 4k–27k figure is a full-schema paste cost, not a Cursor cost.

That makes a general product on this design a poor fit. Hosts that still inline every schema would see the bake-off numbers; Cursor does not. The remaining use is personal: a one-tool menu, local routing, and about 3k fewer tokens when the 11 v1 schemas would otherwise be inspected.

## Inspirations

[TinyAgent](https://github.com/SqueezeAILab/TinyAgent), [MCPTune](https://github.com/McGill-NLP/MCPTune) (args-first synthesis), [MCPScout](https://www.npmjs.com/package/@mcp-scout/mcp-scout) (search / describe / call instead of dumping schemas), tool-smith, plus the official W&B, Hugging Face, and arXiv MCP servers. Novelty is this toolchain and an honest bake-off, not the architecture.

## Use it locally

`examples/mcp.ollama.json` / `examples/mcp.llama.json`. Keep Cursor Agent on a frontier model; do not set this LoRA as the Agent. Hide the raw W&B / HF / arXiv tools so Claude only sees `route_ml_tools`.
