# ml-mcp-router

Stdio MCP with one tool, `route_ml_tools({intent})`. A small LoRA/GGUF
picks the W&B, Hugging Face, or arXiv call; this package validates it
and executes it.

Does **not** ship the GGUF, `llama-server`, or Transformers/PEFT.

```bash
uvx --from . ml-router --smoke
```

Two `mcp.json` blocks are in `examples/`.
