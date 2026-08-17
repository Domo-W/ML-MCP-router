# QLoRA failure gallery

Setup `slm_qlora` on the 265 test rows. Source: `data/eval/slm_qlora.scored.jsonl` joined to `data/sft/canonical.jsonl`.
Primary class is mutually exclusive (first match: invalid → wrong server → wrong tool → bad run_id → bad slots → inexact).
Headline misses are everything except exact-only optional drift.

## Overall

| Bar | n | Rate |
|---|---:|---:|
| Valid | 261 / 265 | 98.5% |
| Correct tool | 253 / 265 | 95.5% |
| Correct args | 244 / 265 | 92.1% |
| Exact | 219 / 265 | 82.6% |
| Headline misses | 25 | |
| Inexact optionals | 23 | |

## Primary class counts

| Class | n |
|---|---:|
| Invalid / extra prose | 4 |
| Wrong server | 2 |
| Wrong tool | 6 |
| Bad run_id | 1 |
| Bad slots | 12 |
| Inexact optionals | 23 |
| Exact / ok | 217 |

## Weak slices

The three slices called out after the bake-off. Rates are on that slice only.

| Slice | n | Rate |
|---|---:|---:|
| Collisions — correct tool | 39 | 84.6% |
| Collisions — correct args | 39 | 66.7% |
| `search_wandb_docs_tool` — correct args | 12 | 50.0% |
| `get_abstract` — correct tool | 10 | 80.0% |

## Takeaways

- 4 invalids, none extra prose — all parsed as JSON and failed `inputSchema`.
- 25 headline misses. 11 of 12 bad-slot rows are `query` drift (docs / Hub / paper frame words).
- Wrong server is rare (2), all collision rows.

## Invalid / extra prose

4 rows. None are extra prose — each parsed as JSON and failed that tool's `inputSchema`.

```
`ft-000250` · `invalid`

intent: Check the difference between runs ppmxie13 and ya1coj52 in paper-repl/dpo-reproduce.
gold: `compare_runs_tool` `{"entity_name":"paper-repl","project_name":"dpo-reproduce","run_id_a":"ppmxie13","run_id_b":"ya1coj52"}`
pred: `{"name":"compare_runs_tool","arguments":{"entity_name":"ppmxie13","project_name":"dpo-reproduce","run_id_a":"ya1coj52"}}`
why: schema: 'run_id_b' is a required property
```

```
`ft-000327` · `invalid`

intent: Does running ppmxie13 in paper-repl/dpo-reproduce lead to overfitting or NaN values?
gold: `diagnose_run_tool` `{"entity_name":"paper-repl","project_name":"dpo-reproduce","run_id":"ppmxie13"}`
pred: `{"name":"diagnose_run_tool","arguments":{"entity_name":"ppmxie13","project_name":"dpo-reproduce"}}`
why: schema: 'run_id' is a required property
```

```
`ft-000678` · `invalid` · collision=`hub_vs_wandb`

intent: Highest-rated phi-2
gold: `hub_repo_search` `{"query":"phi-2","repo_types":["model"]}`
pred: `{"name":"search_papers","arguments":{"query":"phi-2","sort":"rating"}}`
why: schema: Additional properties are not allowed ('sort' was unexpected); parsed `search_papers`
```

```
`ft-000684` · `invalid` · collision=`probe_vs_hub`

intent: Who's ahead in eval-sandbox/adapter-sweep?
gold: `probe_project_tool` `{"entity_name":"eval-sandbox","project_name":"adapter-sweep"}`
pred: `{"name":"compare_runs_tool","arguments":{"entity_name":"eval-sandbox","project_name":"adapter-sweep"}}`
why: schema: 'run_id_a' is a required property; parsed `compare_runs_tool`
```

## Wrong server

2 rows. Gold server and predicted server disagree.

```
`ft-000677` · `wrong_server` · collision=`hub_vs_wandb`

intent: What's the strongest TinyLlama 1.1B chat?
gold: `hub_repo_search` `{"query":"TinyLlama 1.1B chat","repo_types":["model"]}`
pred: `{"name":"probe_project_tool","arguments":{"entity_name":"huggingface","project_name":"TinyLlama_1.1B_chat"}}`
why: gold huggingface `hub_repo_search` vs pred wandb `probe_project_tool` (Hub model search vs W&B runs)
```

```
`ft-000736` · `wrong_server` · collision=`abstract_vs_search`

intent: Metadata for 2211.17192 (speculative decoding)
gold: `get_abstract` `{"paper_id":"2211.17192"}`
pred: `{"name":"hub_repo_details","arguments":{"repo_ids":["2211.17192"],"repo_type":"model"}}`
why: gold arxiv `get_abstract` vs pred huggingface `hub_repo_details` (known paper id vs paper search)
```

## Wrong tool

6 rows. Same server, wrong name.

```
`ft-000035` · `wrong_tool`

intent: What does the paper-repl/dpo-reproduce W&B project contain?
gold: `probe_project_tool` `{"entity_name":"paper-repl","project_name":"dpo-reproduce"}`
pred: `{"name":"search_wandb_docs_tool","arguments":{"query":"paper-repl/dpo-reproduce W&B project"}}`
why: gold `probe_project_tool` vs pred `search_wandb_docs_tool`
```

```
`ft-000086` · `wrong_tool`

intent: What is the val_loss and exact_match for run wo591x1u in eval-sandbox/adapter-sweep?
gold: `get_run_history_tool` `{"entity_name":"eval-sandbox","project_name":"adapter-sweep","run_id":"wo591x1u","keys":["val_loss","exact_match"]}`
pred: `{"name":"probe_project_tool","arguments":{"entity_name":"eval-sandbox","project_name":"adapter-sweep","run_id":"wo591x1u","keys":["val_loss","exact_match"]}}`
why: gold `get_run_history_tool` vs pred `probe_project_tool`
```

```
`ft-000102` · `wrong_tool`

intent: What is the reward and kl curve for the run rjfq21iq in heldout-lab/lora-ablation?
gold: `get_run_history_tool` `{"entity_name":"heldout-lab","project_name":"lora-ablation","run_id":"rjfq21iq","keys":["reward","kl"]}`
pred: `{"name":"probe_project_tool","arguments":{"entity_name":"heldout-lab","project_name":"lora-ablation","run_id":"rjfq21iq"}}`
why: gold `get_run_history_tool` vs pred `probe_project_tool`
```

```
`ft-000106` · `wrong_tool`

intent: What is the reward for the run qslojoli in heldout-lab/lora-ablation over time?
gold: `get_run_history_tool` `{"entity_name":"heldout-lab","project_name":"lora-ablation","run_id":"qslojoli","keys":["reward"]}`
pred: `{"name":"probe_project_tool","arguments":{"entity_name":"heldout-lab","project_name":"lora-ablation","run_id":"qslojoli"}}`
why: gold `get_run_history_tool` vs pred `probe_project_tool`
```

```
`ft-000706` · `wrong_tool` · collision=`diagnose_vs_history`

intent: Is training okay for wo591x1u in eval-sandbox/adapter-sweep?
gold: `diagnose_run_tool` `{"entity_name":"eval-sandbox","project_name":"adapter-sweep","run_id":"wo591x1u"}`
pred: `{"name":"probe_project_tool","arguments":{"entity_name":"eval-sandbox","project_name":"adapter-sweep","run_id":"wo591x1u"}}`
why: gold `diagnose_run_tool` vs pred `probe_project_tool` (health/overfit vs loss curve)
```

```
`ft-000734` · `wrong_tool` · collision=`abstract_vs_search`

intent: Summary of SimPO 2405.14734
gold: `get_abstract` `{"paper_id":"2405.14734"}`
pred: `{"name":"search_papers","arguments":{"query":"SimPO 2405.14734"}}`
why: gold `get_abstract` vs pred `search_papers` (known paper id vs paper search)
```

## Bad run_id

1 rows. Right tool; `run_id` / `run_id_a` / `run_id_b` does not match gold.

```
`ft-000116` · `bad_run_id`

intent: What are the train/loss and eval/reward_acc curves for the run ppmxie13 in paper-repl/dpo-reproduce?
gold: `get_run_history_tool` `{"entity_name":"paper-repl","project_name":"dpo-reproduce","run_id":"ppmxie13","keys":["train/loss","eval/reward_acc"]}`
pred: `{"name":"get_run_history_tool","arguments":{"entity_name":"ppmxie13","project_name":"dpo-reproduce","run_id":"train/loss","include_eval":"reward_acc"}}`
why: run_id gold="ppmxie13" pred="train/loss"
```

## Bad slots

12 rows. Right tool; a required/slot key other than run id is off.

```
`ft-000344` · `bad_slots`

intent: How do I use wandb.require service?
gold: `search_wandb_docs_tool` `{"query":"wandb.require service"}`
pred: `{"name":"search_wandb_docs_tool","arguments":{"query":"wandb.require"}}`
why: query gold="wandb.require service" pred="wandb.require"
```

```
`ft-000345` · `bad_slots`

intent: How do I use wandb.require service in my project?
gold: `search_wandb_docs_tool` `{"query":"wandb.require service"}`
pred: `{"name":"search_wandb_docs_tool","arguments":{"query":"wandb.require"}}`
why: query gold="wandb.require service" pred="wandb.require"
```

```
`ft-000667` · `bad_slots`

intent: How to use wandb.define_metric?
gold: `search_wandb_docs_tool` `{"query":"how to use wandb.define_metric"}`
pred: `{"name":"search_wandb_docs_tool","arguments":{"query":"wandb.define_metric"}}`
why: query gold="how to use wandb.define_metric" pred="wandb.define_metric"
```

```
`ft-000668` · `bad_slots`

intent: How do I log a confusion matrix?
gold: `search_wandb_docs_tool` `{"query":"log a confusion matrix"}`
pred: `{"name":"search_wandb_docs_tool","arguments":{"query":"log confusion matrix"}}`
why: query gold="log a confusion matrix" pred="log confusion matrix"
```

```
`ft-000675` · `bad_slots` · collision=`hub_vs_wandb`

intent: Top Yi 1.5 6B chat right now
gold: `hub_repo_search` `{"query":"Yi 1.5 6B chat","repo_types":["model"]}`
pred: `{"name":"hub_repo_search","arguments":{"query":"Top Yi 1.5 6B chat","repo_types":["model"]}}`
why: query gold="Yi 1.5 6B chat" pred="Top Yi 1.5 6B chat"
```

```
`ft-000676` · `bad_slots` · collision=`hub_vs_wandb`

intent: Trending Granite 3.1 2B instruct
gold: `hub_repo_search` `{"query":"Granite 3.1 2B instruct","repo_types":["model"]}`
pred: `{"name":"hub_repo_search","arguments":{"query":"Trending Granite 3.1 2B instruct","repo_types":["model"]}}`
why: query gold="Granite 3.1 2B instruct" pred="Trending Granite 3.1 2B instruct"
```

```
`ft-000693` · `bad_slots` · collision=`paper_vs_hub`

intent: ORPO paper weights
gold: `search_papers` `{"query":"ORPO"}`
pred: `{"name":"search_papers","arguments":{"query":"ORPO paper weights"}}`
why: query gold="ORPO" pred="ORPO paper weights"
```

```
`ft-000694` · `bad_slots` · collision=`paper_vs_hub`

intent: SimPO paper checkpoint
gold: `search_papers` `{"query":"SimPO"}`
pred: `{"name":"search_papers","arguments":{"query":"SimPO paper checkpoint"}}`
why: query gold="SimPO" pred="SimPO paper checkpoint"
```

```
`ft-000697` · `bad_slots` · collision=`paper_vs_hub`

intent: LLM as a judge paper weights
gold: `search_papers` `{"query":"LLM as a judge"}`
pred: `{"name":"search_papers","arguments":{"query":"LLM as a judge paper weights"}}`
why: query gold="LLM as a judge" pred="LLM as a judge paper weights"
```

```
`ft-000723` · `bad_slots` · collision=`docs_vs_probe`

intent: Teach me how to use wandb.define_metric
gold: `search_wandb_docs_tool` `{"query":"how to use wandb.define_metric"}`
pred: `{"name":"search_wandb_docs_tool","arguments":{"query":"wandb.define_metric"}}`
why: query gold="how to use wandb.define_metric" pred="wandb.define_metric"
```

```
`ft-000727` · `bad_slots` · collision=`docs_vs_probe`

intent: Teach me wandb.require service
gold: `search_wandb_docs_tool` `{"query":"wandb.require service"}`
pred: `{"name":"search_wandb_docs_tool","arguments":{"query":"wandb.require"}}`
why: query gold="wandb.require service" pred="wandb.require"
```

```
`ft-000735` · `bad_slots` · collision=`abstract_vs_search`

intent: Abstract only: Medusa 2401.10774
gold: `get_abstract` `{"paper_id":"2401.10774"}`
pred: `{"name":"get_abstract","arguments":{"paper_id":"Medusa 2401.10774"}}`
why: paper_id gold="2401.10774" pred="Medusa 2401.10774"
```

## Inexact optionals

23 rows have the right tool and required slots but differ on optionals (`sample_runs`, `max_projects`, extra `keys`, …). Not counted as headline misses. Showing 2.

```
`ft-000028` · `inexact_optional`

intent: What's the heldout-lab/lora-ablation W&B project look like with 10 sampled runs?
gold: `probe_project_tool` `{"entity_name":"heldout-lab","project_name":"lora-ablation","sample_runs":10}`
pred: `{"name":"probe_project_tool","arguments":{"entity_name":"heldout-lab","project_name":"lora-ablation","sampled_runs":10}}`
why: extra `sampled_runs`; missing `sample_runs`
```

```
`ft-000029` · `inexact_optional`

intent: What's in the heldout-lab/lora-ablation W&B project, and can you show me 3 runs to get a first look at the project's overall trends?
gold: `probe_project_tool` `{"entity_name":"heldout-lab","project_name":"lora-ablation","sample_runs":3}`
pred: `{"name":"probe_project_tool","arguments":{"entity_name":"heldout-lab","project_name":"lora-ablation"}}`
why: missing `sample_runs`
```

## Weak-slice appendix

Every headline miss in a collision row, a docs-args miss, or a `get_abstract` tool miss (18). Cards are in the class sections above.

| id | Class | Slice | Why |
|---|---|---|---|
| `ft-000344` | Bad slots | `search_wandb_docs_tool` | query gold="wandb.require service" pred="wandb.require" |
| `ft-000345` | Bad slots | `search_wandb_docs_tool` | query gold="wandb.require service" pred="wandb.require" |
| `ft-000667` | Bad slots | `search_wandb_docs_tool` | query gold="how to use wandb.define_metric" pred="wandb.define_metric" |
| `ft-000668` | Bad slots | `search_wandb_docs_tool` | query gold="log a confusion matrix" pred="log confusion matrix" |
| `ft-000675` | Bad slots | `hub_vs_wandb` | query gold="Yi 1.5 6B chat" pred="Top Yi 1.5 6B chat" |
| `ft-000676` | Bad slots | `hub_vs_wandb` | query gold="Granite 3.1 2B instruct" pred="Trending Granite 3.1 2B instruct" |
| `ft-000677` | Wrong server | `hub_vs_wandb` | gold huggingface `hub_repo_search` vs pred wandb `probe_project_tool` (Hub model search vs W&B runs) |
| `ft-000678` | Invalid / extra prose | `hub_vs_wandb` | schema: Additional properties are not allowed ('sort' was unexpected); parsed `search_papers` |
| `ft-000684` | Invalid / extra prose | `probe_vs_hub` | schema: 'run_id_a' is a required property; parsed `compare_runs_tool` |
| `ft-000693` | Bad slots | `paper_vs_hub` | query gold="ORPO" pred="ORPO paper weights" |
| `ft-000694` | Bad slots | `paper_vs_hub` | query gold="SimPO" pred="SimPO paper checkpoint" |
| `ft-000697` | Bad slots | `paper_vs_hub` | query gold="LLM as a judge" pred="LLM as a judge paper weights" |
| `ft-000706` | Wrong tool | `diagnose_vs_history` | gold `diagnose_run_tool` vs pred `probe_project_tool` (health/overfit vs loss curve) |
| `ft-000723` | Bad slots | `docs_vs_probe` | query gold="how to use wandb.define_metric" pred="wandb.define_metric" |
| `ft-000727` | Bad slots | `docs_vs_probe` | query gold="wandb.require service" pred="wandb.require" |
| `ft-000734` | Wrong tool | `abstract_vs_search` | gold `get_abstract` vs pred `search_papers` (known paper id vs paper search) |
| `ft-000735` | Bad slots | `abstract_vs_search` | paper_id gold="2401.10774" pred="Medusa 2401.10774" |
| `ft-000736` | Wrong server | `abstract_vs_search` | gold arxiv `get_abstract` vs pred huggingface `hub_repo_details` (known paper id vs paper search) |
