# P1 - Experiment Repair next steps

_Last updated: 2026-09-18_

Read `docs/experiment-status.md`, `docs/decisions.md`, and
`docs/methodology-scope.md` before continuing. Keep exactly one task in **Now**.

## Now

### T-011 — Recreate routing-configuration ablation

- Description: Recreate the routing-configuration ablation with the repaired population, shared manifests, reward contract, and no post-execution routing features.
- Objective: Isolate which graph/routing components contribute beyond the Flat-MLP baseline.
- Dependencies: T-010 completed.
- Completion criterion: A documented ablation matrix, validated commands, and machine-readable outputs using the repaired protocol.
- Configuration: D-023 fixes a 2x2 matrix over `edge_top_k={3,5}` and the prompt-model lattice `{off,on}` while holding the 36-action space and all other protocol elements fixed.
- Current integrity gate: run `topk3_observed` twice with `--deterministic` into distinct output directories and require identical result JSON metrics before replacing the original nondeterministic gate.
- Gate command after determinism passes: `python scripts/run_p0_routing_ablation.py --seeds 1 --artifact-root ~/repos/graph-router-2 --device cuda:0 --overwrite`.
- Gate aggregation: `python analysis/aggregate_p0_routing_ablation.py --seeds 1 --output-json analysis/p0_routing_ablation_seed1.json --output-csv analysis/p0_routing_ablation_seed1.csv`.
- Full sweep command after the gate passes: `python scripts/run_p0_routing_ablation.py --artifact-root ~/repos/graph-router-2 --device cuda:0`.
- Full aggregation: `python analysis/aggregate_p0_routing_ablation.py`.
- Prepared: manifest, runner, aggregator, compilation, and dry runs passed; experiment-host execution is pending.
- State: in_progress
- Priority: P1
- Blocker: None.

## Next

### T-020 — Verify candidate integration and generate description embeddings
State: in_progress. Priority: P0. Dependency: T-019.

- Description: Validate the three Hugging Face model identifiers and prompt execution contracts, then generate embeddings under `data/router/inductive_embeddings/` without overwriting P0 artifacts.
- Completion criterion: All five unseen candidates pass metadata/integration checks and embedding files contain exactly the 3 unseen model IDs and 2 unseen prompt IDs.
- Command: `python analysis/validate_inductive_candidates.py` followed by the model-loading smoke test.
- Completed: embeddings generated with `sentence-transformers/all-MiniLM-L6-v2`; 3 unseen model vectors and 2 unseen prompt vectors, all dimension 384.
- Completed: `analysis/validate_inductive_candidates.py` passed exact pool, non-overlap, and artifact checks.
- Completed: `AutoConfig.from_pretrained` succeeded for all three unseen model IDs (`llama`, `olmo2`, `phi3`).
- Completed: all three unseen models passed minimal Transformers generation smoke tests.
- Completed: templates for `fewshot` and `self_consistency` added to `configs/prompt_templates.yaml`.
- Completed: budgets fixed at 2 demonstrations for `fewshot` and 3 samples with majority aggregation for `self_consistency`.
- Completed: deterministic execution config added at `configs/inductive_prompt_execution.yaml` (2 training shots; 3 samples, temperature 0.7, top-p 0.95, 256-token cap).
- Blocker: The qnorm log has no response/completion text; a separate benchmark demonstration source is required before held-out generation.

### T-012 — Rewrite methodology and algorithm text
State: pending. Priority: P1. Dependency: T-010 and D-013.

## Later

### T-013 — Correct README paths, defaults, and local-artifact instructions
State: deferred. Priority: P1. Dependency: T-010.

### T-014 — Audit and isolate legacy pre-qnorm artifacts
State: deferred. Priority: P2. Dependency: T-013.

## Blocked

### T-015 — Merge repair branch into `main`
State: blocked. Priority: P1. Blocker: review and explicit user approval.

### T-016 — Add a pre-routing cost predictor
State: blocked. Priority: P2. Blocker: separate research-scope decision.

## Completed (recent)

- Five-seed Edge-GNN, GraphRouter-direct, static-baseline, and Flat-MLP
  outputs verified.
- Flat-MLP added as equal-action-space no-graph baseline; D-014 active.
- Inductive 12-model/6-prompt evaluation protocol adopted as D-016.
- Confirmed candidate manifest created at `configs/inductive_candidates.yaml`.


## Protocol references

Before implementing T-019/T-020, use `docs/cost-information-contract.md` and `docs/inductive-evaluation-protocol.md`. Candidate IDs are confirmed by D-016; held-out interaction outcomes and cost-estimation details remain to be produced.

- Update: first smoke run exposed and fixed over-eager loading of unselected tasks.


- T-021 completed: corrected smoke test validated final-answer extraction and normalized self-consistency voting; the zero score was confirmed as a genuine model error against the HotpotQA gold.


- T-022 completed: full inductive generation produced 8,712 rows; two truncations were identified and regenerated successfully.


- T-023 completed: merged both targeted repair rows and produced `outputs/inductive_full_corrected.jsonl` with 8,712 validated rows.

- T-024 completed: added a reproducible inductive aggregator and generated
  `analysis/inductive_aggregates.json`, with seen/unseen model, prompt, novelty
  quadrant, task, and prompt-model summaries. Confirmed that no separate
  seen-seen masked-pair manifest is available; no such result was invented.

- T-010 completed: ran the canonical P0 aggregator on `morel`; validated 20
  source files and produced JSON, CSV, and LaTeX five-seed tables for nine
  methods without mixing inductive results.
