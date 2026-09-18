# P1 - Experiment Repair next steps

_Last updated: 2026-09-18_

Read `docs/experiment-status.md`, `docs/decisions.md`, and
`docs/methodology-scope.md` before continuing. Keep exactly one task in **Now**.

## Now

### T-025 — Specify Edge-GNN v2 Stage A implementation contract

- Description: Convert D-026 into a code-level plan for exactly three
  controlled arms: GraphRouter-style message passing, matched no-message-
  passing, and the existing Flat-MLP.
- Objective: Isolate graph architectural benefit while holding the 36-action
  space and the complete repaired P0 protocol fixed.
- Dependencies: T-011 completed; D-026 active.
- Completion criterion: A reviewed machine-readable experiment manifest plus a
  precise topology/scorer/data-flow specification, leakage audit, expected
  outputs, seed-1 smoke gate, and five-seed aggregation contract. No training
  run is part of this task.
- Constraints: Preserve frozen P0; do not alter the current objective in Stage
  A; do not use realized reward, performance, cost, or tokens as features; keep
  inductive evaluation separate; do not implement Stage B or density variants.
- State: in_progress
- Priority: P1
- Blocker: None.

## Next

### T-020 — Verify candidate integration and generate description embeddings
State: completed. Priority: P0. Dependency: T-019.

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

- T-011 completed: deterministic seed-1 reproduction passed; the canonical
  five-seed 2x2 routing ablation validated 20 sources, 60 seed-level rows, and
  12 aggregates. It found no consistent top-k 5 or prompt--model-lattice
  advantage and remains separate from frozen P0.

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
