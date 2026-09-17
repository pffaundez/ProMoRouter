# P1 - Experiment Repair next steps

_Last updated: 2026-09-16_

Read `docs/experiment-status.md`, `docs/decisions.md`, and
`docs/methodology-scope.md` before continuing. Keep exactly one task in **Now**.

## Now

### T-022 — Generate the complete inductive evaluation dataset

- Description: Evaluate the 12-model × 6-prompt pool on all persisted test qids using the unified Transformers backend.
- Objective: Produce held-out outcomes for unseen models, unseen prompts, and masked prompt–model combinations under a single comparable protocol.
- Files involved: `experiments/generate_inductive_transformers.py`, `configs/inductive_candidates.yaml`, `outputs/inductive_inductive.jsonl`.
- Dependencies: corrected smoke test; cached model access; sufficient GPU/storage.
- Completion criterion: 72 actions per query are written without schema or runtime errors, with response, raw samples, performance, token counts, and latency fields.
- Command: `python experiments/generate_inductive_transformers.py --source-qnorm ... --split-manifest ... --output outputs/inductive_full.jsonl`
- State: pending
- Priority: P0
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

### T-010 — Generate canonical closed-pool result tables
State: pending. Priority: P0. Dependency: existing five-seed outputs.

### T-011 — Recreate routing-configuration ablation
State: pending. Priority: P1. Dependency: T-010.

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
