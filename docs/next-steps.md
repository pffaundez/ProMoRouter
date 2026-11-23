# P1 - Experiment Repair next steps

_Last updated: 2026-11-23_

Read `docs/experiment-status.md`, `docs/decisions.md`, and
`docs/methodology-scope.md` before continuing. Keep exactly one task in **Now**.

## Now

### T-037 — Review the completed public-artifact GraphRouter reference

- State: implementation, 52 CPU tests and a real 1000-epoch run completed.
- Read docs/graphrouter-reference.md and D-039.
- R rounds to 0.207, but all conditions select the same known model, with
  zero useful incremental incorporation. Do not call scalar agreement proof
  of the authors' complete adaptation result.
- Canonical artifacts: analysis/graphrouter_reference/seed1881_balance/.
- No prompt extension or new training is queued. Optional Morel reproduction
  uses scripts/run_graphrouter_reference.sh only if explicitly desired.
- Older Next/Later items below remain historical and inactive.

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

- T-025 completed: froze the Edge-GNN v2 Stage A topology, three-arm control
  matrix, feature contract, deterministic gates, result schema, and aggregation
  requirements before implementation.

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


## Historical queue reconciliation — 2026-11-20

T-026 is removed from Now and archived below as its original queue snapshot.
Stage A implementation and aggregates exist, but its full completion checks were
not rerun in this session. D-028 changes the active research direction; do not
restart Stage A implementation from the stale queue. T-027 is the sole Now task.
Earlier T-012/T-020 and external-candidate references are retained as history and
must be reconciled with D-028 before execution.

### Archived T-026 snapshot

### T-026 — Implement Edge-GNN v2 Stage A and integrity tests

- Description: Implement only the three arms and validation gates frozen in
  `docs/edgegnn-v2-stage-a.md` and `configs/edgegnn_v2_stage_a.json`.
- Objective: Produce a testable, deterministic Stage A implementation without
  changing the frozen protocol or running the five-seed experiment.
- Dependencies: T-025 completed; D-026 and D-027 active.
- Completion criterion: Static/synthetic topology and leakage tests pass; all
  arms pass a CPU smoke test; runner and aggregator dry runs validate expected
  paths and cardinalities. No GPU training sweep is part of this task.
- Constraints: Independent per-query graphs; matched full-MP/self-only scorer
  with equal depth, activation and normalization;
  shared objective; no reward-selected topology; no realized outcomes as
  features; new output namespace; preserve P0 and inductive artifacts.
- State: in_progress
- Priority: P1
- Blocker: None.



T-027 completed by the implementation contract in docs/support-graph-pilot.md.

T-028 development pilot completed and inspected: no incorporation gain and
selfcheck selected throughout test. T-029 is the sole next action.

T-029 completed: real-checkpoint audit interpreted; see D-030 for evidence.

T-030 completed: cosine aggregation did not improve incorporation or R_qnorm.


T-031 completed: official GraphRouter control selects Qwen but shows no primary
reward improvement over seen-only; k80 increases selections and worsens the
point estimate. See the dated status entry; confidence intervals include zero.
T-032 is the sole current experimental action. Older queued methodology work
remains deferred until this bounded intervention is interpreted.

T-032 completed: reject-all selected; AUC criteria passed but no positive
incorporation gain. The protocol stopped before external evaluation. See D-033
and docs/memoria-intervenciones-unseen.md. Historical pending-run statements
above are dated snapshots, superseded by this closure.

T-033 completed: scope clarified and documented in D-034. Closed interventions
do not answer or reject the pending official-encoder joint experiment.

T-034 completed: official-encoder joint implementation and synthetic integrity
checks passed. No real-data result exists yet. T-035 is the sole current task.

T-035 completed: official joint incorporation gain +0.02976946, with conditional
CI including zero and all 121 calibrated selections Qwen/decompose. See D-035.
T-036 is the sole current task; prior pending-run statements are historical.

T-036 engineering is complete; only the final no-training Morel calculation
and its closure record remain. No new research branch or GPU run is queued.

T-036 completed and current stage closed under D-037. No experimental task
remains active; earlier pending-run and comparison statements are historical.

T-037 reference implementation and full CPU execution completed under D-039.
No automatic continuation into prompts or hyperparameter search follows.
