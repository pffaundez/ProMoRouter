# P1 - Experiment Repair next steps

_Last updated: 2026-09-16_

Read `docs/experiment-status.md`, `docs/decisions.md`, and
`docs/methodology-scope.md` before continuing. Keep exactly one task in **Now**.

## Now

### T-019 — Create the confirmed inductive candidate manifest

- Description: Record the three unseen model IDs and two unseen prompt IDs in
  `configs/inductive_candidates.yaml`, with descriptions, source links, and
  integration metadata.
- Objective: Turn D-016 into a reproducible input contract without changing
  the closed-pool P0/P1 datasets.
- Files involved: `configs/inductive_candidates.yaml`,
  `configs/model_descriptions.json`, `configs/prompt_strategies.yaml`,
  and `docs/inductive-evaluation-protocol.md`.
- Dependencies: D-016; verify model loading and license metadata before
  execution.
- Completion criterion: The manifest validates exactly 12 total model IDs and
  6 total prompt IDs, with 9/4 seen and 3/2 unseen, and no unseen ID appears
  in the training population.
- Command: `python analysis/validate_inductive_candidates.py`
- State: completed
- Priority: P0
- Blocker: None.
- Completed: `configs/inductive_candidates.yaml` created with 12 models total (9 seen, 3 unseen) and 6 prompts total (4 seen, 2 unseen). License and backend checks remain pending.

## Next

### T-020 — Verify candidate integration and generate description embeddings
State: pending. Priority: P0. Dependency: T-019.

- Description: Validate the three Hugging Face model identifiers and prompt execution contracts, then generate embeddings under `data/router/inductive_embeddings/` without overwriting P0 artifacts.
- Completion criterion: All five unseen candidates pass metadata/integration checks and embedding files contain exactly the 3 unseen model IDs and 2 unseen prompt IDs.
- Command: `python scripts/build_inductive_description_embeddings.py --help` (then run it in the active venv).

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
