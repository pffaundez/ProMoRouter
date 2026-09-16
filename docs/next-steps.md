# P1 - Experiment Repair next steps

_Last updated: 2026-09-16_

Read `docs/experiment-status.md`, `docs/decisions.md`, and
`docs/methodology-scope.md` before continuing. Keep exactly one task in **Now**.

## Now

### T-019 — Prepare the inductive unseen-candidate protocol

- Description: Select and document one external model and one external prompt,
  add their static descriptions, and define held-out interaction files for
  unseen-model, unseen-prompt, and masked prompt-model-pair evaluation.
- Objective: Make the intended GraphRouter-style generalization test
  executable without confusing it with the existing closed-pool results.
- Files involved: `configs/model_descriptions.json`,
  `configs/prompt_strategies.yaml`,
  `scripts/build_node_description_embeddings.py`, new held-out datasets,
  `data/router/inductive_embeddings/`, and experiment docs.
- Dependencies: Availability of external model/prompt descriptions and their
  test interaction outcomes.
- Completion criterion: A written protocol identifies the 10 models and 5
  prompts, specifies train/test candidate visibility, and validates that unseen
  candidates have no training interactions.
- State: blocked
- Priority: P0
- Blocker: The external tenth model and fifth prompt, plus evaluation outcomes,
  have not yet been selected or supplied.

## Next

### T-020 — Implement description-conditioned inductive Edge-GNN
State: pending. Priority: P0. Dependency: T-019.

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
- Inductive 10-model/5-prompt evaluation protocol adopted as D-015.


## Protocol references

Before implementing T-019/T-020, use `docs/cost-information-contract.md` and `docs/inductive-evaluation-protocol.md`. Candidate-specific choices and unseen outcomes remain unconfirmed.
