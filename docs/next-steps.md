# P1 - Experiment Repair next steps

_Last updated: 2026-09-16_

Read `docs/experiment-status.md`, `docs/decisions.md`, and
`docs/methodology-scope.md` before continuing. Keep exactly one task in **Now**.

## Now

### T-018 — Complete the Flat 36-action MLP sweep (seeds 2--5)

- Description: Run `train_router_flat_mlp_qnorm.py` for seeds 2, 3, 4, and 5
  using the complete dataset, canonical embeddings, and persisted manifests.
- Objective: Obtain multi-seed variance for the equal-action-space no-graph
  baseline before comparing it with Edge-GNN.
- Files involved: `train_router_flat_mlp_qnorm.py`,
  `data/router/splits/qnorm_complete_seed{2..5}.json`, and
  `outputs/p1_router_flat_mlp_qnorm/`.
- Dependencies: Seed-1 run completed; local embeddings available.
- Completion criterion: Four additional per-seed JSON files and 12 model files;
  every test set has 121 queries; each reward satisfies `R=P-lambda*C`.
- Command:
  ```bash
  python train_router_flat_mlp_qnorm.py \\
    --data-path ~/repos/graph-router-2/data/interaction_logs/grpp_il_v1/router_bipartite_qnorm_complete.jsonl \\
    --seeds 2 3 4 5 \\
    --device cuda:0 \\
    --output-dir outputs/p1_router_flat_mlp_qnorm
  ```
- State: in_progress
- Priority: P0
- Blocker: none

## Next

### T-010 — Generate canonical multi-seed result tables
State: pending. Priority: P0. Dependency: T-018.

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

- Five-seed outputs verified for Edge-GNN, GraphRouter-direct, and static
  baselines.
- Flat-MLP seed 1 completed with 121 test queries for all three lambdas.
- Offline/online and closed-pool scope documented; D-013 active.
- Flat 36-action no-graph baseline implemented; D-014 active.
