# P1 - Experiment Repair next steps

_Last updated: 2026-09-15_

Read `docs/experiment-status.md`, `docs/decisions.md`, and
`docs/methodology-scope.md` before continuing. Keep exactly one task in **Now**.

## Now

### T-018 — Run the Flat 36-action MLP baseline

- Description: Train and evaluate `train_router_flat_mlp_qnorm.py` on the
  complete 797-query dataset for seeds 1--5 and lambda values 0.1, 0.5, 0.9.
- Objective: Compare ProMoRouter against an equal-action-space scorer without
  graph message passing.
- Files involved: `train_router_flat_mlp_qnorm.py`,
  `data/router/*.pt`,
  `data/router/splits/qnorm_complete_seed{1..5}.json`, and
  `outputs/p1_router_flat_mlp_qnorm/`.
- Dependencies: Local embeddings and existing split manifests.
- Completion criterion: Five per-seed JSON result files and 15 model files;
  every test set has 121 queries and rewards satisfy `R=P-lambda*C`.
- Command:
  ```bash
  python train_router_flat_mlp_qnorm.py \\
    --data-path data/interaction_logs/grpp_il_v1/router_bipartite_qnorm_complete.jsonl \\
    --seeds 1 2 3 4 5 \\
    --output-dir outputs/p1_router_flat_mlp_qnorm
  ```
- State: ready
- Priority: P0
- Blocker: Embeddings must exist locally under `data/router/`.

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

- Five-seed Edge-GNN, GraphRouter-direct, and static baseline outputs verified.
- Multi-seed P/C/R aggregates computed for those methods.
- Fixed-pair identities stable across seeds.
- Offline/online and closed-pool scope documented; D-013 active.
- Flat 36-action MLP baseline implemented; D-014 active.
