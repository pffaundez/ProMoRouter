# P1 - Experiment Repair next steps

_Last updated: 2026-09-16_

Read `docs/experiment-status.md`, `docs/decisions.md`, and
`docs/methodology-scope.md` before continuing. Keep exactly one task in **Now**.

## Now

### T-010 — Generate canonical multi-seed result tables

- Description: Create a versioned summary artifact from the machine-readable
  outputs for Edge-GNN, GraphRouter-direct, Flat-MLP, and static baselines,
  reporting P, C, R, mean, standard deviation, and seed list.
- Objective: Replace manually copied or legacy paper numbers with one
  reproducible source and expose the near-tie between Flat-MLP and Edge-GNN.
- Files involved: `outputs/p0_router_edgegnn_qnorm/`,
  `outputs/p0_router_model_only_direct_qnorm/`,
  `outputs/p0_router_flat_mlp_qnorm/`,
  `outputs/p0_baselines_qnorm/`, summary script/output, paper tables.
- Dependencies: All five-seed outputs are present.
- Completion criterion: Every file is checked, rewards pass `R=P-lambda*C`, and
  the generated table includes all methods and lambdas.
- State: ready
- Priority: P0
- Blocker: none

## Next

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
- Flat-MLP five-seed sweep completed: 15 result rows and 15 model files, all
  with 121 test queries.
- Flat-MLP aggregates computed: R=0.5195, 0.3866, 0.3534 for lambdas 0.1,
  0.5, and 0.9 respectively.
- Fixed-pair identities stable across seeds.
- Offline/online and closed-pool scope documented; D-013 active.
- Flat 36-action no-graph baseline implemented; D-014 active.
