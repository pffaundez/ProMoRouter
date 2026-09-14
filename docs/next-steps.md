# P1 - Experiment Repair next steps

_Last updated: 2026-09-14_

Read docs/experiment-status.md, docs/decisions.md, and docs/methodology-scope.md before continuing. Keep exactly one task in Now.

## Now

### T-010 — Generate canonical multi-seed result tables

- Description: Create a versioned summary artifact from the 15 machine-readable seed outputs for Edge-GNN, GraphRouter-direct, and baselines, reporting P, C, R, mean, standard deviation, and seed list.
- Objective: Replace manually copied or legacy paper numbers with one reproducible source.
- Files: outputs/p0_router_edgegnn_qnorm/, outputs/p0_router_model_only_direct_qnorm/, outputs/p0_baselines_qnorm/, new summary artifact, paper tables.
- Dependencies: Five-seed outputs are present; no retraining required.
- Completion criterion: Every file is checked and all aggregates pass R=P-lambda*C validation.
- State: ready
- Priority: P0
- Blocker: none

## Next

### T-011 — Recreate routing-configuration ablation
State: pending. Priority: P1. Dependency: T-010. Blocker: identify exact ablation entry points.

### T-012 — Rewrite methodology and algorithm text
State: pending. Priority: P1. Dependency: T-010 and D-013. Blocker: none.

## Later

### T-013 — Correct README paths, defaults, and local-artifact instructions
State: deferred. Priority: P1. Dependency: T-010. Blocker: none.

### T-014 — Audit and isolate legacy pre-qnorm artifacts
State: deferred. Priority: P2. Dependency: T-013. Blocker: none.

## Blocked

### T-015 — Merge repair branch into main
State: blocked. Priority: P1. Blocker: review and explicit user approval.

### T-016 — Add a pre-routing cost predictor
State: blocked. Priority: P2. Blocker: separate research-scope decision; not current implementation.

## Completed (recent)

- Five-seed outputs verified for all three methods and lambda settings.
- Aggregated P/C/R values computed from machine-readable JSON files.
- Fixed-pair identities checked and stable across seeds.
- Offline/online and closed-pool scope documented in docs/methodology-scope.md and decision D-013.
