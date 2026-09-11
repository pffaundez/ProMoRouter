# P1 - Experiment Repair next steps

_Last updated: 2026-09-11_

This queue is derived from `docs/experiment-status.md`,
`docs/decisions.md`, and a repository check of
`fix/p0-routing-integrity`. Keep exactly one task in **Now**. Move tasks
between sections after significant changes and retain only a short completed
history.

## Continuation contract

A new working session should read, in order:

1. `docs/experiment-status.md`;
2. `docs/decisions.md`;
3. this file.

**Single next action:** execute **T-006**. T-007 follows on the same manifest;
do not run multiple seeds before T-008 inspects the complete seed-1 comparison.

## Verified facts affecting the queue

- The complete joint dataset has been validated locally with 797 queries and
  36 unique actions per query.
- `analysis/build_model_only_router_dataset.py` now accepts `--qid-source`,
  filters to that exact qid population, and rejects duplicate/mismatched qids,
  empty outputs, missing rewards, and non-nine-model candidate sets.
- `router_model_only_qnorm_complete.jsonl` was generated on the experiment
  host with 797 queries, nine models per query, and exact allowlist validation.
- `train_router_model_only_direct_qnorm.py` now supports `--build-only` and
  validates exact qid equality, nine unique expected models, direct prompt
  policy, and reward presence before exiting without training.
- The static evaluator and both learned routers can reuse a persisted split
  manifest, but their input populations must first be identical.
- `router_model_only_direct_qnorm_complete.jsonl` was generated with 797
  queries, zero skipped direct queries, and 797 candidates for each of the nine
  models.
- `analysis/validate_aligned_router_inputs.py` now provides one reproducible
  cross-dataset population, candidate, task, cost, and reward check.
- Cross-input validation passed on the experiment host: all three inputs
  contain the same 797 queries and report zero population, candidate, task,
  cost, or reward errors.
- Edge-GNN seed 1 completed for all three lambdas on 121 test queries. The
  printed P/C/R values and action counts are recorded in T-005 below.
- The lambda=0.1 run selected `selfcheck` for all 121 test queries; this is an
  observed policy pattern to inspect in T-008, not an automatic failure.

## Now

### T-006 — Run GraphRouter-direct on the same seed-1 manifest

- **Description:** Train/evaluate the repaired fixed-direct model-only router
  with the aligned direct dataset and the manifest created/reused by T-005.
- **Objective/reason:** Obtain a leakage-free adaptive-model comparison.
- **Files involved:** `train_router_model_only_direct_qnorm.py`, aligned
  direct data, embeddings, `qnorm_seed1.json`, direct-router outputs.
- **Dependencies:** T-004 and T-005 (completed).
- **Completion criterion:** Three lambda runs finish on the identical test qids;
  result JSON exists; scorer inputs remain query/model embeddings only.
- **Validation command:**
  ```bash
  python train_router_model_only_direct_qnorm.py \
    --router-data data/interaction_logs/grpp_il_v1/router_model_only_direct_qnorm_complete.jsonl \
    --source-data data/interaction_logs/grpp_il_v1/router_bipartite_qnorm_complete.jsonl \
    --seeds 1 \
    --split-manifest data/router/splits/qnorm_complete_seed1.json
  ```
- **State:** ready
- **Priority:** P0
- **Blocker:** none

## Next

### T-007 — Evaluate static baselines on the same seed-1 manifest

- **Description:** Run fixed and oracle baselines with the aligned averaged
  model-only and complete joint datasets.
- **Objective/reason:** Produce directly comparable non-learned references.
- **Files involved:** `analysis/evaluate_qnorm_baselines_on_test.py`, aligned
  datasets, `qnorm_seed1.json`, baseline output JSON.
- **Dependencies:** T-004 and T-005.
- **Completion criterion:** All baseline rows use the manifest's test qids;
  each reports the expected test count; output JSON is created.
- **Validation command:**
  ```bash
  python analysis/evaluate_qnorm_baselines_on_test.py \
    --seed 1 \
    --model-only-path data/interaction_logs/grpp_il_v1/router_model_only_qnorm_complete.jsonl \
    --bipartite-path data/interaction_logs/grpp_il_v1/router_bipartite_qnorm_complete.jsonl \
    --split-manifest data/router/splits/qnorm_complete_seed1.json
  ```
- **State:** queued
- **Priority:** P0
- **Blocker:** T-005 completed; T-006 is not required for static execution

### T-008 — Inspect the seed-1 integrity gate

- **Description:** Compare query counts, manifests, reward arithmetic, selected
  prompt/model distributions, and any fixed-action collapse across T-005 to
  T-007.
- **Objective/reason:** Decide whether the repaired pipeline is sound enough
  for the multi-seed sweep.
- **Files involved:** Seed-1 JSON outputs and split manifest; status and
  decision documents.
- **Dependencies:** T-005, T-006, and T-007.
- **Completion criterion:** Written pass/fail conclusion with evidence; all
  methods share test qids; no algebra errors; suspicious collapse is explained
  or converted into a blocking defect.
- **Validation command:** No single command yet; inspect machine-readable
  outputs and record the checks in `docs/experiment-status.md`.
- **State:** queued
- **Priority:** P0
- **Blocker:** Seed-1 outputs do not yet exist

## Later

### T-009 — Run the repaired five-seed sweep

- **Description:** Execute learned routers and static evaluations for seeds
  1--5 with one shared manifest per seed.
- **Objective/reason:** Produce variability-aware evidence for the long paper.
- **Files involved:** Trainers, evaluator, manifests, output directories.
- **Dependencies:** T-008 must pass.
- **Completion criterion:** Complete outputs for five seeds and three lambdas,
  with matching populations per seed and no validation failures.
- **Validation command:** Run the seed-specific commands established in
  T-005--T-007 for seeds 1 through 5.
- **State:** deferred
- **Priority:** P1
- **Blocker:** Seed-1 gate has not passed

### T-010 — Regenerate all P/C/R tables from outputs

- **Description:** Produce machine-derived means, dispersion, and table rows
  without manually copying historical values.
- **Objective/reason:** Replace currently untrusted paper tables.
- **Files involved:** Result JSONs; summary/table-generation scripts; paper
  tables.
- **Dependencies:** T-009.
- **Completion criterion:** Every reported value traces to output files and
  satisfies reward arithmetic; means and standard deviations are reproducible.
- **Validation command:** To be defined with the table-generation script.
- **State:** deferred
- **Priority:** P1
- **Blocker:** Repaired multi-seed outputs unavailable

### T-011 — Recreate the routing-configuration ablation

- **Description:** Rerun fixed-prompt/adaptive-model,
  adaptive-prompt/fixed-model, and full-joint configurations with aligned data
  and correct rewards.
- **Objective/reason:** Test the paper's central joint-routing claim.
- **Files involved:** Ablation trainers/configuration, manifests, result tables.
- **Dependencies:** T-008; preferably T-009.
- **Completion criterion:** Same datasets/splits, multi-seed statistics, and
  machine-derived `R=P-lambda*C`.
- **Validation command:** Pending exact ablation entry points.
- **State:** deferred
- **Priority:** P1
- **Blocker:** Repaired core pipeline not yet validated

### T-012 — Rewrite methodology to match the implementation

- **Description:** Correct graph construction, inference-time connectivity,
  scorer inputs, offline reward role, and actual loss terms in the paper.
- **Objective/reason:** Remove contradictions between paper, rebuttal, and code.
- **Files involved:** Paper source; algorithm and methodology sections.
- **Dependencies:** Implementation contract in D-003, D-005, and seed-1
  findings where relevant.
- **Completion criterion:** Every methodological claim maps to inspected code;
  no realized outcome is described as a routing-time input.
- **Validation command:** Manual code-to-paper checklist.
- **State:** deferred
- **Priority:** P1
- **Blocker:** None technically; deferred behind experimental integrity

### T-013 — Correct README paths, defaults, and repaired commands

- **Description:** Remove `interaction_logs_v1` references, document the
  complete-only and model-only build workflow, and ensure every shown CLI
  option exists.
- **Objective/reason:** Make third-party reproduction follow the repaired
  pipeline.
- **Files involved:** `README.md`; relevant CLI scripts.
- **Dependencies:** T-001 through T-004 so final filenames/interfaces are
  stable.
- **Completion criterion:** All documented paths and commands match repository
  code and the chosen complete-only workflow.
- **Validation command:** Execute or dry-run each documented preprocessing
  command.
- **State:** deferred
- **Priority:** P1
- **Blocker:** Builder interfaces are not finalized

### T-014 — Audit and isolate legacy pre-qnorm artifacts

- **Description:** Find all references to pre-qnorm scripts/data before
  deciding whether to relocate or remove them.
- **Objective/reason:** Reduce third-party ambiguity without deleting required
  provenance.
- **Files involved:** Legacy scripts, README, pre-qnorm dataset names.
- **Dependencies:** Repaired workflow documented and validated.
- **Completion criterion:** Reference inventory exists and each legacy artifact
  has an explicit keep/move/remove disposition.
- **Validation command:**
  ```bash
  rg -n "train_clean_lambdas|router_model_only\.jsonl|router_bipartite\.jsonl" .
  ```
- **State:** deferred
- **Priority:** P2
- **Blocker:** Core repaired workflow not finalized

## Blocked

### T-015 — Merge the repair branch into main

- **Description:** Merge `fix/p0-routing-integrity` into `main`.
- **Objective/reason:** Publish the validated repaired pipeline as the primary
  repository state.
- **Files involved:** Entire repair branch.
- **Dependencies:** At minimum T-008 and explicit user approval.
- **Completion criterion:** Reviewed merge/PR completed and `main` passes the
  integrity workflow.
- **Validation command:** `git status -sb` and branch/CI checks after merge.
- **State:** blocked
- **Priority:** P1
- **Blocker:** Seed-1 integrity gate has not passed; merge authorization has not
  been given.

### T-016 — Finalize the cost definition for the long paper

- **Description:** Decide whether query-normalized cost remains the final
  deployment-cost formulation and define its stated limitations.
- **Objective/reason:** Stabilize the scientific claim and paper notation.
- **Files involved:** Paper source, README, decision log, possibly data
  preparation.
- **Dependencies:** User/research decision informed by repaired results.
- **Completion criterion:** Explicit decision recorded in
  `docs/decisions.md` and reflected consistently in code/paper.
- **Validation command:** None.
- **State:** blocked
- **Priority:** P1
- **Blocker:** Requires user confirmation; current use of qnorm is confirmed
  for repair experiments but not yet as the final long-paper definition.

### T-017 — Add a pre-routing cost predictor

- **Description:** Decide whether to design and evaluate an estimated-cost
  feature available before routing.
- **Objective/reason:** Potentially add cost awareness without outcome leakage.
- **Files involved:** New predictor/data pipeline, router scorer, experiments,
  paper.
- **Dependencies:** Explicit user decision and a separate evaluation design.
- **Completion criterion:** Either a scoped approved experiment or an explicit
  decision not to pursue it.
- **Validation command:** None until designed.
- **State:** blocked
- **Priority:** P2
- **Blocker:** Not part of the confirmed repaired implementation and not
  authorized as current scope.

## Completed

### T-005 — Run the Edge-GNN seed-1 integrity experiment

- **Description:** Train/evaluate Edge-GNN for seed 1 using the complete joint
  dataset and create/reuse `qnorm_seed1.json`.
- **Objective/reason:** Establish the first repaired learned-router result.
- **Files involved:** `train_router_edgegnn_qnorm.py`, complete joint data,
  embeddings, split manifest, `outputs/router_edgegnn_qnorm/`.
- **Dependencies:** T-004 (completed); verified embeddings.
- **Completion criterion:** All three lambda runs finish; result JSON and model
  files exist; test query count matches the manifest; stored output
  `R = P-lambda*C` within tolerance.
- **Validation command:**
  ```bash
  python train_router_edgegnn_qnorm.py \
    --data-path data/interaction_logs/grpp_il_v1/router_bipartite_qnorm_complete.jsonl \
    --split-manifest data/router/splits/qnorm_complete_seed1.json \
    --output-dir outputs/p0_router_edgegnn_qnorm \
    --seed 1
  ```
- **State:** completed
- **Priority:** P0
- **Blocker:** none
- **Observed result:** 121 test queries for each lambda. Exact printed metrics:
  lambda=0.1: P=0.498273, C=0.389587, R=0.459314;
  lambda=0.5: P=0.417263, C=0.116930, R=0.358799;
  lambda=0.9: P=0.410675, C=0.107292, R=0.314112. The lambda=0.1
  policy selected `selfcheck` for all 121 queries; no single model collapsed.


### T-004 — Cross-validate all three repaired input populations

- **Description:** Verify exact qid equality among the complete joint,
  averaged model-only, and direct-only datasets, plus their respective
  candidate counts.
- **Objective/reason:** Establish the precondition for fair shared-split
  evaluation.
- **Files involved:** The three `*_complete.jsonl` inputs; validation tooling.
- **Dependencies:** T-002 and T-003 (completed).
- **Completion criterion:** All qid sets equal the same 797 IDs; joint rows have
  36 unique prompt--model actions; model-only rows have nine unique models;
  direct rows have nine unique direct-model actions; validation exits zero.
- **Validation command:**
  ```bash
  python analysis/validate_qnorm_pipeline.py \
    --data data/interaction_logs/grpp_il_v1/router_bipartite_qnorm_complete.jsonl \
    --expected-queries-per-task 0
  ```
  The model-only equality command is to be finalized as part of T-001/T-003
  rather than relying on an undocumented one-off snippet.
- **State:** completed
- **Priority:** P0
- **Blocker:** none
- **Observed result:** 797 queries in each input; validation errors: 0;
  populations, candidates, tasks, and rewards aligned.


### T-003 — Generate the aligned direct-only model dataset without training

- **Description:** Use the implemented build-only path to derive
  `router_model_only_direct_qnorm_complete.jsonl` from the complete joint
  source using only `prompt == "direct"`.
- **Objective/reason:** Prepare GraphRouter-direct on the same query population
  without accidentally starting training during data preparation.
- **Files involved:** `train_router_model_only_direct_qnorm.py`,
  `router_bipartite_qnorm_complete.jsonl`,
  `router_model_only_direct_qnorm_complete.jsonl`.
- **Dependencies:** T-001 and T-002 (completed); D-003 and D-007.
- **Completion criterion:** Exactly 797 unique qids; nine unique direct-model
  candidates per query; exact qid equality with the complete joint source; the
  command exits before optimizer construction.
- **Validation command:**
  ```bash
  python train_router_model_only_direct_qnorm.py \
    --source-data data/interaction_logs/grpp_il_v1/router_bipartite_qnorm_complete.jsonl \
    --router-data data/interaction_logs/grpp_il_v1/router_model_only_direct_qnorm_complete.jsonl \
    --rebuild-direct-dataset --build-only
  ```
- **State:** completed
- **Priority:** P0
- **Blocker:** none
- **Observed result:** 797 queries; zero skipped direct queries; nine models
  each with coverage 797; build-only validation passed.


### T-002 — Generate the aligned averaged model-only dataset

- **Description:** Build
  `router_model_only_qnorm_complete.jsonl` from the flat qnorm log while
  filtering against the complete joint qids.
- **Objective/reason:** Give static model-only baselines the same 797-query
  universe as the joint router.
- **Files involved:**
  `train_clean_qnorm_lambdas.jsonl`,
  `router_bipartite_qnorm_complete.jsonl`,
  `router_model_only_qnorm_complete.jsonl`.
- **Dependencies:** T-001 (completed).
- **Completion criterion:** Exactly 797 unique qids; exact qid equality with the
  complete joint dataset; exactly nine unique model candidates per query; no
  missing reward fields.
- **Validation command:**
  ```bash
  python analysis/build_model_only_router_dataset.py \
    --input data/interaction_logs/grpp_il_v1/train_clean_qnorm_lambdas.jsonl \
    --qid-source data/interaction_logs/grpp_il_v1/router_bipartite_qnorm_complete.jsonl \
    --output data/interaction_logs/grpp_il_v1/router_model_only_qnorm_complete.jsonl
  ```
- **State:** completed
- **Priority:** P0
- **Blocker:** none
- **Observed result:** 797 queries; nine models per query; population validation passed.


### T-001 — Add complete-population filtering to the model-only builder

- **Description:** Modify
  `analysis/build_model_only_router_dataset.py` to accept a
  `--qid-source` bipartite JSONL, use its qids as an allowlist, reject
  duplicate source qids, and fail unless the output qid set matches the
  allowlist exactly. Add explicit candidate-count checks for the expected nine
  models per query.
- **Objective/reason:** Prevent the averaged model-only dataset from silently
  retaining queries removed from the complete joint population.
- **Files involved:** `analysis/build_model_only_router_dataset.py`;
  `docs/experiment-status.md`; `docs/next-steps.md`; update
  `docs/decisions.md` only if implementation requires a new policy choice.
- **Dependencies:** D-004, D-005, D-006; availability of
  `router_bipartite_qnorm_complete.jsonl`.
- **Completion criterion:** The builder exposes the allowlist CLI, fails on
  mismatched/duplicate populations or non-nine-model outputs, compiles, and a
  small synthetic test demonstrates both success and expected failure.
- **Validation command:**
  ```bash
  python -m py_compile analysis/build_model_only_router_dataset.py
  ```
  Four synthetic tests passed: valid allowlist, duplicate source qid, empty
  output, and missing model.
- **State:** completed
- **Priority:** P0
- **Blocker:** none


### T-C001 — Audit the qnorm reward and action-space integrity

- **Description:** Verified reward algebra and identified incomplete SQuAD
  coverage in the original 799-query dataset.
- **Objective/reason:** Establish whether existing inputs could support valid
  reruns.
- **Files involved:** `analysis/validate_qnorm_pipeline.py`; original qnorm
  dataset.
- **Dependencies:** Original dataset.
- **Completion criterion:** Audit reported zero reward/data errors and exact
  missing-action cases.
- **Validation command:** `python analysis/validate_qnorm_pipeline.py
  --allow-incomplete`.
- **State:** completed
- **Priority:** P0
- **Blocker:** none

### T-C002 — Produce and validate the complete joint dataset

- **Description:** Non-destructively filtered incomplete queries.
- **Objective/reason:** Establish a uniform 36-action population.
- **Files involved:** `analysis/filter_complete_qnorm_queries.py`,
  `router_bipartite_qnorm_complete.jsonl`.
- **Dependencies:** T-C001.
- **Completion criterion:** Recorded validation: 797 queries, 28,692 edges,
  zero reward/data errors, zero coverage errors.
- **Validation command:** The complete-data validator command in T-004.
- **State:** completed
- **Priority:** P0
- **Blocker:** none

### T-C003 — Implement shared persisted split manifests

- **Description:** Added deterministic split creation and strict manifest
  validation across learned and static methods.
- **Objective/reason:** Ensure identical comparisons per seed.
- **Files involved:** `router/splits.py` and its consumers.
- **Dependencies:** None.
- **Completion criterion:** Deterministic reuse and mismatch failure were
  tested; implementation is present on the repair branch.
- **Validation command:** Python syntax and split tests recorded in
  `docs/experiment-status.md`.
- **State:** completed
- **Priority:** P0
- **Blocker:** none

### T-C004 — Remove GraphRouter-direct outcome leakage and align embeddings

- **Description:** Removed realized cost/token inputs and changed model
  embedding alignment to identifier-based lookup.
- **Objective/reason:** Restore a valid pre-routing feature contract.
- **Files involved:** `train_router_model_only_direct_qnorm.py`.
- **Dependencies:** D-003.
- **Completion criterion:** Repaired scorer and alignment code are present on
  the branch; embedding audit found no missing retained query IDs.
- **Validation command:** Syntax and embedding audit recorded in
  `docs/experiment-status.md`.
- **State:** completed
- **Priority:** P0
- **Blocker:** none

### T-C005 — Establish persistent project handoff documentation

- **Description:** Created consolidated status and append-only decision history.
- **Objective/reason:** Allow continuation without relying on chat history.
- **Files involved:** `docs/experiment-status.md`,
  `docs/decisions.md`, `docs/next-steps.md`.
- **Dependencies:** Repository and experiment audit.
- **Completion criterion:** All three documents exist, cross-reference each
  other, distinguish facts/decisions/actions, and agree on T-001 as the single
  next action.
- **Validation command:** Inspect the three files on
  `fix/p0-routing-integrity`.
- **State:** completed
- **Priority:** P0
- **Blocker:** none

## Hypotheses, not active tasks

- Removing three SQuAD queries is expected to have a small numerical effect,
  but this cannot be treated as a result before rerunning.
- Existing paper tables may combine legacy pipelines or splits; their exact
  provenance remains untrusted until T-010.
- Full joint routing may remain best after repair, but no qualitative ranking
  should be assumed before T-008/T-009.

## Actions requiring user confirmation

- T-015: authorize merging the repair branch only after reviewing seed-1
  evidence.
- T-016: choose the final long-paper cost formulation after repaired results
  are available.
- T-017: decide whether a pre-routing cost predictor belongs in the project
  scope.
