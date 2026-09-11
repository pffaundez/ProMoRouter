# P1 - Experiment Repair decision log

_Last updated: 2026-09-11_

This file records only decisions confirmed in the working conversation or
verified in the repository. General progress, measurements, and task tracking
belong in `docs/experiment-status.md`. Historical entries are preserved when
a decision is superseded or reverted.

## Repository cross-check and current discrepancies

The decisions below were checked against
`fix/p0-routing-integrity` on 2026-09-11. The following discrepancies remain
visible and are not treated as resolved:

- The README repository tree and initial data-preparation section still refer
  to `data/interaction_logs/interaction_logs_v1/`, whereas the repaired
  trainers and validation tools use `data/interaction_logs/grpp_il_v1/`.
- The Edge-GNN default input is still the original incomplete
  `router_bipartite_qnorm.jsonl`. The confirmed complete-only input is used
  only when supplied through `--data-path`.
- The static evaluator still defaults to `router_model_only_qnorm.jsonl` and
  `router_bipartite_qnorm.jsonl`, not their aligned complete-only variants.
- GraphRouter-direct still defaults to the original bipartite source and the
  pre-existing derived direct dataset. Its code validates source coverage and
  qid equality when rebuilt, but the 797-query derivative has not yet been
  generated and verified.
- The repaired implementation confirms that validation/test queries connect
  only to task nodes and that the Edge-GNN objective is
  MSE + KL + CE - entropy. Any paper text describing connections to every
  prompt/model at inference or only regression plus one ranking term remains
  inconsistent until rewritten.

## Confirmed decisions

### D-001 — Isolate P0 repairs on a dedicated branch

- **Date:** 2026-09-11
- **Context/problem:** Integrity repairs must be reviewed and validated without
  silently changing the public `main` branch.
- **Decision:** Perform the repair work on
  `fix/p0-routing-integrity`.
- **Justification:** The branch keeps the previous public state intact while
  the corrected pipeline is audited.
- **Alternatives considered or discarded:** Editing `main` directly was
  discarded during validation.
- **Consequences:** All repair commits and living experiment documentation are
  developed on this branch until an explicit merge decision is made.
- **Files affected:** Repository-wide repair branch; no single file.
- **Status:** active
- **Replaced by:** —

### D-002 — Use the implemented query-normalized reward

- **Date:** 2026-09-11
- **Context/problem:** Reported rewards and methodology must share one
  unambiguous definition.
- **Decision:** Use
  `R(q,a) = P(q,a) - lambda * C(q,a)`, where `C` is the stored
  query-normalized cost and the supported settings are lambda 0.1, 0.5, and
  0.9.
- **Justification:** The qnorm dataset audit found zero algebra errors across
  the stored action edges, and the trainer reads these reward fields.
- **Alternatives considered or discarded:** The convex form
  `(1-lambda)P-lambda*C` was explicitly rejected as inconsistent with the
  declared deployment reward. Legacy non-qnorm rewards are excluded from the
  repaired results.
- **Consequences:** Validators recompute every stored reward; all result tables
  must be regenerated under this definition.
- **Files affected:** `analysis/validate_qnorm_pipeline.py`,
  `train_router_edgegnn_qnorm.py`,
  `train_router_model_only_direct_qnorm.py`, `README.md`.
- **Status:** active
- **Replaced by:** —

### D-003 — Exclude post-execution outcomes from routing-time features

- **Date:** 2026-09-11
- **Context/problem:** Realized performance, output tokens, monetary cost,
  normalized cost, and reward are unavailable before an action executes and
  would leak target information if used by a deployed router.
- **Decision:** Treat those values only as offline labels and evaluation
  outcomes, not numeric scorer inputs.
- **Justification:** Repository inspection confirmed that the Edge-GNN scorer
  does not consume them; GraphRouter-direct previously did and was repaired.
- **Alternatives considered or discarded:** Adding realized
  `C(q,a)` or mean reward to the action feature vector was discarded.
  A separately trained pre-routing cost predictor was discussed but not
  adopted.
- **Consequences:** GraphRouter-direct scores from query/model embeddings only.
  Reward may still affect offline Edge-GNN topology through selection of
  top-k observed training actions; this must be stated explicitly.
- **Files affected:** `train_router_edgegnn_qnorm.py`,
  `train_router_model_only_direct_qnorm.py`.
- **Status:** active
- **Replaced by:** —

### D-004 — Preserve the original qnorm dataset and derive a complete-only input

- **Date:** 2026-09-11
- **Context/problem:** The original 799-query bipartite dataset contains
  incomplete action spaces for two SQuAD queries and omits a third expected
  SQuAD query.
- **Decision:** Keep `router_bipartite_qnorm.jsonl` immutable and use the
  non-destructive derivative
  `router_bipartite_qnorm_complete.jsonl` for repaired experiments.
- **Justification:** Removing incomplete queries avoids comparing methods over
  different candidate sets while preserving provenance.
- **Alternatives considered or discarded:** Overwriting the source file was
  rejected. Regenerating only the missing interactions was not adopted because
  matching the original checkpoints and inference environment has not been
  established.
- **Consequences:** The repaired population is 797 queries and 28,692 action
  edges: 200 HotpotQA, 200 GSM8K, 197 SQuAD, and 200 Alpaca.
- **Files affected:** `analysis/filter_complete_qnorm_queries.py`,
  `analysis/validate_qnorm_pipeline.py`,
  `router/data_validation.py`, `README.md`.
- **Status:** active
- **Replaced by:** —

### D-005 — Require the full unique 4 x 9 action space before joint training

- **Date:** 2026-09-11
- **Context/problem:** Partial or duplicated candidate spaces could be accepted
  silently and invalidate comparisons.
- **Decision:** Require exactly four expected prompts, nine expected models,
  and 36 unique prompt--model actions for every Edge-GNN query.
- **Justification:** This is the declared Cartesian action space and was
  verified on the complete-only dataset.
- **Alternatives considered or discarded:** Warning-only validation is retained
  solely as an explicit diagnostic mode; it is not accepted for repaired
  training.
- **Consequences:** The main trainer fails before optimization on incomplete,
  duplicated, missing, or unexpected actions.
- **Files affected:** `router/data_validation.py`,
  `analysis/validate_qnorm_pipeline.py`,
  `train_router_edgegnn_qnorm.py`.
- **Status:** active
- **Replaced by:** —

### D-006 — Share persisted query splits across compared methods

- **Date:** 2026-09-11
- **Context/problem:** Earlier seeds could change both initialization and data
  partitions, and static baselines could use a different seed/population.
- **Decision:** Create one deterministic split manifest per seed and reuse it
  across Edge-GNN, GraphRouter-direct, and static baselines.
- **Justification:** Comparisons require identical train, validation, and test
  qids.
- **Alternatives considered or discarded:** Independent in-script splits and a
  fixed seed-42 baseline split were discarded.
- **Consequences:** Manifests validate seed, ratios, qid fingerprint, exact qid
  universe, disjointness, and non-empty partitions; population mismatch fails
  loudly.
- **Files affected:** `router/splits.py`,
  `train_router_edgegnn_qnorm.py`,
  `train_router_model_only_direct_qnorm.py`,
  `analysis/evaluate_qnorm_baselines_on_test.py`.
- **Status:** active
- **Replaced by:** —

### D-007 — Define GraphRouter-direct as a fixed-direct, model-only baseline

- **Date:** 2026-09-11
- **Context/problem:** The model-only comparison needs a pre-routing feature
  contract and a fixed prompting policy.
- **Decision:** Build its candidates from `prompt == "direct"` and route only
  over the nine model choices using query and model embeddings.
- **Justification:** This isolates adaptive model selection while keeping the
  prompt fixed and complies with D-003.
- **Alternatives considered or discarded:** Averaging outcomes across prompts
  describes a different model-only baseline and is not the repaired
  GraphRouter-direct policy.
- **Consequences:** Its derived dataset must contain exactly the same qids as
  the complete joint source before comparison.
- **Files affected:** `train_router_model_only_direct_qnorm.py`.
- **Status:** active
- **Replaced by:** —

### D-008 — Align embedding matrices by semantic identifier

- **Date:** 2026-09-11
- **Context/problem:** Relying on serialized dictionary order could assign a
  model the wrong embedding.
- **Decision:** Align model embeddings to the trainer's expected model IDs
  explicitly; verify task, prompt, model, and query embedding coverage before
  running.
- **Justification:** Identifier-based alignment removes an order-dependent
  failure mode. The audit found the expected 4 task, 4 prompt, and 9 model IDs,
  all with dimension 384.
- **Alternatives considered or discarded:** Positional reliance on file order
  was discarded.
- **Consequences:** All 797 retained queries are covered. The two unused query
  embeddings may remain because they are not indexed by the selected dataset.
- **Files affected:** `train_router_model_only_direct_qnorm.py`; local
  artifacts under `data/router/`.
- **Status:** active
- **Replaced by:** —

### D-009 — Gate the multi-seed sweep on a seed-1 integrity run

- **Date:** 2026-09-11
- **Context/problem:** Launching all runs before checking aligned inputs and
  routing behavior would waste compute and could reproduce invalid results.
- **Decision:** Run and inspect seed 1 first; launch seeds 1--5 only after the
  seed-1 run passes integrity and behavior checks.
- **Justification:** A single shared-split run can expose mismatched qids,
  invalid outputs, or fixed-action collapse early.
- **Alternatives considered or discarded:** Immediately launching the
  five-seed sweep was deferred.
- **Consequences:** No new multi-seed result is trusted until the seed-1 gate
  passes.
- **Files affected:** Experiment procedure and output directories; no exclusive
  code file.
- **Status:** active
- **Replaced by:** —

### D-010 — Do not mix newly generated interactions with the March logs

- **Date:** 2026-09-11
- **Context/problem:** Filling missing actions with a different model
  checkpoint or inference environment would create a heterogeneous dataset.
- **Decision:** Do not add newly generated LLM interactions to the repaired
  March dataset unless the original checkpoints and generation environment can
  be reproduced.
- **Justification:** Complete-case filtering is currently more defensible than
  mixing non-comparable generations.
- **Alternatives considered or discarded:** Ad hoc regeneration of the 50
  missing actions was discarded for the current repair.
- **Consequences:** The repaired experiment uses 797 complete queries.
- **Files affected:** Interaction-log generation procedure and
  `router_bipartite_qnorm_complete.jsonl`; no source file is overwritten.
- **Status:** active
- **Replaced by:** —

### D-011 — Maintain append-only experiment decision history

- **Date:** 2026-09-11
- **Context/problem:** Significant choices could otherwise be lost or silently
  rewritten as the repair evolves.
- **Decision:** Maintain this file after every adopted, replaced, or reverted
  significant decision. Preserve prior entries and change their status instead
  of deleting their history.
- **Justification:** A durable audit trail distinguishes confirmed choices from
  hypotheses and pending work.
- **Alternatives considered or discarded:** Keeping decisions only in chat or
  folding them into the general status document was rejected.
- **Consequences:** Future changes must update both this decision history and,
  when they affect execution state, `docs/experiment-status.md`.
- **Files affected:** `docs/decisions.md`,
  `docs/experiment-status.md`.
- **Status:** active
- **Replaced by:** —

### D-012 — Maintain a single executable next-action queue

- **Date:** 2026-09-11
- **Context/problem:** Status and decision history alone do not establish an
  unambiguous execution order or expose blockers between tasks.
- **Decision:** Maintain `docs/next-steps.md` after every significant change,
  keep exactly one task in `Now`, move tasks between `Now`, `Next`,
  `Later`, `Blocked`, and `Completed`, and retain only a short completed
  history.
- **Justification:** Another working session must be able to continue using
  only the three project handoff documents, without reconstructing chat
  history.
- **Alternatives considered or discarded:** Keeping the roadmap only in the
  status document or chat was rejected because it mixes facts with intended
  actions.
- **Consequences:** Every significant implementation or experiment change must
  synchronize `docs/experiment-status.md`, this decision log when a decision
  changes, and `docs/next-steps.md`. The current sole next action is T-001.
- **Files affected:** `docs/experiment-status.md`,
  `docs/decisions.md`, `docs/next-steps.md`.
- **Status:** active
- **Replaced by:** —

## Decisions requiring confirmation

These are unresolved choices, not adopted decisions:

1. Whether query-normalized cost is the final deployment-cost definition for
   the long paper, and how its limitations will be presented.
2. Whether trainer/evaluator defaults should point directly to complete-only
   datasets or require explicit CLI paths to preserve source visibility.
3. The exact rebuild interface for
   `router_model_only_qnorm_complete.jsonl`; the required 797-qid equality is
   confirmed, but the builder change has not yet been implemented.
4. Whether to retain, relocate, or remove legacy pre-qnorm scripts and
   artifacts after all references are audited.
5. Whether to add a separately evaluated pre-routing cost predictor. No such
   predictor is part of the current repaired implementation.
6. When and how `fix/p0-routing-integrity` will be merged into `main`.
