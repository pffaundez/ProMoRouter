# P1 - Experiment Repair decision log

_Last updated: 2026-11-24_

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
  pre-existing derived direct filename, although the aligned 797-query
  complete derivative has now been generated and build-only validated.
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
- **Implementation update (2026-09-11):** The averaged model-only builder now
  accepts the complete bipartite dataset as a qid allowlist and validates exact
  output population equality before writing. The aligned derivative was then
  generated successfully with 797 queries and nine models per query.

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
- **Implementation update (2026-09-11):** The repaired 797-query experiment
  uses the explicit shared manifest name `qnorm_complete_seed1.json`, avoiding
  collisions with manifests produced from the former 799-query population.

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
- **Implementation update (2026-09-11):** GraphRouter-direct now has a
  `--build-only` mode and validates exact qid equality, the nine expected
  models, direct prompt policy, and rewards before exiting without training.
  The complete derivative passed with 797 queries and full 797-per-model
  coverage.

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
- **Operational update (2026-09-11):** T-001 passed four synthetic tests and
  moved to `Completed`; T-002 became the sole task in `Now`.
- **Operational update (2026-09-11):** T-002 generated and validated the
  797-query averaged model-only dataset; T-003 became the sole task in `Now`.
- **Operational update (2026-09-11):** T-003 generated and validated the
  797-query direct dataset; T-004 is now the sole task in `Now`.

## Decisions requiring confirmation

These are unresolved choices, not adopted decisions:

1. Whether query-normalized cost is the final deployment-cost definition for
   the long paper, and how its limitations will be presented.
2. Whether trainer/evaluator defaults should point directly to complete-only
   datasets or require explicit CLI paths to preserve source visibility.
3. Whether to retain, relocate, or remove legacy pre-qnorm scripts and
   artifacts after all references are audited.
4. Whether to add a separately evaluated pre-routing cost predictor. No such
   predictor is part of the current repaired implementation.
5. When and how `fix/p0-routing-integrity` will be merged into `main`.


### D-013 — Closed-pool offline training with online per-query selection

- Date: 2026-09-14
- Context: The paper must define offline/online operation and candidate-pool scope.
- Decision: ProMoRouter is trained offline from logged interactions and performs online per-query action selection without online parameter updates. The evaluated pool is closed: four prompts and nine models, identical across training and inference.
- Justification: This is the behavior and candidate population used by the repaired trainers and five-seed outputs.
- Alternatives: Continual online learning and zero-shot routing to unseen prompts/models were not evaluated and are not claimed. A pre-routing cost predictor remains a separate extension.
- Consequences: The paper must state closed-pool limitations; adding candidates requires embeddings, pipeline support, and dedicated evaluation.
- Files affected: trainers, docs/methodology-scope.md, paper methodology.
- Status: superseded
- Replaced by: D-016


### D-014 — Add a flat 36-action no-graph baseline

- Date: 2026-09-15
- Context: Reviewers require an equal-action-space comparison to separate the
  contribution of the graph from the benefit of joint prompt-model actions.
- Decision: Add a Flat-MLP baseline scoring all 36 prompt-model actions from
  concatenated query, prompt, and model embeddings, with no graph message
  passing and no realized outcome features.
- Justification: It isolates architectural benefit while preserving the same
  candidates, rewards, splits, and evaluation protocol as ProMoRouter.
- Alternatives: A different action space or outcome-derived cost features were
  rejected for this comparison.
- Consequences: Run this baseline on seeds 1--5 before final table generation.
- Files affected: `train_router_flat_mlp_qnorm.py`, baseline outputs, paper
  tables, and experiment documentation.
- Status: superseded
- Replaced by: D-018


### D-015 — Evaluate description-based inductive generalization with external candidates

- Date: 2026-09-16
- Context: The closed-pool experiments do not test whether the graph can
  generalize to unseen models or prompting strategies.
- Decision: Define a future inductive evaluation with ten total models and five
  total prompting strategies. Train on the existing nine-model/four-prompt
  pool; introduce external candidates only at inference using description-derived
  embeddings. Also evaluate prompt-model pairs whose interactions are masked
  during training.
- Justification: This directly tests the intended GraphRouter-style
  description-conditioned generalization rather than only interpolation within
  the original pool.
- Alternatives: Treating the existing nine models/four prompts as unseen via
  random query splits was rejected as insufficient; closed-pool results remain
  a separate evaluation.
- Consequences: New candidate descriptions, embeddings, and held-out
  interaction outcomes are required. The protocol must not use realized reward,
  performance, or cost to construct inference features.
- Files affected: description configs, embedding builder, inductive trainer,
  held-out datasets, experiment documentation, and paper.
- Status: superseded
- Replaced by: D-016


### D-016 — Use three unseen models and two unseen prompt strategies

- Date: 2026-09-16
- Context: D-015 left the external candidate count open, preventing a
  reproducible inductive evaluation design.
- Decision: Use three external unseen models—
  `HuggingFaceTB/SmolLM2-1.7B-Instruct`,
  `allenai/OLMo-2-1124-7B-Instruct`, and `microsoft/phi-4`—and two
  external unseen prompting strategies, `fewshot` and `self_consistency`.
  The existing nine models and four prompts remain the seen training pool.
- Justification: The models add distinct families and approximately 1.7B,
  7B, and 14B scales; the prompts add demonstration-based and sampling-based
  strategies that are not duplicates of direct, cot, decompose, or selfcheck.
- Alternatives considered or discarded: Reducing the protocol to one unseen
  model and one unseen prompt was discarded as insufficiently informative.
  `plan_then_execute` and `critic_then_answer` were deferred because they
  overlap strongly with existing strategies. Gemma was retained only as a
  fallback pending license review.
- Consequences: The inference candidate space becomes 12 models x 6 prompts
  (72 actions). New descriptions, embeddings, integration checks, and held-out
  outcomes are required; P0/P1 closed-pool artifacts remain unchanged.
- Files affected: `configs/inductive_candidates.yaml`, description configs,
  `data/router/inductive_embeddings/`, inductive datasets and validators,
  `docs/inductive-evaluation-protocol.md`, and `docs/next-steps.md`.
- Status: active
- Replaced by: —


### D-017 — Fix unseen prompt execution budgets

- Date: 2026-09-16
- Context: The unseen prompt strategies can dominate cost if their execution
  budgets are left unspecified.
- Decision: Execute `fewshot` with exactly two fixed demonstrations per task
  and `self_consistency` with exactly three independent samples, aggregated
  by majority over normalized final answers.
- Justification: These budgets make the strategies operationally distinct while
  keeping evaluation cost bounded and reproducible.
- Alternatives considered or discarded: One demonstration was rejected as
  semantically one-shot; five self-consistency samples were deferred as a
  sensitivity analysis.
- Consequences: Demonstration content, normalization, decoding parameters,
  and realized token costs must be recorded for held-out evaluation.
- Files affected: `configs/inductive_candidates.yaml`,
  `configs/prompt_templates.yaml`, inductive evaluation harness and docs.
- Status: active
- Replaced by: —


### D-018 — Replace fewshot with step-back reasoning

- Date: 2026-09-16
- Context: The qnorm interaction log contains no generated response text, so
  two-shot demonstrations cannot be extracted without an additional dataset.
- Decision: Use `step_back` as the second unseen prompt strategy and retain
  `self_consistency` as the other unseen strategy with three samples.
- Justification: `step_back` requires only a static instruction and is
  distinct from direct, cot, decompose, and selfcheck, while avoiding
  demonstration-data leakage and extra data dependencies.
- Alternatives considered or discarded: `fewshot` with two shots was
  discarded for this protocol; `structured_output`, `self_refine`, and
  `least_to_most` were deferred due to incompatibility or overlap.
- Consequences: The unseen prompt set is now `step_back` and
  `self_consistency`; the two-shot configuration is no longer used.
- Files affected: `configs/inductive_candidates.yaml`,
  `configs/prompt_templates.yaml`, `configs/inductive_prompt_execution.yaml`,
  and inductive protocol docs.
- Status: active
- Replaced by: —


### D-019 — Use automatic token-level F1 for Alpaca

- Date: 2026-09-16
- Context: The configuration declared a disabled judge, while the builder
  implementation already computes Alpaca performance with token-level F1.
- Decision: Set Alpaca to `primary_metric: f1` and `eval_mode: auto`, using
  the same reference-output F1 for inductive evaluation.
- Justification: This matches the implemented scorer and avoids introducing an
  uncalibrated external judge.
- Alternatives considered or discarded: A new LLM judge was rejected for the
  current repair because it would break comparability with existing logs.
- Consequences: Historical P0 Alpaca records use the same F1 computation; no
  manual gold answers are created.
- Verification update (2026-09-16): 7,200 raw/qnorm Alpaca edges joined with
  zero missing keys and zero performance mismatches.
- Files affected: `configs/rq2_dataset_builder_smoke.yaml`, dataset builder,
  validation scripts, and paper metrics description.
- Status: active
- Replaced by: —


### D-020 — Use a unified backend for the inductive test

- Date: 2026-09-17
- Context: P0 logs were generated with vLLM, while the available environment
  currently supports Transformers but has no vLLM installation.
- Decision: Keep P0 unchanged. Create a separate inductive test split and
  regenerate all 72 prompt-model actions (12 models x 6 prompts) with the same
  sequential Transformers backend, including seen and unseen candidates.
- Justification: A unified backend avoids mixing generated outcomes from
  different inference engines within the inductive comparison. The separate
  split prevents the result from appearing as an ad hoc join with P0.
- Alternatives considered or discarded: Mixing P0 vLLM outcomes with only
  unseen Transformers outcomes was rejected as the primary inductive table.
  Installing an unverified vLLM stack in the current environment was deferred.
- Consequences: The inductive results are a separate protocol and are not
  numerically merged with P0. Backend, model revisions, decoding parameters,
  token counts, and costs must be recorded.
- Files affected: inductive generation harness, held-out dataset manifests,
  `docs/inductive-evaluation-protocol.md`, and experiment documentation.
- Status: active
- Replaced by: —


### D-021 — Repair truncated inductive generations instead of rerunning the full sweep

- **Date:** 2026-09-18
- **Context/problem:** The completed 8,712-row inductive generation contained two empty extracted responses because generations ended at the 256-token cap.
- **Decision:** Regenerate only the affected qids/model/prompt pairs with `--qids` and a 512-token cap, then replace those rows in a corrected full artifact.
- **Justification:** The raw outputs showed valid reasoning truncated exactly at `The final answer is`; targeted regeneration preserves the completed sweep and avoids unnecessary recomputation.
- **Alternatives considered or discarded:** Rerunning all 8,712 actions was discarded as wasteful; treating empty responses as valid was discarded because it would bias metrics.
- **Consequences:** The final inductive artifact must record the two repaired rows and pass a post-merge integrity validator.
- **Files affected:** `experiments/generate_inductive_transformers.py`, `outputs/inductive_full.jsonl`, repair JSONL files, and validation/docs artifacts.
- **Status:** active
- **Replaced by:** —


### D-022 — Aggregate inductive outcomes separately with realized token cost

- **Date:** 2026-09-18
- **Context/problem:** The corrected 8,712-row inductive artifact required a
  reproducible summary without mixing its unified-Transformers outcomes with
  the P0 vLLM results or treating realized costs as router inputs.
- **Decision:** Aggregate inductive action outcomes in a standalone artifact.
  Report raw input, output, and total token counts; use
  `cost_proxy_tokens = tokens_total`; and derive P/C/R summaries with token cost
  min-max normalized within each query across its 72 actions.
- **Justification:** This preserves the existing query-normalized reward form,
  makes the cost proxy auditable, and respects D-003 and D-020. All token,
  performance, normalized-cost, and reward values remain post-execution
  evaluation outcomes only.
- **Alternatives considered or discarded:** Mixing the aggregates with P0 was
  rejected. A monetary proxy was not introduced because no frozen model price
  table is part of the corrected artifact. A separate masked-pair aggregate was
  not fabricated because no seen-seen mask manifest exists.
- **Consequences:** `analysis/inductive_aggregates.json` is the machine-readable
  descriptive summary. It distinguishes seen/unseen nodes and the four novelty
  quadrants, and explicitly records that masked seen-seen pairs are unavailable.
  These summaries do not constitute router-selection results.
- **Files affected:** `analysis/aggregate_inductive_results.py`,
  `analysis/inductive_aggregates.json`, and continuity documentation.
- **Status:** active
- **Replaced by:** —


### D-023 — Recreate the historical routing ablation as a controlled 2x2 matrix

- **Date:** 2026-09-18
- **Context/problem:** The canonical P0 table does not show a general Edge-GNN
  advantage over Flat-MLP, and the README references four ablation scripts that
  are absent. The existing trainer exposes the intended topology controls as
  flags rather than separate scripts.
- **Decision:** Recreate T-011 with a 2x2 matrix over observed reward-selected
  edge density (`top-k` 3 versus 5) and the complete prompt-model lattice
  (disabled versus enabled). Hold the repaired 797-query population, 36-action
  candidate space, embeddings, split manifests, loss, early stopping, lambdas,
  and seeds fixed. Run seed 1 as an integrity gate before seeds 1--5.
- **Justification:** This is the smallest faithful reconstruction of the
  historical configuration ablation and changes one graph-density factor at a
  time. It does not conflate candidate-space size with graph topology.
- **Alternatives considered or discarded:** Implementing GraphRouter-style
  query-to-candidate connectivity, a new edge-aware scorer, a factorized router,
  or a new regression-plus-ranking objective inside T-011 was rejected because
  those changes belong to the subsequent Edge-GNN v2 experiment. Treating
  `--full-prompt-model-lattice` as a complete query-action graph was also
  rejected: the flag adds prompt-model edges only.
- **Consequences:** T-011 remains a P0 closed-pool ablation and must not be
  presented as Edge-GNN v2. All four configurations score the same 36 actions.
  Results are pending execution on the experiment host.
- **Files affected:** `configs/p0_routing_ablation.json`,
  `scripts/run_p0_routing_ablation.py`,
  `analysis/aggregate_p0_routing_ablation.py`, and continuity documentation.
- **Status:** active
- **Replaced by:** —


### D-024 — Require deterministic aggregation for the T-011 ablation

- **Date:** 2026-09-18
- **Context/problem:** Two new runs of `topk3_observed` with identical seed,
  data, split, configuration, and current code produced materially different
  P/C/R values. The trainer used CUDA `index_add_` for message, degree, and
  entropy accumulation without enabling deterministic algorithms.
- **Decision:** Preserve frozen P0 results unchanged, but require
  `--deterministic` for every T-011 run. In this mode, configure deterministic
  PyTorch/CUDA/cuDNN behavior and replace CUDA atomic index accumulation with
  mathematically equivalent one-hot matrix aggregation. Verify two identical
  seed-1 repetitions before rerunning the 2x2 gate.
- **Justification:** Configuration effects cannot be distinguished from runtime
  variance while identical runs diverge. Determinism is an integrity gate, not
  a performance-oriented hyperparameter.
- **Alternatives considered or discarded:** Continuing directly to five seeds
  was rejected because seed would not uniquely identify an execution. Running
  the full matrix on CPU was deferred because the deterministic GPU path can be
  tested first. Replacing the frozen P0 table was rejected.
- **Consequences:** The first nondeterministic T-011 gate is diagnostic only and
  cannot be used as the ablation result. If two deterministic repetitions are
  not identical, T-011 remains blocked. The deterministic ablation is a new
  controlled evaluation and is not numerically merged into frozen P0.
- **Files affected:** `train_router_edgegnn_qnorm.py`,
  `configs/p0_routing_ablation.json`, `scripts/run_p0_routing_ablation.py`, and
  continuity documentation.
- **Status:** active
- **Replaced by:** —


### D-025 — Accept the deterministic T-011 matrix as the canonical historical ablation

- **Date:** 2026-09-18
- **Context/problem:** The original T-011 seed-1 gate was nondeterministic, so
  configuration differences could not be separated from CUDA accumulation
  variance.
- **Decision:** Accept only the deterministic five-seed 2x2 sweep produced
  under D-024 as the canonical T-011 result. Retain all earlier nondeterministic
  outputs as diagnostic evidence only.
- **Justification:** Two independent deterministic seed-1 repetitions were
  identical except for `model_path`; the complete aggregate validates 20
  sources, 60 seed-level rows, 12 aggregate rows, shared seed identities, 121
  queries per result, and exact reward algebra.
- **Alternatives considered or discarded:** Mixing the original seed-1 gate
  with deterministic seeds 2--5 was rejected. Replacing frozen P0 results with
  the deterministic trainer was also rejected because T-011 is a separate
  controlled ablation.
- **Consequences:** T-011 can be closed. The results show no consistent top-k 5
  benefit and no general prompt--model-lattice benefit. They do not evaluate a
  GraphRouter-style query-to-candidate topology and must not be presented as
  Edge-GNN v2.
- **Files affected:** `analysis/p0_routing_ablation_aggregates.json`,
  `analysis/p0_routing_ablation_table.csv`, and continuity documentation.
- **Status:** active
- **Replaced by:** —


### D-026 — Stage Edge-GNN v2 to isolate architecture before objective and density

- **Date:** 2026-09-18
- **Context/problem:** Closed-pool P0 shows no broad Edge-GNN advantage over
  Flat-MLP, while T-011 shows no consistent benefit from the historical
  density/lattice controls. Changing topology, objective, scorer, and action
  representation simultaneously would prevent causal interpretation.
- **Decision:** Freeze a staged minimal design. Stage A compares three arms
  under the same 36 actions and current training objective: GraphRouter-style
  message passing with `task`, `query`, `prompt`, and `model` nodes and explicit
  query--task/query--prompt/query--model edges; an otherwise matched two-layer
  self-only arm; and the existing Flat-MLP. Prompt and model are independent
  entities. An edge-aware score may use a static learned or description-derived
  prompt--model edge representation, but never realized reward, performance,
  cost, or tokens. Stage B is permitted only after Stage A and changes only the
  objective to compare the current loss against `L_reg + beta * L_rank`, with
  validation reward for early stopping. Graph-density variants and a factorized
  router are conditional follow-ups. Stage C evaluates the selected design on
  the already validated inductive protocol as a separate result family.
- **Justification:** Stage A directly separates the effect of graph message
  passing from the effect of the joint action space. Staging avoids an
  uninterpretable combinatorial sweep and follows the negative T-010/T-011
  evidence.
- **Alternatives considered or discarded:** Implementing every requested
  topology, density, objective, and scorer variant simultaneously was rejected.
  Starting with density was rejected because T-011 already shows weak and
  inconsistent density effects. Presenting the proposed v2 as a result before
  execution is prohibited.
- **Consequences:** The next task is an implementation plan and integrity
  review for Stage A only. P0 remains frozen; inductive results remain separate.
  Any expansion to Stage B or factorized/density variants requires a recorded
  gate decision based on Stage A evidence.
- **Files affected:** Experiment design and continuity documentation; no v2
  implementation file exists yet.
- **Status:** superseded for the new research phase; historical contract preserved
- **Replaced by:** D-028 (new research phase only)


### D-027 — Use per-query ego graphs and a matched self-only control in Stage A

- **Date:** 2026-09-18
- **Context/problem:** A global complete query--candidate graph could allow
  transductive influence between queries, and the historical Flat-MLP uses a
  different objective from the current Edge-GNN. Either issue would weaken the
  architectural comparison.
- **Decision:** Build each Stage A example as an independent query ego graph
  containing its task, all four prompts, and all nine models, with complete
  bidirectional query--task, query--prompt, and query--model relations. Compare
  full message passing against a two-layer self-only arm with the same self
  transforms, activation, LayerNorm, dropout, and compositional edge-aware
  scorer. Retrain the Flat-MLP
  architecture under the shared Stage A objective as an additional control in
  a new output namespace; preserve its canonical P0 results unchanged.
- **Justification:** Per-query graphs eliminate cross-query message paths and
  label-selected topology. The self-only arm controls processing depth while
  disabling neighbor exchange; relation parameters remain inactive and their
  count must be reported. A newly matched Flat-MLP run
  avoids substituting historical numbers trained with a different objective.
- **Alternatives considered or discarded:** A global train-plus-evaluation
  graph was rejected due to transductive coupling. Reward-selected top-k edges
  were rejected for Stage A because they confound architecture with topology.
  A learned 36-pair lookup was rejected because it cannot represent unseen
  prompt--model pairs later.
- **Consequences:** `docs/edgegnn-v2-stage-a.md` and
  `configs/edgegnn_v2_stage_a.json` are the binding pre-implementation
  contract. Stage A remains closed-pool; objective and density variants remain
  gated. No v2 result exists yet.
- **Files affected:** `docs/edgegnn-v2-stage-a.md`,
  `configs/edgegnn_v2_stage_a.json`, and continuity documentation.
- **Status:** active
- **Replaced by:** —


### D-028 — Prioritize GraphRouter-style support graphs for joint routing to unseen LLMs

- **Date:** 2026-11-20
- **Context/problem:** The adopted research focus is joint prompt--model
  selection when an LLM is completely excluded from router training. Merely
  producing scores for a new candidate does not demonstrate useful incorporation.
  Stage A isolates message passing in a closed pool but supplies no behavioral
  support at inference. Its per-query topology is not the mandatory architecture
  for this new research phase.
- **Decision:** Adopt the following four-step methodological policy:
  1. Use GraphRouter's inductive support principle as the methodological starting
     point, extended to known prompts and a new LLM. Construct a support graph
     from permitted historical and calibration interactions rather than merely
     candidate availability. This is not a wholesale restoration of historical
     code. Retain Stage A as historical evidence and a separately labelled
     control; preserve its artifacts and all frozen P0 results.
  2. Train the GNN while excluding an external held-out LLM from training labels,
     outcome-derived topology, and hyperparameter/checkpoint selection. Freeze
     every learned router parameter before incorporation. Compare seen-only,
     unseen-description-only, and unseen-with-calibration conditions on the same
     target queries. Report R, P, C and incorporation behavior. Run a pilot before
     expanding to multiple held-out models, seeds and support samples.
  3. Diagnose selection profiles by model, prompt, pair, task and lambda. Record
     concentration, opportunity capture, optimal-pair hits, and beneficial,
     harmful and tied switches relative to the seen-only policy. A fixed-pair
     policy is a diagnostic flag, not automatically a failure; compare it with a
     validation-selected best fixed pair and observed oracle headroom.
  4. Choose subsequent interventions from the flags. Alternatives may change
     graph representation or training, or operate outside the graph, such as an
     incorporation decision rule, uncertainty estimate or complementary estimator.
     Evaluate changes separately rather than bundling them into the initial pilot.
- **Representation policy:** The primary method remains graph-based. The first
  design option to evaluate is query--model interaction edges conditioned on the
  prompt, with historical quality/cost attributes and an explicit multiedge
  contract. Prompt identity/embedding must affect message construction; separate
  binary links alone do not preserve the full query--prompt--model observation.
  Pair nodes, interaction nodes and hyperedges are alternatives, not adopted
  architecture. The definitive representation, message directions, loss and
  support budget must be specified before implementation.
- **Information boundary:** Outcomes of historical/support executions may be
  inputs in this new protocol; outcomes of target queries may not. Support and
  target qids must be disjoint. Training must mask target interaction outcomes
  and any topology derived from them when predicting those interactions.
  Target queries must not exchange information or modify the support used by
  other targets. Calibration updates input evidence and node representations,
  never learned weights. Candidate descriptions/embeddings must not encode
  held-out evaluation outcomes. Preserve identifier alignment, deterministic
  controls, shared manifests and reward algebra.
- **Metrics and fairness:** Retain R = P - lambda*C for comparable diagnostic
  results. Use one fixed normalization reference across pool conditions so that
  incorporation gain is not an artifact of renormalization. Separate
  post-execution normalized C from deployable pre-routing information; report
  support execution counts and calibration cost separately from target inference
  cost. Specify amortization before claiming net savings. No support size,
  held-out pilot model, statistical success threshold or cost predictor is
  finalized by this decision.
- **Justification:** This tests whether a frozen graph router can exploit
  prompt-conditioned evidence of a new LLM, rather than assuming that adding a
  node or selecting the candidate more often improves utility. GraphRouter
  already uses few-shot evidence without retraining; calibration itself is not
  claimed as novelty. Exact novelty and effectiveness remain to be established.
- **Scope:** Known prompts and tasks are the initial controlled setting.
  Simultaneously unseen prompts/tasks and the external 12-model/6-prompt
  generation protocol remain separate historical/optional evaluations.
- **Alternatives considered or discarded:** Treating Stage A as proof of unseen
  generalization, attributing its design to demonstrated over-densification,
  imposing interaction nodes as already chosen, or immediately launching a new
  training sweep were not adopted.
- **Consequences:** Supersede D-026's execution-order restriction for this phase.
  D-027 remains the contract for historical Stage A, not for the new support
  graph. Refine D-003 only for this new protocol: legitimate support outcomes
  are permitted, target outcomes remain prohibited; P0 rules are unchanged.
  D-005's full 4x9 requirement remains historical; the new held-out protocol must
  validate its own explicit seen/support/test pools. Prepare the design and
  evaluation contract before changing trainers or running experiments.
- **Files affected:** docs/decisions.md, docs/experiment-status.md,
  docs/next-steps.md; future protocol and implementation files not yet created.
- **Status:** active
- **Replaced by:** —



### D-029 — Converge through one prompt-conditioned support-graph pilot

- **Date:** 2026-11-20
- **Decision:** Implement D-028 now with a single minimal graph variant. Parallel
  query--model support edges retain prompt identity, quality and cost. Task
  features are concatenated with queries; two shared message/update rounds
  construct per-prompt model states without learned pair-ID tables. This is a
  GraphRouter-inspired adaptation, not an exact reproduction.
- **Execution:** Use qwen2.5-14b, seed 1, eight support queries and lambda 0.9 as
  the first development pilot. Train only on seen models, MSE + 0.05 CE, and
  select the epoch on seen validation. The same frozen checkpoint yields the
  seen-only, description-only and calibrated conditions.
- **Information/cost:** Disjoint support/targets; no external-model tensors in
  training or tuning; episode targets excluded from support. Fit/support costs
  use seen-only minmax; test metrics retain the fixed original full-pool qnorm
  reference across conditions. This fit/evaluation scale limitation is explicit.
  No monetary or amortized calibration savings claim.
- **Convergence:** Read the pilot R/P/C, calibration and incorporation gains and
  pair/task concentration before any further variant or sweep. No automatic
  branching into objectives, architectures or new research questions.
- **Status:** active; implementation provided, real-data execution pending.
- **Files:** router/support_graph.py, train_router_support_graph.py,
  tests/test_support_graph.py, docs/support-graph-pilot.md.


### D-030 — Condition support aggregation on the target query in one matched pilot

- **Date:** 2026-11-20
- **Evidence:** The Qwen seed1/k8/lambda0.9 audit reproduced the original choices.
  Calibration changed Qwen scores (best-score mean increase +0.15176394 on 54
  oracle opportunities), improved median best-pair rank 13 to 9, and reduced
  its mean margin deficit from -0.25701923 to -0.10525529, without selections.
  Seen scores were unchanged. Its preferred prompt changed uniformly from
  selfcheck to decompose. The router R=0.34954213 exceeded the validation-selected
  fixed-pair-by-task baseline R=0.32354404 by +0.02599808. Neither this difference
  nor the concentration establishes instance-level adaptation or significance.
- **Decision:** Close the score audit and change only support aggregation:
  replace uniform means with target-query cosine weights, softmax temperature
  fixed at 0.1, on raw query-text embeddings excluding task embeddings.
  Apply the same weighting in training, validation and inference. No new
  trainable parameters, objective changes, support search or additional folds.
- **Controls:** Same architecture weights/shapes, hidden=128, seed1, Qwen
  held out, k=8, lambda0.9, splits, episodes, optimizer, epoch limit, patience
  and cost references. Refit once because the training aggregation changes;
  do not claim this is an inference-only treatment of the old checkpoint.
  New outputs get a query_cosine suffix and cannot overwrite prior results.
- **Integrity:** Every target has separate support weights and representations;
  no target-to-target paths, outcomes or batch statistics. Missing edges
  receive zero messages; the description-only candidate retains its raw
  representation. Old checkpoints/default uniform behavior remain supported.
- **Convergence:** Compare this single run with the recorded uniform pilot.
  Inspect R/P/C, incorporation/calibration gain, selections and prompt/task
  concentration; save scores in the result to avoid a further audit run.
  Query-conditioned weights functioning is not assumed to imply useful routing.
  Do not launch sweeps or introduce another mechanism before interpreting it.
- **Status:** active; code and synthetic checks ready; real-data run pending.
- **Files:** router/support_graph.py, train_router_support_graph.py,
  audit_support_graph.py, tests/test_support_graph.py and continuity docs.


### D-031 — Establish a pinned official GraphRouter model-only reference first

- **Date:** 2026-11-22
- **Context:** Both simplified joint support pilots failed to select Qwen.
  Query-conditioned aggregation produced R_qnorm=0.34030043 versus uniform
  0.34954213, with zero incorporation/calibration gain; it does not justify
  another aggregation variant. Their state-per-pair architecture is not the
  official GraphRouter encoder. Do not infer GraphRouter works on our fold
  merely from its published experiments.
- **Decision:** Implement one fixed-Direct model-only control using the official
  encoder at upstream commit 0edfb8758d45acd93c8029354867ffb49bb8a97f.
  Retain its directed graph, GeneralConv/BatchNorm layers, attribute processing,
  compatibility decoder, BCE and edge-level masking. Keep code/config defaults
  explicit (including embedding_dim=8/internal width16 and 32 mask draws).
  Adapt the public training loop to select on seen-only validation macro-F1,
  load the chosen state, freeze all parameters/buffers and evaluate test only
  after training. No replacement with our earlier support scorer.
- **Experiment:** Same 797-query seed1 manifest, Direct only, eight seen models,
  external Qwen excluded from fitting/tuning. Known history uses all 557 train
  qids; 549 produce supervised targets, eight reserved qids supply seen context.
  Nested 8/80 Qwen support sets drawn from train are only revealed after weights
  freeze. The 72 additional qids may have seen-model training observations.
  Evaluate seen-only, description-only, calibrated_8 and calibrated_80 from one
  checkpoint. GPU0 runner; 1000 epochs; no hyperparameter or model sweep.
- **Cost contract:** Primary C is raw cost_proxy_money normalized with fixed
  min/max from fitting seen Direct actions only. Primary R=P-0.9C is used for
  supervision and evaluation. Extrapolated C is not clipped. Stored full-pool
  qnorm C/R is SECONDARY historical output only; it must not be conflated with
  the primary objective. No silent fallback if monetary costs are absent.
  This changes the cost scale relative to the previous joint pilots and is
  part of an adapted control, not a numerically matched reproduction.
- **Outputs:** R/P/C in all conditions, separate historical qnorm summaries,
  Direct-only oracle opportunities for both cost references, incorporation and
  calibration gains, selections/counts/task profiles, scores and raw support
  monetary cost. Freeze weights and BatchNorm stats and test target isolation.
  Bootstrap is conditional on one fitted checkpoint, not seed-level evidence.
- **Convergence:** Close the cosine-aggregation contrast; pause joint-model
  modifications pending this reference result. One training and four evaluations.
  Interpret whether k8/k80 adds utility before extending the working encoder
  with prompts. Zero oracle margin is a scenario limitation, not failed routing.
- **Status:** active; local CPU tests/synthetic control passed; real GPU run pending.
- **Files:** router/graphrouter_official.py, router/LICENSE.GraphRouter,
  train_graphrouter_unseen.py, tests/test_graphrouter_unseen.py,
  requirements-graphrouter.txt, scripts/run_graphrouter_unseen.sh,
  docs/graphrouter-unseen-control.md and continuity docs.


### D-032 — One internally selected incorporation threshold, then close

- **Date:** 2026-11-23
- **Evidence:** The official fixed-Direct GraphRouter control selects Qwen but
  does not improve primary reward over seen-only: description gain -0.00274406,
  k8 gain -0.00224102, k80 gain -0.01890197. This closes T-031; technical
  incorporation is not evidence of useful incorporation. Conditional gain
  intervals include zero; this is not proof of general harm.
- **Decision:** Freeze the uniform SupportGraphRouter architecture and its
  seed1/k8/lambda0.9 defaults. Test one external inference rule: select the
  predicted best new-model prompt only when its score margin over the predicted
  best seen pair exceeds an internally chosen threshold (possibly negative).
  Compare with reject-all and threshold-zero argmax. Do not modify the GNN,
  objective, aggregation, support budget or hyperparameters.
- **Bounded protocol:** Internal holdouts llama3.1-8b and codellama-34b; external
  holdouts yi-34b and mixtral-8x7b. Exclude both external models from every
  inner fit and decision; additionally exclude each inner holdout in its fit.
  Two inner fits use six seen models. Split validation 59 tune / 60 check;
  checkpoint choice and threshold selection use tune only. Gate once on check:
  pooled and per-fold AUC > 0.5, positive pooled gain, nonnegative fold gains.
  Undefined AUC or reject-all fails. Persist the frozen threshold before
  revealing any external support/test outcomes.
- **Closure:** Failed inner gate stops before outer training/evaluation. If it
  passes, train one shared seven-model base and independently add each external
  model with frozen weights. Both must improve over reject-all and argmax to
  pass descriptive external confirmation. No automatic search after failure.
  At most three training trajectories; new namespace preserves older artifacts.
- **Interpretation:** Frozen architecture is not reuse of a checkpoint that
  already saw a held-out model. Every incorporation happens after the relevant
  fit is frozen. Model pools and evaluation-cost contract differ from the
  official monetary-normalized control. Query reuse and a single seed limit
  confirmation; no pristine-dataset or generalization significance claim.
- **Status:** active; 21 CPU support/audit/threshold tests passed, including
  both synthetic orchestration paths. Morel real-data execution pending.
- **Files:** train_support_threshold.py, scripts/run_support_threshold.sh,
  tests/test_support_threshold*.py, docs/support-threshold.md and continuity docs.


### D-033 — Close the tested incorporation interventions and retain a calculation record

- **Date:** 2026-11-23
- **Evidence:** Real T-032 output selected threshold=null (reject-all). Check AUC
  was 0.61718098 pooled, 0.64488017 for Llama-8B and 0.66379310 for CodeLlama.
  AUC passed; reject-all and zero gain failed the gate. All nonempty tuning
  threshold policies lost reward; the best gained -0.00416935844 over 118
  query-model records. No external fit or Yi/Mixtral evaluation occurred.
- **Decision:** Close the global-threshold intervention under D-032 without
  further threshold searches. Consolidate the uniform support pilot/audit,
  cosine aggregation, official GraphRouter control and threshold intervention
  in docs/memoria-intervenciones-unseen.md for reporting and calculation memory.
- **Scope:** No evidence of useful incorporation from these interventions.
  Do not infer general impossibility, causal diagnosis, significant harm or
  negative external results. Model-selected prompts differ from oracle prompts;
  repeated query-model records differ from independent queries. Preserve cost
  contracts and separate the official Direct control from joint qnorm results.
- **Reproducibility:** Record source locations, attachment/dataset/frozen-threshold
  hashes, exact stopping criteria, recalculation snippet and reporting wording.
  Original real-data outputs remain on Morel; no research artifact is overwritten.
- **Status:** active closure decision; T-032 complete. No new training queued.
- **Files:** docs/memoria-intervenciones-unseen.md and continuity documents.


### D-034 — Return to a minimal joint extension of the official GraphRouter

- **Date:** 2026-11-23
- **Correction:** D-033 closes the tested support/threshold interventions, not the
  original joint-routing research question. The official Direct control and the
  simplified joint scorer never tested an official-encoder joint extension.
- **Decision:** Specify one minimal extension in docs/graphrouter-joint-contract.md:
  sources are query–prompt nodes with concatenated query/prompt embeddings;
  destinations remain one shared node per LLM. Preserve the vendored
  FeatureAlign/GeneralConv/edge processing and compatibility decoder; adjust
  query input dimensionality, not the scoring head. This is NOT a separate
  prompt-node architecture and NOT the previous SupportGraphRouter.
- **Protocol:** Seed1/Qwen development pilot; four prompts, one training,
  known-only/description/calibrated k8 from one frozen checkpoint. Keep the
  official Direct adapter's optimizer/masks/BCE/checkpoint convention, but
  supervise one best joint action per original query. Use the Direct-known
  training monetary reference for all prompts and evaluation conditions.
- **Transparency:** Input dimensionality, action count, positive-label proportion,
  history size and calibration action budget increase. Thus this is a specified
  extension, not an unchanged official reproduction or equal-budget prompt
  ablation. The benchmark/Qwen fold is development, not pristine confirmation.
- **Next step:** Implement this contract and its isolation/reduction tests before
  any real-data training. No external utility gate, threshold search or new
  scorer is part of the design. No real joint result exists yet.
- **Status:** active design contract; implementation pending.

#### D-034 implementation update — 2026-11-23

- Implemented router/graphrouter_joint.py and train_graphrouter_joint_unseen.py
  without modifying the official encoder/decoder source. Query–prompt sources,
  shared model nodes and original compatibility scoring are retained.
- The runner fixes seed1/Qwen/lambda0.9, 1000 epochs and three frozen conditions.
  Direct-reference metadata checks verify the common primary monetary units;
  absent/mismatched metadata blocks a matched historical interpretation, not
  the primary within-checkpoint contrast.
- All 35 CPU tests pass: 14 GraphRouter tests, including 10 new joint tests,
  plus 21 support/audit/threshold regression tests. A two-epoch synthetic
  end-to-end test checks optimizer, checkpoint selection, frozen incorporation,
  graph counts, reward algebra and output protection. A controlled Direct
  reduction fixture reproduces original-adapter scores with copied weights.
- GPU/real-data execution remains pending; no research result is claimed.
  T-034 implementation is complete; T-035 is the sole Morel run/interpretation.


### D-035 — Interpret the official joint pilot as positive but fully concentrated

- **Date:** 2026-11-23
- **Evidence:** T-035 real output improves primary R from 0.27727088 to
  0.30704034 (+0.02976946), conditional paired bootstrap 95% CI
  [-0.04242451, 0.10897677]. Description R=0.19038813; support adds +0.11665220.
  Every calibrated query selects Qwen/decompose; description selects
  Qwen/Direct throughout. Known scores change exactly zero.
- **Decision:** Complete T-035 with a descriptive positive incorporation signal,
  not a confirmed instance-routing advantage. Record concentration, uncertainty,
  cost-driven gain and score-scale disparity in docs/graphrouter-joint-results.md.
- **Limits:** One reused seed/fold; 32 new-model calibration actions. The unchanged
  known candidates score around 0.00065–0.00279 versus new-model scores near
  0.48 after support. This is an observed ranking-scale flag, not a causal
  diagnosis. Gain over the realized base and oracle expansion gain use different
  references and must not be compared as identical quantities.
- **Next:** Before any further training, assess a fixed-policy comparator using
  permitted information only: known-model validation for known actions and the
  reserved support for new-model prompts. No held-out-model validation outcomes
  or test outcomes may select the baseline. No architecture change or sweep
  is adopted by this result.
- **Status:** active interpretation; T-035 complete, bounded comparator review
  pending. No new GPU training authorized or launched by this documentation.


### D-036 — Bound the last comparison and close this development stage

- **Date:** 2026-11-23
- **Decision:** Implement one no-training/no-inference fixed-policy audit of the
  existing official joint pilot, then close this stage regardless of outcome.
  No new GNN, thresholds, support budgets or training seeds are introduced.
- **Selectors:** fixed_known uses known-model validation only; fixed_new_support
  uses the eight held-out-model support queries only. fixed_adapted compares
  those two fixed actions on the same support queries, retaining known on ties.
  Never tune using held-out-model validation or test.
- **Closure:** Nonpositive gain against either main fixed control means no
  advantage over both. Positive gains with intervals crossing zero are
  uncertain. Positive conditional intervals still do not establish instance
  routing when all decisions are the same pair. Record the result and close;
  do not automatically add another intervention under any outcome.
- **Implementation:** audit_graphrouter_joint_fixed.py and CPU-only shell runner;
  stdlib only, original data hash checked, policies fixed before reading test,
  original result unchanged. Seven new tests passed; full 42-test suite passed.
- **Status:** audit ready; one Morel calculation pending, not a training run.
  See docs/graphrouter-joint-closure.md.


### D-037 — Close the joint pilot after the final fixed-policy audit

- **Date:** 2026-11-23
- **Evidence:** fixed_known, selected on known validation, is selfcheck/Qwen-7B:
  primary R=0.34255881. Calibrated GNN R=0.30704034, gain -0.03551847,
  conditional CI [-0.10844570,0.03267945]. fixed_adapted selects Qwen/CoT:
  R=0.24274546; GNN gain +0.06429488, CI [-0.00222867,0.13043471].
- **Decision:** Apply D-036's no_advantage_over_both_fixed_controls verdict and
  close T-036 and the current stage. No further fit, audit, variant or sweep
  is queued. Do not automatically reopen earlier support/threshold approaches.
- **Interpretation:** Preserve technical incorporation and the descriptive gain
  over the GNN's own base, but do not claim instance-routing benefit or
  superiority over the fixed known policy. Both comparison intervals include
  zero; neither general inferiority nor impossibility is established.
- **Reporting:** docs/graphrouter-joint-closure.md contains final metrics,
  selector provenance, uncertainty, source hashes and reusable wording. Status,
  result notes and calculation memory are synchronized.
- **Status:** active closure decision; current experimental stage complete.


### D-038 — Correct attribution of the historical GraphRouter-direct baseline

- **Date:** 2026-11-23
- **Code evidence:** P0's aggregator labels model-only Direct result files as
  GraphRouter-direct. Their trainer, train_router_model_only_direct_qnorm.py,
  instantiates DirectModelOnlyRouter, an embedding-concatenation MLP with
  no official GraphRouter encoder or GNN message passing.
- **Correction:** Describe that historical baseline as a Direct model-only MLP
  (legacy label GraphRouter-direct). P0 superiority over it must not be reported
  as superiority over the official GraphRouter paper. Preserve verified numbers;
  the correction concerns attribution and interpretation.
- **Scope:** Distinguish P0 closed-pool routing, the recent adapted official
  Direct control, and the recent official-encoder joint extension. The latter
  is not ProMoRouter v1 with only a new node added. Published few-shot results
  and our marginal/fixed-policy comparisons do not measure identical claims.
- **Record:** docs/graphrouter-comparison-clarification.md includes source paths,
  primary paper reference and the observed score-scale flag without assigning
  a causal diagnosis.
- **Status:** active reporting correction. Pilot remains closed; no new run queued.


### D-039 — Establish a public-artifact GraphRouter reference before prompts

- **Date:** 2026-11-23
- **Authorization:** User requested implementation of the GraphRouter few-shot
  reference before further ProMoRouter adaptation.
- **Sources:** Pin ulab-uiuc/GraphRouter at 0edfb8758d45acd93c8029354867ffb49bb8a97f,
  download the authors' precomputed CSV and pinned model embeddings/descriptions.
  Verify hashes and preserve the published data instead of our Qwen/PR logs.
- **Protocol:** Six known/four unseen in authors' order; Balance reward
  0.5P-0.5C using stored normalized costs, 70/10/20 per-task contiguous public
  split, 80 train support queries and 320 new-model observations. Encoder and
  compatibility decoder unchanged. Follow public config and training masks.
- **Fidelity:** Public release contains 60 query blocks/600 interaction rows per
  task; paper Table2 says 600 Cases, whose unit is not resolved here. An empty
  fourth-model field is resolved only by the official positional mapping.
  Public GeneralConv width16/AdamW differs from paper prose GAT width32/Adam
  schedule. Known validation history enters test, and all query features enter
  train BatchNorm as in code. Primary evaluates final state because public
  test() does not reload best; best-validation state is preserved separately.
  These details are explicit, not silently repaired or mixed.
- **Engineering:** 52 CPU tests pass; 10 new tests include parser alignment,
  withheld outcomes, gradient-step equivalence to upstream, frozen state and
  synthetic end-to-end training. Real CSV/parser/artifacts were verified.
- **Real result:** 1000 CPU epochs completed on the released data. R=0.20734849
  for known-only, description and fewshot80, rounding to published 0.207.
  All 48 choices are known LLaMA-3 (8b); support changes scores, not decisions.
  There are 14 new-model oracle opportunities. Thus scalar agreement is not
  proof of reproducing useful adaptation or the authors' exact policy.
- **Record:** docs/graphrouter-reference.md and
  analysis/graphrouter_reference/seed1881_balance hold metrics and manifest;
  the two checkpoints are preserved in a separately delivered archive. Runtime Torch2.15 CPU/PyG2.8.1; no LLM generation.
- **Status:** Reference implementation and one full run complete. No prompt
  extension launched; inspect the fidelity record rather than assuming the
  published reward establishes incremental incorporation.


### D-040 — Finish the bounded fidelity audit as a partial reproduction

- **Date:** 2026-11-23
- **Scope:** Audit the pinned public release, published protocol and recorded
  CPU reference. Do not start variants or repeat the full benchmark unless a
  concrete error affecting that result is found.
- **Evidence:** Execute the actual upstream train_validate on an engineering
  fixture with matched initialization/RNG. Masks, train scores, parameters/BN
  and validation scores match exactly. Test error is 1.1920929e-7 with the
  reference's history-edge ordering and zero with upstream order. Canonical
  metric/argmax checks have zero error; original result hash is unchanged.
- **Correction:** Future GPU runs now draw mask RNG on CPU like upstream.
  The recorded CPU primary was already equivalent, so no full rerun is needed.
  A manual suspicion about best_f1 was explicitly refuted: it is updated;
  public test() evaluates current state without loading the saved best.
- **Verdict:** partial_reproduction. Public-code blocks and scalar R=0.207
  agreement are supported; the exact original Table5 policy/useful adaptation
  is not verified. Auxiliary IDs/sampler, few-shot driver, configuration-to-
  checkpoint mapping/predictions and exact runtime remain unresolved.
- **Checks:** 18 selected reference/audit/Direct CPU tests pass; no new
  benchmark fit, ProMoRouter change or parameter search.
- **Record:** docs/graphrouter-fidelity-verdict.md and
  analysis/graphrouter_reference/fidelity_audit_final.json.
- **Status:** bounded audit complete; no further automatic run queued.


### D-041 — Train incorporation episodes inside the eight known models, GPU0

- **Date:** 2026-11-23
- **Authorization:** User accepts pseudo-new episodes and explicitly selects GPU0.
  A positive GraphRouter reproduction is no longer a prerequisite for this
  direct ProMoRouter hypothesis test. Previous reference results stay intact.
- **Single treatment:** Preserve current official-encoder joint query–prompt
  graph/scorer. Compare matched full-history control against limited-history
  episodes, both trained only on eight models/four prompts. Real Qwen-14B
  remains absent from all fitting/validation numerical inputs until freeze.
- **Pairing:** Same initialization, grouped target-query masks, labels, optimizer,
  32 draws/epoch, 1000 epochs, seed1/lambda0.9 and known-validation checkpoint
  rule. Only one candidate's permitted history differs: eight support queries
  (32 actions) in the episodic arm. Rotate all eight pseudo roles four times
  per epoch. All target-query outcome edges are hidden in BOTH arms.
- **Control clarification:** Historical training masked individual edges.
  Query grouping is shared by both new arms to prevent target-outcome leakage;
  the old checkpoint is historical only, not the matched primary comparator.
- **Evaluation:** Freeze both best-known-validation checkpoints, then evaluate
  8x4 known, 9x4 description, 9x4 calibrated on identical qids/support.
  Include fixed controls in the same runner, paired final-R and incorporation-
  gain contrasts, uncertainty and concentration; no separate audit/GPU sweep.
- **Verification:** Eight new CPU tests pass, plus ten joint regression tests.
  Poisoning all real-unseen outcomes leaves training/checkpoint/episode hashes
  unchanged. CPU tests are engineering only; no real-data fit occurred here.
- **GPU:** Runner explicitly masks physical GPU0 and fails without CUDA rather
  than substituting CPU. This execution workspace has no CUDA or remote Morel
  channel; the single Morel command is ready.
- **Status:** implementation complete; paired GPU0 execution pending.
- **Files:** router/episodic_incorporation.py, train_graphrouter_joint_episodic.py,
  scripts/run_graphrouter_joint_episodic.sh, two new test modules,
  docs/joint-episodic-incorporation.md and synchronized handoff docs.

### D-042 — Define ProMoRouter v2 and its two research questions

- **Date:** 2026-11-24
- **Decision:** v1 is the reviewed ProMoRouter manuscript. v2 addresses
  methodological issues by returning to GraphRouter's approach and extending
  the action to `(p, m)`, checking the OpenReview feedback for architecture and
  reporting issues, and adding unseen-model adaptation.
- **RQ1:** Promote v1 RQ2 to the primary question in v2. Evaluate practical
  joint prompt–model routing against fixed-prompt GraphRouter and the other
  baselines. Retain the main-table layout of v1: P, C and R at λ ∈ {0.1, 0.5,
  0.9}, with reward as the primary criterion. Compare the two oracle action
  spaces to quantify the ceiling separately from measured router performance.
  Remove v1 RQ1 as a research question.
- **RQ2:** Measure incremental adaptation to models withheld during
  training. Start with a matched 0-vs-8 support-query contrast at fixed
  candidate pool within a split. Moving across folds changes held-out model
  identities and is evidence about coverage, not a support ablation.
  GraphRouter's four-new-model, 80-query Balance experiment is the historical
  reference. The current 6→9 protocol retains its eight-query budget.
- **Verified:** The current 6→9 runner implements the joint and Direct
  variants with the pinned upstream encoder and shared fitting procedure.
  It uses λ = 0.9 with a frozen known-Direct monetary cost reference and
  chooses graph checkpoints by known-validation macro-F1. v1 instead
  specifies query-normalized cost and validation reward. Reporting secondary
  qnorm fields on existing predictions does not align their fitting or
  checkpoint selection with v1. Resolve and document the v2 contract before
  extending the sweep.
- **Evidence:** Supplied B/C evaluation artifacts use the same query manifest
  and eight support queries. At λ = 0.9, joint beats Direct in mean reward
  but trails the adapted MLP and adapted fixed pair in both folds. The joint
  graph selects one prompt–model pair per fold. These runs evaluate
  supported routing and do not estimate incremental adaptation.
- **Status:** RQs and reporting structure confirmed. Implementation, completed
  checks and remaining work are tracked in
  [the v2 checklist](promorouter-v2-checklist.md). The OpenReview comments
  and final cost/checkpoint contract remain open.


### D-043 — Consolidate v2 scope and completed adaptation experiments

- **Date:** 2026-11-26
- **Version boundary:** v1 is the reviewed ProMoRouter short paper that did
  not pass review. The supplied PDF agrees with the [public manuscript](https://openreview.net/pdf?id=e2yVbty4We)
  on §4 and Table 2. v2 is the current revision.
- **RQ1, primary:** “Does joint prompt–model routing outperform model-only
  routing?” is carried forward from v1 RQ2. Retire v1 RQ1 about prompting and
  the performance–cost trade-off as a standalone RQ. Retain the three-regime
  P/C/R presentation and distinguish router outcomes from oracle ceilings.
- **Methodology:** The return to the pinned public GraphRouter encoder and its
  extension to joint prompt–LLM actions is implemented. The joint adapter uses
  query–prompt sources, one node per LLM, observed [R,C,P] edge attributes,
  two GeneralConv layers with BatchNorm and a sigmoid dot-product decoder.
  GraphRouter Direct uses the same encoder with Direct-only actions. The
  original paper's few-shot experiment was partially reconstructed (D-039/040).
  Finish the audit and resolve cost/checkpoint conventions before expanding
  the main table. v1 uses query-normalized token usage and validation reward.
  The completed 6→9 protocol uses training-only monetary normalization
  and known-validation macro-F1 for graph checkpoint selection.
- **OpenReview:** The public PDF was available. The forum requires verification
  and no reviewer comments were retrieved via the public search and API lookup.
  Review architecture, comparison validity and presentation once the comments
  or an export are available. Record each comment's source, response and
  evidence of closure in the checklist.
- **RQ2, new:** How does v2 compare with other routers when adding unseen LLMs
  with limited support, and what effect does that support have with frozen
  weights? Compare methods primarily by test reward with the expanded pool.
  Use a matched 0-vs-8 support contrast at a fixed pool and checkpoint to
  isolate the effect of support. GraphRouter's few-shot experiment motivates
  this question. The v1 manuscript did not test unseen-model incorporation.
- **T-039 completed:** The actual GPU 0 run produced matched calibrated reward
  0.30704 and identical 121/121 Qwen2.5-14B/decompose decisions. The episodic
  fit gave no calibrated improvement. Score analysis gave AUC around 0.50 for
  using Qwen versus the known alternatives. This closes the paired pilot.
- **6→9 completed:** A, B and C used the same 121 test queries, eight support
  queries and one seed, retaining each of the nine LLMs once. Mean reward was
  0.27755 for joint GNN, 0.11847 for GraphRouter Direct, 0.34594 for adapted
  Flat-MLP, 0.31982 for fixed-known and 0.30706 for fixed-adapted. The joint
  GNN beat Direct in all folds (36 versus 9 actions) and trailed Flat-MLP
  and fixed-known in all folds. Each fold selected a different held-out LLM
  with decompose for all 121 queries. The shared test set gives dependent
  folds. A includes Qwen2.5-14B, already used in development.
- **Status:** Incorporation with frozen weights works mechanically. Useful
  adaptation by query remains open. The [v2 checklist](promorouter-v2-checklist.md)
  records the completed work and remaining requirements.

### D-044 — Incorporar las observaciones de ARR a la revisión de v2

- **Fecha:** 2026-11-26.
- **Fuente:** Revisé la metarrevisión, los cuatro informes y las respuestas
  de autor de la sumisión 16731 en la [discusión de ARR](https://openreview.net/forum?id=e2yVbty4We#discussion).
  Con esto queda resuelta la falta de acceso anotada en D-043.
- **Decisión:** Priorizo la validez de las comparaciones, la regeneración de
  las tablas desde los resultados por consulta, la descripción del método
  y la evaluación de modelos no vistos. Mantengo la RQ de routing conjunto
  como principal y la adaptación como RQ2. La sugerencia de ampliar la
  antigua Tabla 1 se resuelve retirando esa RQ del manuscrito.
- **Arquitectura y coste:** El encoder fijado y su extensión a parejas ya están
  implementados. Reescribiré el Algoritmo 1 y las figuras siguiendo el runner.
  Haré explícito dónde entran los resultados observados y verificaré el
  criterio de checkpoint y la escala de coste antes de producir las tablas
  finales. Las comprobaciones del runner actual quedan como evidencia parcial.
- **Evaluación:** Reportaré las diferencias de acciones y presupuesto entre
  baselines, con Flat-MLP como comparador conjunto. Revisaré las discrepancias
  de recompensa de las tablas 2 y 3 de v1, la robustez entre semillas y los
  límites de una evaluación offline. Distinguiré los tiempos medidos del router
  de la latencia de servir una respuesta.
- **Seguimiento:** Las tareas, su prioridad y la evidencia pendiente están en el
  [checklist de v2](promorouter-v2-checklist.md).

### D-045 — Construir la tabla principal con nuevas corridas de v2

- **Fecha:** 2026-11-26.
- **Decisión:** Cerraré primero los ajustes de arquitectura y el protocolo
  de coste y selección de checkpoint. Después ejecutaré la evaluación de RQ1
  en pool cerrado para λ = 0.1, 0.5 y 0.9. Construiré la tabla principal con
  los resultados nuevos de v2 y los comparadores evaluados bajo el mismo
  protocolo, verificando P, C y R desde las decisiones por consulta.
- **Trazabilidad:** Las inconsistencias de v1 motivan la auditoría de las
  métricas. Las corridas 6→9 conservan su papel como evidencia para RQ2.


### D-046 — Nombrar ProMoRouter Lite y pasar a RQ2

- **Decisión:** En el manuscrito llamaré **ProMoRouter Lite (MLP)** al
  método evaluado como Joint Flat-MLP. Puntúa las mismas 36 parejas
  prompt–modelo con un MLP sobre embeddings de consulta, prompt y modelo.
  Usaré **ProMoRouter v2 (Joint GNN)** para la variante de grafo.
- **Atribución:** Dejaré explícito que Lite se evaluó originalmente como
  comparador Flat-MLP. El nombre no implica cambios en el método ni en
  las corridas. En RQ2, **ProMoRouter Lite (updated)** identifica la
  variante que actualiza sus pesos con el soporte.
- **RQ1:** La [comparación de cinco semillas](rq1-closed-pool-results.md)
  permite cerrar la lectura principal en pool conocido. Lite lidera en
  recompensa media entre los métodos evaluables en los tres regímenes.
  La ventaja específica del grafo sigue abierta frente al MLP con las
  mismas acciones.
- **RQ2:** Retomo las corridas 6→9 como comparación entre routers con el
  pool ampliado. El siguiente contraste evalúa, para cada GNN, las
  condiciones de cero y ocho consultas de soporte con el mismo checkpoint,
  pool de nueve modelos, normalización y consultas de test. Mantendré la
  recompensa absoluta entre routers como comparación principal y usaré
  la diferencia emparejada para medir el efecto del soporte.

### D-047 — Separar la adaptación sin reentrenar de la comparación entre sistemas

- **RQ2. ¿Qué método se adapta mejor sin reentrenar?** Compararé el GNN conjunto y GraphRouter Direct con pesos congelados. Usaré el contraste 0 frente a 8 consultas con el mismo checkpoint, pool ampliado y test para medir el aporte del soporte. Reportaré ΔR y cambios de selección, junto con las diferencias de espacio de acciones y número de observaciones.
- **RQ3. ¿Qué sistema obtiene mayor recompensa al incorporar modelos nuevos con consultas de soporte?** Compararé la recompensa absoluta con ocho consultas de soporte entre los dos GNN, ProMoRouter Lite actualizado y las políticas fijas. Indicaré qué métodos actualizan pesos y el tiempo de incorporación registrado.
- **Evidencia:** Ya tengo las condiciones 0 y 8 para ambos GNN y los resultados de los sistemas con ocho consultas en A, B y C. Lite actualizado entra en RQ3. Su contraste propio 0 frente a 8 permitiría medir una pregunta adicional sobre el efecto de su actualización.

### D-048 — Ordenar las preguntas sobre incorporación de LLM nuevos

- **RQ2. ¿Qué sistema obtiene mayor recompensa al incorporar modelos nuevos con consultas de soporte?** Reportaré primero la comparación con ocho consultas y el pool ampliado. Incluiré ProMoRouter v2, GraphRouter Direct, ProMoRouter Lite actualizado y las políticas fijas. Indicaré qué métodos actualizan pesos y cuántas observaciones usan.
- **RQ3. ¿Cuánto aporta el soporte a los routers congelados al incorporar modelos nuevos?** Reportaré el contraste emparejado de 0 frente a 8 consultas para los dos GNN con checkpoint, candidatos y test fijos. Si el contraste de Lite aporta evidencia relevante, lo incluiré como comparación complementaria e indicaré que la condición con soporte actualiza sus pesos. Para atribuir su efecto específicamente a las observaciones nuevas, usaré un control de actualización con solo datos conocidos y el mismo presupuesto de pasos.
- Este orden sustituye la asignación anterior de RQ2 y RQ3. RQ1 conserva la comparación en el pool conocido.
