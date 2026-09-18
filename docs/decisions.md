# P1 - Experiment Repair decision log

_Last updated: 2026-09-18_

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
  query--task/query--prompt/query--model edges; an otherwise matched no-message-
  passing arm; and the existing Flat-MLP. Prompt and model are independent
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
- **Status:** active
- **Replaced by:** —


### D-027 — Use per-query ego graphs and a matched no-message-passing control in Stage A

- **Date:** 2026-09-18
- **Context/problem:** A global complete query--candidate graph could allow
  transductive influence between queries, and the historical Flat-MLP uses a
  different objective from the current Edge-GNN. Either issue would weaken the
  architectural comparison.
- **Decision:** Build each Stage A example as an independent query ego graph
  containing its task, all four prompts, and all nine models, with complete
  bidirectional query--task, query--prompt, and query--model relations. Compare
  full message passing against an otherwise identical zero-message-passing arm
  using the same compositional edge-aware scorer. Retrain the Flat-MLP
  architecture under the shared Stage A objective as an additional control in
  a new output namespace; preserve its canonical P0 results unchanged.
- **Justification:** Per-query graphs eliminate cross-query message paths and
  label-selected topology. The no-message-passing arm isolates message passing
  while holding scorer capacity and inputs fixed. A newly matched Flat-MLP run
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
