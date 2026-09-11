# ProMoRouter experiment status

_Last updated: 2026-09-11_

## Experiment objective

Repair and revalidate the ProMoRouter experimental pipeline for a long-paper
resubmission. The immediate objective is to ensure that joint prompt--model
routing is evaluated with:

- the declared reward, (R(q,a)=P(q,a)-\lambda C(q,a));
- no post-execution information available to a router at decision time;
- identical query splits across learned and static methods;
- a complete (4\times9=36)-action space for every evaluated query; and
- reproducible inputs, validation checks, and multi-seed results.

## Current status

### Verified facts

- Development branch: `fix/p0-routing-integrity`.
- Permanent decision history is maintained in `docs/decisions.md`; unresolved
  choices are kept separate from confirmed decisions.
- The executable task queue is maintained in `docs/next-steps.md`, with exactly
  one current action: T-004.
- The canonical Edge-GNN trainer is `train_router_edgegnn_qnorm.py`.
- Its canonical input schema is the one in
  `router_bipartite_qnorm.jsonl`: one row per query with candidate actions in
  `action_edges`.
- The original qnorm dataset has SHA-256
  `62a42546eba7e3c057dd33b1bb4e2ca3ddaf81573d4276daad6a4ae3befa2369`.
- The original dataset contains 799 queries and 28,750 action edges.
- Every stored qnorm reward satisfies (R=P-\lambda C) for
  `lambda={0.1,0.5,0.9}); no reward-algebra errors were found.
- The original dataset has incomplete coverage:
  - `squad-train-000073` is missing 6 actions;
  - `squad-train-000175` is missing 8 actions;
  - `squad-train-000194` is absent entirely (36 actions).
- A non-destructive complete-only dataset was generated and validated:
  `router_bipartite_qnorm_complete.jsonl`.
- The complete-only dataset contains 797 queries and 28,692 action edges:
  200 HotpotQA, 200 GSM8K, 197 SQuAD, and 200 Alpaca.
- It has zero reward/data errors and zero within-query coverage errors.
- All four embedding files exist locally under `data/router/` and their
  internal IDs, dimensions, and coverage have been verified.
- Query embeddings cover all 797 retained queries. The file contains 799 query
  embeddings: zero are missing and the two extras correspond to removed
  incomplete queries.
- Task, prompt, and model embedding matrices are correctly aligned:
  - tasks: `['gsm8k', 'hotpotqa', 'squad', 'alpaca']`, shape `(4, 384)`;
  - prompts: `['direct', 'cot', 'decompose', 'selfcheck']`, shape `(4, 384)`;
  - models: the expected nine model IDs in trainer order, shape `(9, 384)`.
- The main Edge-GNN scorer does not consume realized reward, performance, cost,
  token count, or monetary cost as numeric routing-time features.
- Offline reward labels do affect Edge-GNN topology: top-k training actions are
  selected by reward to form observed structural edges.
- The previous GraphRouter-direct implementation consumed realized cost and
  token-use features. Those inputs have been removed on the repair branch.
- The previous GraphRouter-direct implementation could misalign model
  embeddings by relying on order. Model embeddings are now aligned by ID.
- The averaged model-only builder now accepts a bipartite qid allowlist and
  rejects duplicate/mismatched populations, empty outputs, missing rewards,
  and candidate sets other than the expected nine models.
- `router_model_only_qnorm_complete.jsonl` was generated on the experiment
  host: 797 queries, nine models per query, population validation passed.
- GraphRouter-direct now exposes `--build-only` and validates exact qid/model
  coverage, direct prompt policy, and reward presence before any training.
- `router_model_only_direct_qnorm_complete.jsonl` was generated on the
  experiment host: 797 queries, zero skipped direct queries, and all nine
  models have coverage 797.
- A dedicated cross-input validator now checks the three repaired populations,
  candidates, tasks, costs, and reward arithmetic together.

### Open hypotheses

- Removing three SQuAD queries will probably have a small numerical effect, but
  no result magnitude should be assumed before rerunning.
- Existing tables may mix runs, splits, or legacy reward pipelines. Exact
  reported values remain untrusted until regenerated from the repaired pipeline.

## Confirmed decisions

1. Use the query-normalized deployment reward
   (R(q,a)=P(q,a)-\lambda C(q,a)).
2. Treat realized performance, output length, monetary cost, and reward as
   offline outcomes/labels, not routing-time inputs.
3. Preserve `router_bipartite_qnorm.jsonl` unchanged as the source artifact.
4. Use `router_bipartite_qnorm_complete.jsonl` for repaired experiments.
5. Require exactly four prompts, nine models, and 36 unique prompt--model
   actions per query.
6. Persist and reuse one split manifest per seed across all compared methods.
7. Fail loudly on mismatched query populations instead of silently evaluating
   different test sets.
8. Run and inspect seed 1 before launching the complete five-seed sweep.
9. Do not mix newly generated LLM interactions with the March interaction logs
   unless the original checkpoints and inference environment are reproduced.
10. Preserve significant decision history in `docs/decisions.md`; supersede or
    revert entries explicitly rather than silently rewriting them.
11. Maintain a single executable next action and task-state transitions in
    `docs/next-steps.md`.

## Modified files

The following files are modified or added on
`fix/p0-routing-integrity`:

- `router/__init__.py`
- `router/splits.py`
- `router/data_validation.py`
- `analysis/validate_qnorm_pipeline.py`
- `analysis/filter_complete_qnorm_queries.py`
- `analysis/evaluate_qnorm_baselines_on_test.py`
- `analysis/build_model_only_router_dataset.py`
- `analysis/validate_aligned_router_inputs.py`
- `train_router_edgegnn_qnorm.py`
- `train_router_model_only_direct_qnorm.py`
- `README.md`
- `docs/experiment-status.md`
- `docs/decisions.md`
- `docs/next-steps.md`

Key effects:

- shared, persisted split manifests;
- dataset fingerprint and split-overlap checks;
- strict full-action-space validation before training;
- a non-destructive complete-query filter;
- removal of post-execution features from GraphRouter-direct;
- ID-based model-embedding alignment;
- query-population consistency checks for derived model-only data;
- documented validation and training commands; and
- an append-only decision log with explicit implementation discrepancies and
  decisions requiring confirmation; and
- a dependency-aware task queue with one unambiguous next action.

## Current datasets and artifacts

### Canonical source and derived inputs

| Artifact | Status | Role |
|---|---|---|
| `shards/train__<model>.jsonl` (9 files) | Keep | Original per-model interaction logs |
| `train_clean.jsonl` | Keep | Combined and cleaned interaction log |
| `train_clean_qnorm_lambdas.jsonl` | Keep | Flat interactions with qnorm costs and rewards |
| `router_bipartite_qnorm.jsonl` | Keep immutable | Original grouped dataset; 799 queries, incomplete |
| `router_bipartite_qnorm_complete.jsonl` | Current repaired input | Complete-only Edge-GNN dataset; 797 queries |
| `router_model_only_qnorm.jsonl` | Legacy/stale for repair | Original model-only aggregate; do not use for repaired runs |
| `router_model_only_qnorm_complete.jsonl` | Current repaired input | Validated 797-query averaged model-only dataset |
| `router_model_only_direct_qnorm.jsonl` | Legacy/stale for repair | Original direct derivative; do not use for repaired runs |
| `router_model_only_direct_qnorm_complete.jsonl` | Current repaired input | Validated build-only derivative; 797 queries and nine direct models |

### Legacy artifacts

These belong to the pre-qnorm pipeline and must not be used for repaired
results:

- `train_clean_lambdas.jsonl`
- `router_model_only.jsonl`
- `router_bipartite.jsonl`

They are still referenced by legacy scripts, so they should not be removed
until those scripts are retired or moved to a legacy area.

### Embeddings

The following files exist:

- `data/router/query_embeddings.pt`
- `data/router/task_embeddings.pt`
- `data/router/prompt_embeddings.pt`
- `data/router/model_embeddings.pt`

Their IDs, shapes, and query coverage are verified. All retained queries have
embeddings; the two unused query embeddings are harmless because the trainer
indexes embeddings only for qids present in the selected dataset.

## Verifications performed

- Repository implementation inspected against the paper/rebuttal description.
- Reward algebra checked across all 28,750 original action edges.
- Original action-space coverage checked query by query.
- Missing SQuAD queries/actions identified exactly.
- Complete-only filter run on the original dataset.
- Complete-only output validated:
  - 797 queries;
  - 28,692 edges;
  - 36 unique actions per query;
  - zero reward/data errors;
  - zero coverage errors.
- Shared split creation and deterministic manifest reuse tested.
- Python syntax compilation completed for modified files.
- Import behavior from `analysis/` checked.
- Git diff whitespace validation completed.
- Repair branch recloned and the complete filtering/validation workflow rerun
  successfully from the remote branch.
- Decision log cross-checked against the repair branch; unresolved README and
  default-input discrepancies are recorded in `docs/decisions.md`.
- Next-step queue cross-checked against the current builders, trainer CLIs, and
  dataset state; T-004 is the only task in `Now`.
- Model-only builder syntax compilation passed, together with four synthetic
  tests covering a valid allowlist, duplicate source qids, empty output, and a
  missing model candidate.
- Averaged model-only generation completed on the experiment host: 797 queries,
  nine models per query, and population validation passed.
- GraphRouter-direct build-only changes passed remote syntax compilation.
- Direct dataset generation completed on the experiment host: 797 queries,
  zero skipped direct queries, nine models each covering all 797 queries, and
  build-only validation passed.
- Cross-input validator passed syntax compilation and three synthetic tests:
  valid candidates, missing model detection, and reward-error detection.
- Embedding audit completed:
  - dataset qids: 797;
  - query embedding qids: 799;
  - missing query embeddings: 0;
  - extra query embeddings: 2;
  - task/prompt/model ID order matches the trainer;
  - task, prompt, and model embedding dimension: 384.

## Problems resolved

- Reward consistency is now automatically checked.
- Partial action spaces are no longer silently accepted by the Edge-GNN.
- The original dataset is preserved while a complete-only derivative is used.
- Learned and static methods can share persisted splits by seed.
- Split manifests validate seed, ratios, qid universe, overlap, and dataset
  fingerprint.
- GraphRouter-direct no longer receives realized cost/token information.
- GraphRouter-direct model embeddings are aligned by identifier.
- Derived direct-model data must match the source query population.
- Averaged model-only data can now be restricted to an exact qid allowlist and
  is validated before being written.
- The aligned averaged model-only dataset has been generated for all 797
  retained queries.
- Direct-only data can now be generated and validated without starting an
  optimizer or loading embeddings.
- The aligned direct-only dataset has been generated for all 797 retained
  queries.
- Cross-dataset equality and reward checks now have a reusable validator.

## Problems pending

1. Cross-validate exact qid and candidate equality across the three repaired
   inputs using `analysis/validate_aligned_router_inputs.py`.
3. Run Edge-GNN, GraphRouter-direct, and static baselines on the same seed-1
   manifest.
4. Compare seed-1 action distributions and confirm no fixed-pair collapse.
5. Run seeds 1--5 only after seed 1 passes.
6. Regenerate every (P,C,R) table from machine-readable outputs.
7. Recreate the routing-configuration ablation with the corrected pipeline.
8. Rewrite the paper's graph construction, scorer, and loss so they match the
   actual implementation.
9. Decide whether query-normalized cost is the final deployment cost
   definition and document its limitations.
10. Retire or isolate legacy pre-qnorm scripts and artifacts.
11. Correct remaining README paths and commands that refer to absent or legacy
    files.

## Exact next step

Execute T-004 from `docs/next-steps.md` on the experiment host after pulling
the branch:

```bash
python analysis/validate_aligned_router_inputs.py \
  --bipartite data/interaction_logs/grpp_il_v1/router_bipartite_qnorm_complete.jsonl \
  --model-only data/interaction_logs/grpp_il_v1/router_model_only_qnorm_complete.jsonl \
  --direct data/interaction_logs/grpp_il_v1/router_model_only_direct_qnorm_complete.jsonl \
  --expected-queries 797
```

The command must report 797 queries for all three inputs, zero validation
errors, and aligned populations, candidates, tasks, and rewards. Do not begin
training until this check passes.
