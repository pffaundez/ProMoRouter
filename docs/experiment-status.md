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

## Modified files

The following files are modified or added on
`fix/p0-routing-integrity`:

- `router/__init__.py`
- `router/splits.py`
- `router/data_validation.py`
- `analysis/validate_qnorm_pipeline.py`
- `analysis/filter_complete_qnorm_queries.py`
- `analysis/evaluate_qnorm_baselines_on_test.py`
- `train_router_edgegnn_qnorm.py`
- `train_router_model_only_direct_qnorm.py`
- `README.md`

Key effects:

- shared, persisted split manifests;
- dataset fingerprint and split-overlap checks;
- strict full-action-space validation before training;
- a non-destructive complete-query filter;
- removal of post-execution features from GraphRouter-direct;
- ID-based model-embedding alignment;
- query-population consistency checks for derived model-only data; and
- documented validation and training commands.

## Current datasets and artifacts

### Canonical source and derived inputs

| Artifact | Status | Role |
|---|---|---|
| `shards/train__<model>.jsonl` (9 files) | Keep | Original per-model interaction logs |
| `train_clean.jsonl` | Keep | Combined and cleaned interaction log |
| `train_clean_qnorm_lambdas.jsonl` | Keep | Flat interactions with qnorm costs and rewards |
| `router_bipartite_qnorm.jsonl` | Keep immutable | Original grouped dataset; 799 queries, incomplete |
| `router_bipartite_qnorm_complete.jsonl` | Current repaired input | Complete-only Edge-GNN dataset; 797 queries |
| `router_model_only_qnorm.jsonl` | Pending alignment check | Input used by qnorm model-only/static analyses |
| `router_model_only_direct_qnorm.jsonl` | Derived/stale unless rebuilt | Must be rebuilt from the complete-only source |

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

## Problems pending

1. Ensure the qnorm model-only dataset is filtered/rebuilt for the same 797
   queries before static baseline evaluation.
2. Rebuild `router_model_only_direct_qnorm.jsonl` from the complete-only
   bipartite dataset.
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

Rebuild both model-only inputs from the same 797-query population before any
training. The next code change is to make the model-only dataset builder accept
the complete bipartite dataset as its qid allowlist and produce:

- `router_model_only_qnorm_complete.jsonl`; and
- `router_model_only_direct_qnorm_complete.jsonl`.

After validating that both contain exactly the same 797 qids, run all three
methods with the shared seed-1 split manifest. Do not start seed 1 before these
derived inputs are aligned.
