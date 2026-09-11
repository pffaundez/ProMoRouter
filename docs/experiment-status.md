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
- All four embedding files exist locally under `data/router/`; their internal
  IDs, dimensions, and coverage have not yet been inspected.
- The main Edge-GNN scorer does not consume realized reward, performance, cost,
  token count, or monetary cost as numeric routing-time features.
- Offline reward labels do affect Edge-GNN topology: top-k training actions are
  selected by reward to form observed structural edges.
- The previous GraphRouter-direct implementation consumed realized cost and
  token-use features. Those inputs have been removed on the repair branch.
- The previous GraphRouter-direct implementation could misalign model
  embeddings by relying on order. Model embeddings are now aligned by ID.

### Open hypotheses

- The query-embedding file likely contains the two removed incomplete queries
  as harmless extras; this must be verified.
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

Their semantic contents and ID coverage are pending verification.

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

1. Inspect embedding IDs, dimensions, and coverage against the 797-query input.
2. Ensure the qnorm model-only dataset is filtered/rebuilt for the same 797
   queries before static baseline evaluation.
3. Rebuild `router_model_only_direct_qnorm.jsonl` from the complete-only
   bipartite dataset.
4. Run Edge-GNN, GraphRouter-direct, and static baselines on the same seed-1
   manifest.
5. Compare seed-1 action distributions and confirm no fixed-pair collapse.
6. Run seeds 1--5 only after seed 1 passes.
7. Regenerate every (P,C,R) table from machine-readable outputs.
8. Recreate the routing-configuration ablation with the corrected pipeline.
9. Rewrite the paper's graph construction, scorer, and loss so they match the
   actual implementation.
10. Decide whether query-normalized cost is the final deployment cost
    definition and document its limitations.
11. Retire or isolate legacy pre-qnorm scripts and artifacts.
12. Correct remaining README paths and commands that refer to absent or legacy
    files.

## Exact next step

Inspect embedding coverage and identifier alignment before any training run.
From `~/repos/ProMoRouter`, with the existing environment active, run:

```bash
python - <<'PY'
import json
import torch
from pathlib import Path

root = Path.home() / "repos/graph-router-2"
data = root / "data/interaction_logs/grpp_il_v1/router_bipartite_qnorm_complete.jsonl"
emb = root / "data/router"

rows = [json.loads(line) for line in data.open() if line.strip()]
dataset_qids = {row["qid"] for row in rows}

q = torch.load(emb / "query_embeddings.pt", map_location="cpu")
query_qids = set(q) if isinstance(q, dict) else set()

print("Dataset queries:", len(dataset_qids))
print("Query embeddings:", len(query_qids))
print("Missing query embeddings:", len(dataset_qids - query_qids))
print("Extra query embeddings:", len(query_qids - dataset_qids))

for name in ["task", "prompt", "model"]:
    obj = torch.load(emb / f"{name}_embeddings.pt", map_location="cpu")
    print(f"\n{name}:")
    print("  ids:", obj.get("ids"))
    print("  shape:", tuple(obj["embeddings"].shape))
PY
```

Expected, but not yet verified:

- zero missing query embeddings;
- two extra query embeddings;
- four task IDs;
- four prompt IDs; and
- nine model IDs.

Do not start seed 1 until this check passes.
