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
  one current action: T-009.
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
- Cross-input validation passed on the experiment host: each input has 797
  queries and the combined validator reported zero errors.
- Edge-GNN seed 1 completed for all three lambdas on 121 test queries using
  the repaired complete population and dedicated split manifest.
- Seed-1 results were P=0.498273, C=0.389587, R=0.459314 for lambda=0.1;
  P=0.417263, C=0.116930, R=0.358799 for lambda=0.5; and
  P=0.410675, C=0.107292, R=0.314112 for lambda=0.9.
- GraphRouter-direct seed 1 completed on the same manifest with P=0.406660,
  C=0.390340, R=0.367626 for lambda=0.1; P=0.364210, C=0.224417,
  R=0.252001 for lambda=0.5; and P=0.315689, C=0.100702,
  R=0.225057 for lambda=0.9.
- GraphRouter-direct selected 4, 6, and 4 distinct models across the three
  lambdas; no single-model collapse was observed.

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
- Cross-input validation completed on the real repaired inputs: 797 queries in
  each dataset, zero validation errors, and aligned populations, candidates,
  tasks, and rewards.
- Edge-GNN seed 1 completed and produced three model files plus the Overleaf
  row; all test splits contain 121 queries and printed R values match P-lambda*C
  to rounding precision.
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
- A repaired Edge-GNN seed-1 run now exists for all three lambda settings.
- A repaired GraphRouter-direct seed-1 run now exists for all three lambda
  settings on the shared manifest.
- Static seed-1 baseline results now exist for all three lambda settings. The
  machine-readable file stores their metrics under each lambda's nested
  `results` object; the top-level lambda entries do not themselves hold P/C/R.
- Seed-1 integrity gate passed with a behavioral warning: every method reports
  n=121, learned-router rewards satisfy P-lambda*C, and Edge-GNN exceeds Best
  Fixed Pair in reward by about 0.002 for each lambda.

## Problems pending

1. Run GraphRouter-direct and static baselines on the same seed-1 manifest.
2. Inspect seed-1 action distributions, including the lambda=0.1 selfcheck
   pattern, and confirm no fixed-pair collapse.
3. Run seeds 1--5 only after the seed-1 gate passes.
4. Regenerate every (P,C,R) table from machine-readable outputs.
5. Recreate the routing-configuration ablation with the corrected pipeline.
6. Rewrite the paper's graph construction, scorer, and loss so they match the
   actual implementation.
7. Decide whether query-normalized cost is the final deployment cost
   definition and document its limitations.
8. Retire or isolate legacy pre-qnorm scripts and artifacts.
9. Correct remaining README paths and commands that refer to absent or legacy
   files.

## Exact next step

Execute T-009 from `docs/next-steps.md`: run seeds 2--5 for Edge-GNN,
GraphRouter-direct, and static baselines, creating one validated split manifest
per seed and preserving the seed-1 outputs. Investigate whether the lambda=0.1
`selfcheck) concentration persists across seeds.


## Synchronization update (2026-09-14)

The repaired five-seed sweep is complete. Verified output files exist for seeds 1--5 for Edge-GNN, GraphRouter-direct, and static baselines under outputs/p0_*_qnorm/.

Aggregated test rewards (mean +/- sample standard deviation over five seeds):

| Method | lambda=0.1 R | lambda=0.5 R | lambda=0.9 R |
|---|---:|---:|---:|
| Edge-GNN | 0.4815 +/- 0.0368 | 0.3830 +/- 0.0357 | 0.3557 +/- 0.0343 |
| GraphRouter-direct | 0.4245 +/- 0.0542 | 0.2895 +/- 0.0599 | 0.2322 +/- 0.0308 |
| Best Fixed Pair | 0.4894 +/- 0.0409 | 0.3709 +/- 0.0219 | 0.3239 +/- 0.0215 |

Fixed-pair identities are stable across all five seeds: llama3.1-70b + selfcheck for lambda=0.1 and qwen2.5-7b + selfcheck for lambda=0.5 and 0.9. These are verified outputs and supersede the earlier pending sweep status.

Embeddings and generated outputs remain experiment-host artifacts and are not present in the public Git tree unless explicitly added. See docs/methodology-scope.md for offline/online and closed-pool scope.


## Synchronization update (2026-09-15)

A Flat 36-action MLP baseline was added in `train_router_flat_mlp_qnorm.py`.
It uses the same complete 4x9 action space, qnorm rewards, embeddings, and
persisted splits as Edge-GNN, but no graph/message passing; each prompt-model
pair is scored from concatenated query, prompt, and model embeddings. This is
an implementation addition; its results are not yet available.


## Flat-MLP seed-1 update (2026-09-16)

The full Flat 36-action MLP seed-1 run completed with finite early stopping
(32, 33, and 36 epochs for lambda 0.1, 0.5, and 0.9). Test results on the
shared 121-query split were: lambda=0.1 P=0.551948, C=0.476123, R=0.504335;
lambda=0.5 P=0.478261, C=0.167827, R=0.394347; lambda=0.9 P=0.366372,
C=0.061860, R=0.310698. The policy selected multiple prompts and models in
all three settings; no fixed-pair collapse was observed. These are verified
seed-1 baseline results, not five-seed aggregates.


## Flat-MLP five-seed update (2026-09-16)

The Flat 36-action no-graph MLP sweep completed for seeds 1--5. Every run
used 121 test queries and the shared qnorm manifests. Aggregated test results
(mean +/- sample standard deviation) are:

| lambda | P | C | R |
|---:|---:|---:|---:|
| 0.1 | 0.5660 +/- 0.0392 | 0.4644 +/- 0.0576 | 0.5195 +/- 0.0429 |
| 0.5 | 0.4409 +/- 0.0547 | 0.1087 +/- 0.0487 | 0.3866 +/- 0.0500 |
| 0.9 | 0.4106 +/- 0.0338 | 0.0635 +/- 0.0073 | 0.3534 +/- 0.0297 |

Relative to the repaired Edge-GNN aggregates, Flat-MLP has higher reward at
lambda=0.1 and 0.5 and is nearly tied at lambda=0.9 (0.3534 vs 0.3557). This
verified result means the current evidence does not isolate a broad graph
advantage; the graph contribution must be reported cautiously and further
architectural analysis remains required.


## Inductive evaluation protocol update (2026-09-16)

The target extension is an inductive protocol with ten total models and five
total prompting strategies. Training uses the existing nine-model/four-prompt
pool; one additional model and one additional prompt are held out as unseen
candidates and introduced only at inference through description-derived
embeddings. Evaluation also includes prompt-model combinations masked from
training. This protocol is a planned extension, not a completed result.

A valid evaluation requires static descriptions and pre-execution metadata for
the new candidates, plus interaction outcomes for their test actions. The
current nine-model/four-prompt logs alone cannot establish performance or cost
for an external tenth model or fifth prompt.


## Design documents added (2026-09-16)

- `docs/cost-information-contract.md` defines the separation between pre-routing estimated cost and post-execution realized cost.
- `docs/inductive-evaluation-protocol.md` defines closed-pool, unseen-model, unseen-prompt, joint-unseen, masked-pair, zero-shot, and few-shot conditions.
- D-016 confirms the concrete unseen set: three models and two prompt strategies.
These documents are protocol specifications; no unseen-candidate result has been produced yet.


## Inductive candidate manifest update (2026-09-16)

The confirmed candidate manifest is now stored at
`configs/inductive_candidates.yaml`. It defines 12 inference models (9 seen
and 3 unseen) and 6 prompting strategies (4 seen and 2 unseen), for 72
prompt--model actions. The unseen entries are SmolLM2-1.7B-Instruct,
OLMo-2-1124-7B-Instruct, Phi-4, fewshot, and self_consistency.

This is a configuration artifact, not an experimental result. Backend,
license, description-embedding, and held-out outcome checks remain pending.
P0/P1 closed-pool datasets and embeddings were not modified.


## Inductive embedding builder update (2026-09-16)

Added `scripts/build_inductive_description_embeddings.py`. It reads
`configs/inductive_candidates.yaml` and writes only
`data/router/inductive_embeddings/`, preserving all closed-pool P0/P1
embedding artifacts. The script has not yet been executed; backend, license,
and model-loading checks remain pending.


## Inductive embedding dependency update (2026-09-16)

The first embedding attempt failed because the active virtual environment does
not provide `sentence-transformers`. The repository requirement was added to
`requirements.txt`; no embeddings were generated and no P0 artifacts changed.


## Inductive embeddings generated (2026-09-16)

The inductive builder completed successfully with the
`sentence-transformers/all-MiniLM-L6-v2` encoder. Metadata reports three
unseen model IDs and two unseen prompt IDs, with shapes 3x384 and 2x384,
respectively. The artifacts are isolated under
`data/router/inductive_embeddings/`; closed-pool embeddings remain unchanged.
Model loading, prompt execution, and held-out outcome generation are still
unverified.


## Inductive candidate validator added (2026-09-16)

Added `analysis/validate_inductive_candidates.py` to verify the confirmed
9/3 model split, 4/2 prompt split, non-overlap, and exact 3x384/2x384
embedding artifacts. The validator has not yet been run in the user's local
checkout.


## Inductive candidate validation passed (2026-09-16)

The local run of `analysis/validate_inductive_candidates.py` passed. It
verified the 9 seen + 3 unseen model split, 4 seen + 2 unseen prompt split,
non-overlap, and exact embedding shapes (3x384 and 2x384). Model loading and
prompt execution remain unverified.


## Unseen model configuration smoke test passed (2026-09-16)

`AutoConfig.from_pretrained` successfully resolved all three candidates:
SmolLM2-1.7B-Instruct (`llama`), OLMo-2-1124-7B-Instruct (`olmo2`), and
Phi-4 (`phi3`). This checked configuration accessibility only; no full
weights were loaded and generation compatibility remains pending.


## Unseen generation smoke test: SmolLM2 (2026-09-16)

SmolLM2-1.7B-Instruct loaded successfully with Transformers and generated the
expected answer to a minimal arithmetic prompt (`2+2 equals 4`). OLMo-2 and
Phi-4 generation tests remain pending; no router experiment has been run with
unseen candidates.


## Unseen generation smoke tests complete (2026-09-16)

OLMo-2-1124-7B-Instruct and Phi-4 also loaded and generated successfully by
changing only the `model_id` in the minimal Transformers test. All three
unseen model candidates therefore pass configuration and basic generation
checks. This does not yet validate the router's candidate integration,
costs, or the fewshot/self_consistency execution contracts.


## Unseen prompt templates added (2026-09-16)

Added `fewshot` and `self_consistency` execution templates to
`configs/prompt_templates.yaml`. The templates define the prompt text only;
demonstration selection, sample count, aggregation, decoding parameters, and
cost accounting still require an executable evaluation harness.


## Unseen prompt budgets confirmed (2026-09-16)

The execution budgets are fixed: `fewshot` uses exactly two demonstrations
per task; `self_consistency` uses exactly three independent samples and
majority aggregation over normalized final answers. Demonstration content,
decoding parameters, and realized costs remain to be specified.


## Deterministic unseen prompt execution configured (2026-09-16)

Added `configs/inductive_prompt_execution.yaml`. Few-shot calibration uses
the first two valid training qids per task (sorted lexicographically), excluding
validation and test. Self-consistency uses three samples with fixed decoding
parameters and deterministic normalized-majority aggregation. No held-out
interactions or realized costs have been generated yet.


## Prompt demonstration source audit (2026-09-16)

The first record of `router_bipartite_qnorm_complete.jsonl` contains
`query_text` and action outcomes, but no response/completion text. Therefore
the qnorm log cannot supply few-shot demonstrations. Demonstrations must come
from a separate benchmark/training source, with IDs and answers recorded
without using validation or test outcomes. No demonstrations have been
fabricated and no held-out interactions have been generated.
