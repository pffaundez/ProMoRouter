# ProMoRouter experiment status

_Last updated: 2026-09-18_

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
  one current action: T-025.
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


## Unseen prompt strategy revised (2026-09-16)

Because the qnorm log lacks response text, the proposed two-shot `fewshot`
strategy was replaced by `step_back`, which requires only a static instruction.
`self_consistency` remains with three samples. The unseen prompt set is now
`step_back` and `self_consistency`; no demonstration dataset is required.


## Unseen prompt smoke test passed (2026-09-16)

Using SmolLM2-1.7B-Instruct, the `step_back` template generated a principle,
application, and final answer. The `self_consistency` contract generated
exactly three samples with the configured sampling parameters; all three
answered the arithmetic test consistently. This validates prompt execution
only, not routing performance or held-out cost.


## Gold-reference source verified (2026-09-16)

`configs/rq2_dataset_builder_smoke.yaml` specifies reproducible Hugging Face
sources and gold fields: `hotpot_qa/distractor` (`answer`), `gsm8k/main`
(`answer`), `squad` (`answers`), and `tatsu-lab/alpaca` (`output`).
Manual gold-answer authoring is not required and is discouraged. GSM8K,
HotpotQA, and SQuAD have automatic metrics in the existing builder; Alpaca is
configured for judge-based evaluation and needs an explicit judge policy.


## Alpaca metric aligned with implementation (2026-09-16)

Updated `configs/rq2_dataset_builder_smoke.yaml` to declare automatic
token-level F1 for Alpaca, matching the builder's `alpaca_f1` scorer. The
decision avoids introducing a new judge and preserves comparability. Historical
P0 logs still require an audit confirming the same scorer was used.


## Alpaca F1 provenance verified (2026-09-16)

The raw `train_clean_qnorm_lambdas.jsonl` contains response-level Alpaca
records with `performance.metric = f1`. A join against
`router_bipartite_qnorm_complete.jsonl` checked 7,200 Alpaca action edges with
zero missing keys and zero performance mismatches. Alpaca may therefore be
included in the inductive evaluation using automatic token-level F1.


## Endpoint availability check (2026-09-16)

The local Ollama-compatible endpoint at `localhost:11434/v1` is unavailable,
and the active environment has no `vllm` executable. Since all three unseen
models already pass direct Transformers generation, the inductive interaction
generator should use a sequential Transformers backend rather than requiring a
new vLLM installation. No held-out interactions have been generated yet.


## Inductive protocol boundary confirmed (2026-09-17)

D-020 fixes the evaluation boundary: P0 remains frozen with its vLLM-generated
closed-pool results. The inductive protocol will use a separate test manifest
and regenerate all 72 actions with one sequential Transformers backend, so seen
and unseen candidates are comparable within that protocol. Results will not be
numerically merged with P0.


## Unified inductive generator added (2026-09-17)

Added `experiments/generate_inductive_transformers.py`. It loads one candidate
model at a time, evaluates the 12x6 action space on the persisted test qids,
supports `--max-queries` and `--dry-run`, and records response, performance,
tokens, latency, backend, and a token-count cost proxy. Hugging Face
`datasets` was added to `requirements.txt`. The generator has not yet been
run beyond smoke validation; monetary cost normalization remains a separate
post-processing step.


## Inductive generator smoke-test fix (2026-09-17)

The first real smoke test reached dataset loading but failed because the
generator loaded every configured task, including HumanEval whose cache exposed
only a `test` split. The generator now loads only tasks represented in the
selected qids and falls back to `test` only when the requested split is
unavailable. No interaction output was produced by the failed run.


## Inductive generator syntax fix (2026-09-17)

A literal escaped newline introduced during the task-loading patch caused a
syntax error in the first rerun. The source line was corrected; no data or
output artifacts were produced by the failed invocation.


## Inductive generator escaped-newline fix verified (2026-09-17)

A second correction removed the remaining literal `\\n` token from the task
loading line. The repository source now contains a valid newline and is ready
for the targeted smoke rerun.


## Inductive smoke output audit (2026-09-17)

The first smoke JSONL was structurally produced, but its `response` field
included the serialized prompt and assistant transcript because the generator
decoded the full sequence. This is a metric-invalid output. The generator was
fixed to decode only newly generated tokens; the smoke test must be rerun.


## Inductive answer-extraction fix (2026-09-17)

The targeted SmolLM2 smoke output exposed a second correctness issue: for self_consistency, majority voting compared complete generated strings, so equivalent answers with different wrappers (for example, “The final answer is: ...” versus the bare answer) were treated as different classes. The generator now extracts the final non-empty line, removes common answer prefixes, uses that normalized answer for voting and scoring, and preserves the raw generated samples in the audit field. The previous smoke output must not be used for metrics; the targeted smoke test must be rerun after pulling commit e1cefe0.


## Inductive prefix-normalization correction (2026-09-17)

The rerun confirmed that transcript removal works, but exposed a second prefix form: `The final answer is ...` was not matched by the extractor, so self-consistency still voted on three distinct strings. The extractor now accepts both colon and `is` forms. The smoke output shown in the conversation remains diagnostic only and must be regenerated after commit f77591f.


## Inductive prefix extraction finalization (2026-09-17)

Local verification showed that the form `The final answer is: ...` could retain a leading colon after stripping `is`. The extractor now removes the complete `is`, `is:`, or colon prefix before self-consistency voting. Commit: ae8832b.


## Inductive smoke extraction validated; gold alignment pending (2026-09-17)

The corrected smoke output now has the expected structure: `self_consistency.response` is `Allie Goertz`, the three raw samples are preserved, and no prompt transcript appears in the scored response. However, both smoke rows report `performance=0.0`. This is not evidence of model quality yet; it indicates that the generator's dataset index/gold extraction or metric normalization must be audited for the selected HotpotQA qid before the 72-action inductive run. Full inductive generation is paused until this alignment check passes.


## Inductive smoke gold audit resolved (2026-09-17)

The selected HotpotQA example was audited with the same loader used by the generator. Its gold answer is `President Richard Nixon`, while SmolLM2 generated `Allie Goertz` for `step_back` and did not reach the correct answer under self-consistency. Therefore `performance=0.0` is a genuine model error on this smoke query, not a qid, gold-field, extraction, or metric bug. The structural smoke validation passed; these two rows remain diagnostic and are not paper results.


## Complete inductive generation finished (2026-09-18)

The unified Transformers generation completed with exactly 8,712 rows: 121 test queries, 12 models, and 6 prompt strategies (72 actions per query). All rows use the Transformers backend and have non-null performance. Two rows have empty responses and require inspection before the dataset is accepted as complete; no aggregate inductive result should be reported yet.


## Inductive empty-response audit (2026-09-18)

The two empty `response` fields are caused by max-token truncation, not silent model failures. Both raw responses end at `The final answer is` with `output_tokens=256`: GSM8K/lLaMA-3.1-8B under `cot`, and Alpaca/lLaMA-3.1-70B under `step_back`. The generator now supports `--qids` for targeted regeneration. These rows must be regenerated with a larger token cap before final validation.


## Inductive dataset completed and repaired (2026-09-18)

Verified on the experiment host:

- `outputs/inductive_full.jsonl` initially contained exactly 8,712 rows (121 test qids × 12 models × 6 prompts).
- All rows used the Transformers backend and had non-null performance.
- Two rows were truncated at the 256-token cap and had empty extracted responses.
- Targeted regeneration with `--qids` and `--max-new-tokens 512` repaired both rows:
  - GSM8K `gsm8k-train-000072`, `llama3.1-8b` + `cot`: response `54`, performance 1.0.
  - Alpaca `alpaca-train-000102`, `llama3.1-70b` + `step_back`: response with POS tags, performance 0.5.

The repair files were received and inspected. They must be merged into a corrected full JSONL, followed by validation of exact population, uniqueness, non-empty responses, and metric fields. Until that merge/validation is run, the inductive dataset is not yet a final paper artifact.


## Repair merge tool added (2026-09-18)

Because the full inductive JSONL remains on the experiment host and was not available in this workspace, the merge was not executed remotely. Added `analysis/merge_inductive_repairs.py`, which deterministically replaces the two repair keys and validates 8,712 rows, 121 queries, 12 models, 6 prompts, unique keys, non-empty responses, and repaired performance values. The host-side command is the required finalization step.


## Corrected inductive artifact finalized (2026-09-18)

The host-side merge script completed successfully and produced `outputs/inductive_full_corrected.jsonl` with exactly 8,712 rows. The merge script itself validated replacement of both truncated keys, 121 queries, 12 models, 6 prompts, unique action keys, non-empty responses, non-null performance, and repaired performances (GSM8K 1.0; Alpaca 0.5). This is now the canonical inductive artifact for downstream analysis; the original `inductive_full.jsonl` remains the pre-repair audit artifact.


## Inductive action aggregates completed (2026-09-18)

T-024 is complete. Added `analysis/aggregate_inductive_results.py` and ran it
against the corrected artifact. The script revalidates the exact 8,712-row,
121-query, 12-model, 6-prompt Cartesian coverage and writes the machine-readable
aggregate `analysis/inductive_aggregates.json`. Its source artifact SHA-256 is
`a9235215fd7d4d7f38a087696fb316dbe5fa39efaa8509eef02dfa3349477273`.

### Verified descriptive results

- Across all 8,712 actions, mean performance is 0.341804 and mean realized
  token cost proxy is 288.795 tokens (2,515,978 tokens total).
- Seen models average 0.343011 performance and 289.631 tokens; unseen models
  average 0.338185 performance and 286.284 tokens. This is a descriptive
  performance difference of -0.004826 for the selected unseen model set.
- Seen prompts average 0.335602 performance and 238.898 tokens; unseen prompts
  average 0.354209 performance and 388.588 tokens. The +0.018607 performance
  difference comes with +149.691 mean tokens.
- The unseen-prompt aggregate is heterogeneous: `self_consistency` has the
  highest prompt mean performance (0.418787) and the highest mean token cost
  (507.337), whereas `step_back` averages 0.289631 performance and 269.840
  tokens.
- The joint unseen-model/unseen-prompt quadrant averages 0.337349 performance
  and 391.003 tokens, versus 0.334602 performance and 240.555 tokens for the
  seen-model/seen-prompt quadrant.
- The 36 prompt-model pairs involving at least one unseen node were not present
  in the training pool. They average 0.349007 performance and 337.034 tokens,
  versus 0.334602 and 240.555 for the 36 seen-model/seen-prompt pairs.
- Using per-query min-max normalized token cost across all 72 actions, the
  unobserved-node pairs have mean rewards 0.313729, 0.172619, and 0.031508 for
  lambda 0.1, 0.5, and 0.9. The observed pairs have 0.314724, 0.235215, and
  0.155706 respectively. These reward values belong only to the standalone
  inductive Transformers protocol and are not merged with P0.

### Limits and open hypotheses

- These are action-outcome aggregates, not router selections. They do not by
  themselves demonstrate that a trained router can identify the best unseen
  action, nor do they provide selection frequency or oracle regret.
- No manifest declares seen-model/seen-prompt pairs deliberately masked during
  training. Therefore the separate masked-pair condition in the protocol
  cannot be evaluated from this artifact; only pairs unobserved because they
  contain an unseen node are identified.
- No inferential test or multi-seed generation was performed. Apparent mean
  differences are verified descriptive facts, but explanations and
  generalization claims remain hypotheses.


## Canonical P0 aggregator prepared (2026-09-18)

T-010 implementation is prepared but has not been executed. Added
`analysis/aggregate_p0_closed_pool_results.py` for the five seed-level JSON
files from each of Edge-GNN, GraphRouter-direct, static/oracle baselines, and
Flat-MLP. The script:

- requires exactly seeds 1--5 and all three qnorm lambda keys;
- normalizes the four distinct output schemas;
- requires 121 test queries per method, seed, and lambda;
- checks finite P/C/R values and `R = P - lambda * C` within `1e-8`;
- checks learned-method split-manifest basenames against
  `qnorm_complete_seed<seed>.json`;
- checks stored action-count distributions when present;
- records SHA-256 for all 20 source JSON files; and
- writes standalone P0 JSON, CSV, and LaTeX aggregates using sample standard
  deviation across seeds.

The static-baseline result schema does not serialize `split_manifest`. The
aggregator therefore validates its 121-query coverage but records that exact
manifest identity requires the original generation command or a separate
provenance artifact. No inductive artifact is read by this script.


## Canonical P0 five-seed tables completed (2026-09-18)

T-010 is complete. The aggregator ran on `morel` and reported 20 validated
source JSON files, 135 normalized seed-level result rows, and 27 aggregate rows
covering nine methods and three lambda settings. The returned JSON, CSV, and
LaTeX artifacts were cross-checked for matching method sets and exact numeric
agreement. Every normalized result records 121 test queries and satisfies
`R = P - lambda * C` within the configured `1e-8` tolerance.

Canonical files:

- `analysis/p0_closed_pool_aggregates.json`;
- `analysis/p0_closed_pool_table.csv`; and
- `analysis/p0_closed_pool_table.tex`.

### Verified aggregate results

- Flat-MLP has the highest learned-method reward at lambda 0.1
  (`0.5195 +/- 0.0429`) and lambda 0.5 (`0.3866 +/- 0.0500`).
- Edge-GNN has reward `0.4815 +/- 0.0368`, `0.3830 +/- 0.0357`, and
  `0.3557 +/- 0.0343` for lambda 0.1, 0.5, and 0.9 respectively.
- At lambda 0.9, Edge-GNN and Flat-MLP remain close: `0.3557` versus `0.3534`
  mean reward. This does not establish a broad graph advantage.
- Best Fixed Pair exceeds Edge-GNN at lambda 0.1 (`0.4894` versus `0.4815`),
  while Edge-GNN exceeds it at lambda 0.5 (`0.3830` versus `0.3709`) and
  lambda 0.9 (`0.3557` versus `0.3239`).
- GraphRouter-direct has lower mean reward than both joint learned methods at
  all three lambda settings: `0.4245`, `0.2895`, and `0.2322`.

These facts were reproduced by the host-side aggregator and verified from the
returned aggregate artifacts. The continuation workspace did not receive the
20 raw seed JSON files, so it did not independently recompute their hashes or
metrics. Static-baseline JSON files still lack serialized split-manifest paths;
their exact split provenance remains supported by the recorded generation
procedure rather than by fields inside those JSON files.


## T-011 routing ablation prepared (2026-09-18)

Repository inspection confirmed that the four optional ablation scripts named
in the README do not exist. The canonical Edge-GNN trainer already exposes the
underlying controls through `--edge-top-k` and
`--full-prompt-model-lattice`. D-023 therefore defines a controlled 2x2 matrix:

| Configuration | Observed top-k | Full prompt-model lattice |
|---|---:|---:|
| `topk3_observed` | 3 | no |
| `topk5_observed` | 5 | no |
| `topk3_lattice` | 3 | yes |
| `topk5_lattice` | 5 | yes |

All four configurations retain the repaired 797-query population, the same 36
candidate actions, shared seed manifests, current loss, and validation-reward
early stopping. The lattice flag adds prompt-model edges; it does not connect
validation/test queries to all prompts and models.

Added:

- `configs/p0_routing_ablation.json` as the machine-readable contract;
- `scripts/run_p0_routing_ablation.py` with seed/config selection, dry-run,
  resume-by-default, and explicit overwrite support; and
- `analysis/aggregate_p0_routing_ablation.py` with coverage, split, count,
  reward-algebra, source-hash, mean, and sample-standard-deviation checks.

Syntax compilation and dry runs passed: four commands are planned for the seed-1
gate and 20 commands for the complete five-seed sweep. No ablation training or
result aggregation has run in the continuation workspace; T-011 results remain
pending execution on `morel`.

The first host-side gate attempt stopped before training because the initial
runner resolved `data/` inside the ProMoRouter checkout, while the experiment
artifacts on `morel` live under `~/repos/graph-router-2`. The runner now accepts
`--artifact-root` and resolves only data, embedding, and split-manifest paths
against that root; code and outputs remain in ProMoRouter. This is an execution
path fix, not a protocol change. No ablation result was produced by the failed
attempt.


## T-011 nondeterminism gate failed (2026-09-18)

The first 2x2 seed-1 gate completed with valid coverage, reward algebra, and
non-collapsed action distributions. However, a direct repetition of
`topk3_observed` with the same seed and stored configuration did not reproduce
the first run. Reward differences between the two current-code runs were
`+0.01348`, `-0.01847`, and `-0.00522` for lambda 0.1, 0.5, and 0.9. P and C
also changed materially. Checkpoint metadata confirmed identical declared
configuration, so the full sweep was not authorized.

Inspection identified CUDA `index_add_` accumulation in message aggregation,
degree counting, and the entropy term without deterministic-algorithm mode.
D-024 adds an explicit deterministic mode using non-atomic one-hot matrix
aggregation plus deterministic PyTorch/CUDA/cuDNN settings. Frozen P0 remains
unchanged. T-011 must now pass two identical deterministic seed-1 repetitions
before the 2x2 gate is rerun.


## T-011 deterministic routing ablation completed (2026-09-18)

T-011 is complete. Two independent deterministic `topk3_observed` seed-1
runs were identical after excluding only the output-specific `model_path`.
Each contained 121 test queries for all three lambda settings, all prompt,
model, and task counts summed to 121, and every result satisfied
`R = P - lambda * C`. The nondeterministic gate outputs remain diagnostic only.

The deterministic 2x2 sweep then completed for seeds 1--5. The reproducible
aggregator validated 20 exact source files, 60 seed-level rows, and 12
configuration/lambda aggregates. All 20 source hashes are unique, every group
contains seeds 1--5, every result contains 121 test queries, and reward algebra
holds to a maximum observed absolute error of `4.44e-16`. The returned JSON and
CSV agree exactly. Canonical artifacts:

- `analysis/p0_routing_ablation_aggregates.json` (SHA-256
  `b2c7a65eb766c75aab338f5adfc70e6c91caf94d5c744145d8dc53d72afdf6ce`);
- `analysis/p0_routing_ablation_table.csv` (SHA-256
  `c982825bab7908f9479b9ee8adbfe2c4bf258e29b5f3edc60f24768ad0cb2745`).

### Verified aggregate results

| Configuration | lambda | Performance | Cost | Reward |
|---|---:|---:|---:|---:|
| top-k 3, observed | 0.1 | 0.5037 | 0.3241 | 0.4713 +/- 0.0261 |
| top-k 3, observed | 0.5 | 0.4514 | 0.1274 | 0.3877 +/- 0.0337 |
| top-k 3, observed | 0.9 | 0.4159 | 0.0867 | 0.3378 +/- 0.0371 |
| top-k 5, observed | 0.1 | 0.5221 | 0.3509 | 0.4870 +/- 0.0504 |
| top-k 5, observed | 0.5 | 0.4423 | 0.1295 | 0.3776 +/- 0.0428 |
| top-k 5, observed | 0.9 | 0.4151 | 0.0772 | 0.3456 +/- 0.0211 |
| top-k 3, lattice | 0.1 | 0.5037 | 0.3241 | 0.4713 +/- 0.0261 |
| top-k 3, lattice | 0.5 | 0.4515 | 0.1309 | 0.3861 +/- 0.0308 |
| top-k 3, lattice | 0.9 | 0.4194 | 0.0939 | 0.3349 +/- 0.0312 |
| top-k 5, lattice | 0.1 | 0.5221 | 0.3509 | 0.4870 +/- 0.0504 |
| top-k 5, lattice | 0.5 | 0.4423 | 0.1295 | 0.3776 +/- 0.0428 |
| top-k 5, lattice | 0.9 | 0.4151 | 0.0772 | 0.3456 +/- 0.0211 |

Increasing top-k from 3 to 5 changes mean reward by `+0.0156`, `-0.0101`,
and `+0.0078` without the lattice for lambda 0.1, 0.5, and 0.9. This is not a
consistent benefit across cost regimes. Enabling the full prompt--model lattice
changes mean reward by `0.0000`, `-0.0016`, and `-0.0029` at top-k 3 and by
exactly `0.0000` at top-k 5 for all lambdas. Several lattice on/off results are
identical for every seed, so this ablation provides no evidence of a general
lattice benefit.

These are descriptive five-seed results, not inferential significance claims.
The lattice result does not evaluate GraphRouter-style test-query connectivity:
the current flag only adds prompt--model edges. The exact equality in several
cells is verified; whether it is caused by optimization convergence, graph
symmetry, or limited influence of those relations is an open hypothesis.


## Edge-GNN v2 minimal controlled design frozen (2026-09-18)

D-026 defines the next experiment before implementation. Stage A will hold the
36-action space, repaired population, embeddings, split manifests, reward,
current objective, early stopping, seeds, and scorer capacity fixed while
comparing: (1) GraphRouter-style full message passing with explicit
query--task, query--prompt, and query--model connections; (2) an otherwise
matched no-message-passing variant; and (3) the existing Flat-MLP. Prompt and
model remain independent nodes and the scorer consumes only pre-routing
representations, optionally including a static prompt--model edge embedding.

Only after Stage A separates architectural message passing from the common
action space may Stage B compare the current objective with
`L_reg + beta * L_rank`, using validation reward for early stopping. Graph
density variants and a factorized router are conditional follow-ups rather
than simultaneous changes. The existing corrected inductive artifact remains
a separate Stage C evaluation and must not be mixed with P0. Edge-GNN v2 has
not been implemented or executed; all expected advantages remain hypotheses.


## Edge-GNN v2 Stage A contract completed (2026-09-18)

T-025 is complete. The pre-implementation contract is frozen in
`docs/edgegnn-v2-stage-a.md` and
`configs/edgegnn_v2_stage_a.json`. It resolves two implementation-level
confounds found during repository inspection:

- Stage A uses an independent ego graph per query, preventing message passing
  between train, validation, or test queries.
- The primary comparison is full message passing versus an otherwise matched
  no-message-passing arm with the same edge-aware scorer. The Flat-MLP
  architecture is retrained under the shared Stage A objective in a new output
  namespace rather than reusing historical results trained with a different
  loss.

Every ego graph contains one query, its task, all four prompts, and all nine
models, with complete bidirectional query--task, query--prompt, and
query--model relations. It has no reward-selected top-k topology and no
prompt--model lattice. The edge representation is compositional from prompt
and model description embeddings; pair-ID lookups and all realized outcomes
are prohibited.

The contract fixes the repaired 797-query population, 36 actions, seeds 1--5,
shared manifests, three lambdas, current Edge-GNN objective, validation-reward
early stopping, deterministic execution, and expected aggregation cardinality
(15 sources, 45 seed-level rows, nine aggregates). Edge-GNN v2 remains a
proposal: no implementation, checkpoint, or result has been produced.
