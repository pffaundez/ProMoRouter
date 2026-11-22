# Official GraphRouter fixed-Direct control

Question: does a pinned official GraphRouter encoder use Qwen's support to
improve model-only routing on our existing data? One training, four frozen
conditions; no architecture or hyperparameter sweep.

## Provenance and implementation fidelity

Upstream: https://github.com/ulab-uiuc/GraphRouter
Commit: `0edfb8758d45acd93c8029354867ffb49bb8a97f`.
`router/graphrouter_official.py` copies FeatureAlign and EncoderDecoderNet
unchanged from upstream model/graph_nn.py (MIT notice retained). Use the code's
query/task concatenation, directed query-to-model graph, two GeneralConv layers,
edge MLP, BatchNorm and sigmoid compatibility decoder. No prompt nodes, pair
states or support summarizer are substituted. Edge attributes remain ordered
as [reward, cost, performance]. GeneralConv aggregation is PyG's implementation.

This is an adapted official-code control, NOT a reproduction of paper numbers.
Differences: our nine models, four datasets/embeddings, seed1 persisted splits,
fixed Direct outcomes, R=P-0.9C, and training-only monetary normalization.
Official config.yaml values retained: embedding_dim=8 (internal width 16),
AdamW lr=.0003, weight decay=.0001, mask rate=.5, 32 mask draws per optimizer
step, 1000 epochs. The paper's descriptive hyperparameters differ from the
public config; this control deliberately pins the public code/config.

## Training and selection

Use existing 797-query manifest: 549 fitting targets, 8 small support, 119
validation, 121 test. The known-model graph uses all 557 training queries,
including seen-model history at the 8 reserved qids. Only the 549 fitting qids
produce supervised labels. No external-model nodes, attributes, labels or
outcomes enter training/validation/checkpoint selection.

Labels are one-hot winners among eight seen models, determined with the declared
reward. Each of 32 random masks predicts about half the eligible edges while
hiding their attributes; other training edges supply context. This preserves
upstream edge-level masking (not query-level masking). Loss is binary cross
entropy. Backpropagating each mask's loss/32 before one optimizer step is
equivalent to the upstream averaged mask objective, without retaining 32 graphs.
No validation/test outcomes enter visible context. Choose the first highest
validation macro-F1 checkpoint, load it explicitly, then freeze all parameters
and BatchNorm buffers. Test is not evaluated during training. Those last two
items correct the public loop, which calls test each epoch and comments out
checkpoint loading in test().

Deterministic PyTorch mode is required; no silent nondeterministic fallback.
GPU validation is pending on Morel; local tests cover CPU. A deterministic-op
error must be reported rather than bypassed by disabling the integrity setting.

## Four conditions, same checkpoint

- seen_only: eight seen candidates and complete permitted known-model history.
- description_only: add an isolated Qwen node initialized by its description.
- calibrated_8: add Qwen's Direct observations at the prior eight support qids.
- calibrated_80: same eight plus 72 seeded additional train queries selected
  without inspecting outcomes. These may be seen-model training qids, but their
  Qwen outcomes are only exposed after weights freeze, as in auxiliary support.

No target query supplies outgoing observed edges. Targets do not influence one
another. Frozen eval-mode BatchNorm prevents target-batch statistics. Explicit
tests check this. Keep the same known history in all conditions.

## Cost reference and metrics

Require existing raw `cost_proxy_money` from Direct records; missing/invalid
values cause a clear error, with no automatic token-cost substitution. Fit one
global min/max using only the 549 fitting queries and eight seen Direct models.
Apply those endpoints everywhere, including held-out support and test. Do not
clip extrapolated costs or refit the reference on new models.

PRIMARY P/C/R uses this fixed training monetary reference. SECONDARY C_qnorm
and R_qnorm uses the stored original full-pool query-normalized cost, fixed
across all four conditions, solely for historical comparison. The two R scales
must never be conflated with each other or with the earlier joint-prompt tables.
Primary model supervision/checkpoint labels use the primary cost reference.

Report each condition's R/P/C, both normalization references, model/task counts,
per-query selections and candidate scores. Compute incorporation against
seen_only and calibration against description_only. Recompute Direct-only
oracle gain/opportunities for BOTH cost references: the earlier 54 joint-prompt
opportunities are not applicable. Record 8/80 Qwen execution counts and summed
raw calibration monetary proxy. Bootstrap intervals are paired over test queries
and conditional on this one checkpoint; they do not quantify seed variability.

## Run on Morel

```bash
source ~/repos/graph-router-2/.venv/bin/activate
cd ~/repos/ProMoRouter
git pull --ff-only
python -m pip install -r requirements-graphrouter.txt
bash scripts/run_graphrouter_unseen.sh
```

The runner defaults to GPU 0. Optionally pass the data root and cuda:1 as its
two positional arguments. No LLM generations are needed: support is replayed
from logged Direct executions. Output:
`outputs/graphrouter_unseen/qwen2.5-14b/seed1_lam0.9/results.json` and router.pt.
Existing output directories are never overwritten.

## Stop rule

Inspect this one result before modifying the method or launching another fold.
If Direct has no oracle margin, state the limitation instead of calling failed
incorporation. If k8 works, establish the reference before adding prompts. If
only k80 works, attribute the pilot difference to the support budget under this
protocol. If selection harms R, incorporation is technical but not useful. If
neither works, success of GraphRouter in this setting remains unestablished;
do not assume the prompt extension alone caused prior failures.
