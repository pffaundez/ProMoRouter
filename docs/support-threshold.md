# One bounded margin-threshold experiment

Date: 2026-11-23. Decision D-032. This is an inference policy on the **uniform
SupportGraphRouter**, not a modification of the official GraphRouter encoder.

## Question and fixed design

Can a single internally learned threshold improve useful incorporation of a
held-out model without updating router weights during incorporation?

Use the existing uniform architecture and defaults: hidden 128, AdamW
lr 0.0005 / weight decay 0.0001, batch 32, MSE + 0.05 CE, clip 5,
80 epochs maximum, patience 18, seed 1, lambda 0.9, eight support queries,
four prompts. Cost processing remains the uniform pilot's contract: per-query
seen-only minmax for training/support; stored original full-pool qnorm for
all evaluation policies. Do not compare its reward numerically with the official
control's primary monetary-normalized reward. Calibration requires 8 x 4 = 32
logged outcomes per new model; deployment monetary savings are not evaluated.

Freeze these model roles before execution:

| Role | Models |
|---|---|
| Internal model holdouts | llama3.1-8b, codellama-34b |
| External model holdouts | yi-34b, mixtral-8x7b |
| Shared outer known pool | All seven models excluding both external holdouts |

Both external models are excluded from **every** inner fit, checkpoint choice,
and threshold choice/check. Each inner fit additionally excludes its own
internal holdout: six seen models. Exclusion applies to target labels, episodic
support, validation checkpoint labels and normalization. Excluding an already
trained model only at inference would not test unseen-model incorporation.

There are two inner training trajectories and, only if the gate passes, one
shared outer trajectory. The two external candidates are added separately to
the same seven-model base, never used as context for one another. This is a
controlled seven-to-eight-model experiment, not the earlier eight-to-nine
Qwen development experiment. Existing Qwen checkpoints cannot be reused because
they were trained on the new external holdouts.

## Query partitions and threshold

Reuse the seed-1 query split and its eight reserved training support qids:
549 fitting, eight support, 119 validation, 121 test. Deterministically split
validation qids into 59 tuning and 60 checking qids using seed+2718.

For each inner holdout:

1. Train on the six-model pool. Select the checkpoint on seen-only reward on
   the 59 tuning qids. Freeze all weights.
2. Reveal that holdout's outcomes on the eight support qids. Score the tuning
   and checking queries independently. No target outcomes enter the scorer.
3. For each query record `margin = max_new_prompt_score - max_seen_pair_score`.
   The proposed new prompt is the **predicted** best prompt, not an oracle
   selected from target rewards. Label its actual reward difference against
   the predicted seen pair.

Pool the two internal tuning folds and choose the threshold maximizing mean
reward gain. Enumerate all distinct strict-threshold policies, including
negative thresholds and reject-all; ties prefer fewer incorporations. A
negative threshold can compensate a systematic score deficit. Threshold zero
is the original argmax decision with ties resolved in favor of seen models.
`null` encodes reject-all. There is no threshold search on checking or test data.

Check the chosen threshold once on the separate 60 checking qids per fold.
Proceed only when **all** predeclared criteria hold:

- The policy is not reject-all.
- Margin AUC exceeds 0.5 both pooled and within each internal model fold.
- Pooled reward gain is positive and neither model fold has negative gain.

AUC distinguishes truly beneficial from harmful predicted new pairs, ignoring
true reward ties. Undefined AUC fails the gate. The per-fold checks prevent a
pooled AUC from appearing useful solely because score scales differ by fold.
The gate is an operational filter, not a significance claim. Report a 2,000
resample query-cluster bootstrap gain interval without using it to tune the
threshold. Queries shared across internal folds are resampled together.

## Stop or confirm, then close

Always write `frozen_threshold.json` and its SHA-256 **before** any external
calibration or evaluation. If the internal gate fails, write
`status=stopped_internal_gate`, do not train the outer router or evaluate the
external models, and close this margin intervention as unsupported by its
internal test. Do not change thresholds or gate criteria after viewing results.

If the gate passes, train the shared seven-model base once using seen-only
validation checkpoint selection and the same fixed hyperparameters. Freeze its
weights, add each external model separately with eight support queries, and
compare reject-all, threshold-zero argmax, and the frozen threshold on test.
Report P/C/R, paired gain, beneficial/harmful selections, pair concentration,
per-task profiles and conditional bootstrap intervals. Assert unchanged model
state, unchanged seen scores and unchanged threshold artifact.

External confirmation succeeds descriptively only if **each** external model
has positive gated gain over reject-all and improves over threshold-zero
argmax. Report uncertainty regardless; one seed and conditional bootstraps do
not establish broad generalization. If either fails, close this intervention
without an automatic architecture or hyperparameter sweep.

These holdout roles have not been evaluated in this new protocol, but the
benchmark and test qids were used in earlier development. This is model-holdout
confirmation on a reused benchmark, not a pristine independent dataset. Absolute
score-threshold transfer across model pools is itself the hypothesis being tested.

## Run once on Morel

```bash
source ~/repos/graph-router-2/.venv/bin/activate
cd ~/repos/ProMoRouter
git pull --ff-only
bash scripts/run_support_threshold.sh
```

The runner defaults to GPU 0; pass the data root and `cuda:1` as positional
arguments to use GPU 1. There is no benefit to two concurrent GPU jobs here:
external execution depends on the completed internal gate.

Return `outputs/support_threshold/seed1_lam0.9/results.json`, including when it
reports an internal stop. All checkpoints and the frozen decision are saved in
that new directory. An existing directory is refused rather than overwritten.

## Engineering checks

21 support/audit/threshold tests pass on CPU, including negative thresholds,
reject-all, predicted-prompt labels, disjoint validation partitions, nested
model exclusion, and synthetic end-to-end execution of both stop and continue
paths. The orchestration tests explicitly force their gate outcome to exercise
both paths; they are not positive research results. Real-data execution is
pending on Morel.
