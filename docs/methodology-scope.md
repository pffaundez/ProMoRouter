# Methodological scope for paper rewrite

_Last updated: 2026-09-14_

This document records scope details that must remain explicit in the long-paper
rewrite. It separates facts verified against the repaired implementation from
interpretations and extensions that are not evaluated.

## Verified facts

### Offline training and online inference

ProMoRouter is trained **offline** from previously collected interaction logs.
Each logged action has an observed task performance and cost, combined into the
deployment reward

[
R(q,a)=P(q,a)-\lambda C(q,a).
]

At inference time, the router performs **per-query action selection**: it
embeds the query, scores the available candidate actions, and returns the
highest-scoring prompt--model pair.

The current implementation does **not** perform online parameter updates after
each deployed interaction. Thus, “online inference” means online decision
making, not continual online learning.

Realized performance, output length, monetary cost, and reward are outcomes or
training labels. They are not numeric routing-time features supplied to the
scorer.

### Closed candidate pools

The evaluated inference pool is the same pool represented during training:

- 4 prompting strategies: `direct`, `cot`, `decompose`, and `selfcheck`;
- 9 candidate models;
- 36 prompt--model actions per query.

The repaired experiments use the same candidate populations and persisted
query splits for all compared methods.

Therefore, the reported experiments study **closed-pool routing**. They do not
measure zero-shot generalization to unseen models or unseen prompting
strategies.

## Limitations of the current setting

- Performance and cost estimates used as labels are available only after an
  interaction has been executed.
- Results depend on the coverage, quality, and distribution of the logged
  interactions.
- Actions absent or rare in the logs cannot be learned reliably.
- A deployment distribution different from the logged/test distribution may
  cause covariate shift.
- The router does not update itself from post-deployment feedback.
- A fixed candidate pool limits claims about adding new models or prompt
  strategies.

## Hypotheses and extensions (not evaluated)

A new model or prompt strategy could be added in principle if its identifier and
embedding were added to the graph and candidate set, and if the scorer and data
pipeline supported the new node. This would require a dedicated evaluation; it
must not be described as demonstrated generalization.

A future extension could estimate pre-execution cost (for example, with a
length/cost predictor) and expose that estimate to the scorer. This is outside
the current repaired experiments and must remain separate from the present
implementation.

## Wording recommended for the paper

> ProMoRouter is trained offline from logged interactions and performs per-query
> online action selection at inference time, without online parameter updates.
> We evaluate closed-pool routing: the model and prompt candidates available at
> inference are the same candidates represented during training. Extending the
> pool to unseen models or prompting strategies is possible in principle but is
> outside the scope of this evaluation.

## Files and artifacts supporting these statements

- `train_router_edgegnn_qnorm.py`
- `train_router_model_only_direct_qnorm.py`
- `router/data_validation.py`
- `router/splits.py`
- `data/router/query_embeddings.pt`
- `data/router/prompt_embeddings.pt`
- `data/router/model_embeddings.pt`
- `docs/experiment-status.md`
- `docs/decisions.md`
