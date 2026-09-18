# Edge-GNN v2 Stage A implementation contract

_Status: design frozen; not implemented or executed._

## Question isolated by Stage A

Does heterogeneous message passing improve joint prompt--model routing when
the query population, 36-action space, embeddings, scorer, optimization,
labels, splits, and evaluation are held fixed?

Stage A is a closed-pool experiment. It does not establish inductive
generalization and it does not modify frozen P0 artifacts.

## Controlled arms

| Arm | Message passing | Scorer | Role |
|---|---|---|---|
| `edgegnn_v2_full_mp` | Two heterogeneous layers | Shared edge-aware scorer | Treatment |
| `edgegnn_v2_no_mp` | Disabled; projected node states only | Same scorer and parameter dimensions | Primary matched control |
| `flat_mlp_shared_objective` | None | Existing flat concatenation architecture | External architecture control |

The first two arms are the causal comparison. The Flat-MLP arm is retrained
under the Stage A objective and deterministic execution contract in a new
output directory. Existing P0 Flat-MLP results remain immutable and may be
shown only as a separately labelled historical reference.

## Transductive-safe graph unit

Each query is represented by an independent ego graph. No graph contains more
than one query, and no message can pass between train, validation, or test
queries.

Nodes:

- one `query` node;
- one node for the query's `task`;
- four independent `prompt` nodes;
- nine independent `model` nodes.

Bidirectional relations:

- `query <-> task`;
- `query <-> prompt` for all four prompts;
- `query <-> model` for all nine models.

The graph therefore exposes the same complete candidate identities for every
query without using action outcomes to select topology. Stage A has no
reward-selected top-k edges and no prompt--model lattice relation. Graph
density variants belong to a later gated experiment.

## Shared edge-aware scorer

For the two matched arms, each candidate action is scored as

`s(q,p,m) = f_theta(h_q, h_p, h_m, e_pm)`.

`edgegnn_v2_full_mp` obtains `h_q`, `h_p`, and `h_m` after two message-passing
layers. `edgegnn_v2_no_mp` uses the corresponding projected initial states.
The scorer, hidden sizes, dropout, and parameterization are otherwise shared.

`e_pm` must be compositional and candidate-description based, for example a
shared projection of `[x_p, x_m, x_p * x_m]`. A learned lookup table indexed by
the 36 closed-pool pair IDs is prohibited because it cannot represent an
unseen prompt--model pair in the later inductive evaluation.

## Information contract

Allowed routing inputs:

- query embedding;
- task identity/embedding;
- prompt identity and description embedding;
- model identity and description embedding;
- compositional `e_pm` derived only from prompt/model representations.

Prohibited routing inputs:

- realized reward or performance;
- realized normalized or monetary cost;
- realized input/output token counts;
- topology selected using any validation/test outcome;
- test-query outcomes or cross-query message passing.

Performance, cost, and reward remain offline labels/evaluation outcomes. Stage
A introduces no pre-routing cost predictor.

## Frozen experimental controls

- Input: repaired 797-query complete-only qnorm dataset.
- Actions: four prompts x nine models = 36 per query.
- Seeds: 1--5.
- Splits: `qnorm_complete_seed<seed>.json`, shared across all arms.
- Lambdas: 0.1, 0.5, and 0.9.
- Reward: `R = P - lambda * C`.
- Hidden dimension: 256.
- Message-passing layers: two for `edgegnn_v2_full_mp`.
- Dropout: 0.10.
- Optimizer: AdamW, learning rate `5e-4`, weight decay `1e-4`.
- Maximum epochs: 80; patience: 18.
- Early stopping: validation reward.
- Deterministic execution: required for all arms.
- Stage A objective: current Edge-GNN objective,
  `MSE + 0.10*KL + 0.05*CE - 0.005*entropy`, temperature 0.10.

The Flat-MLP Stage A implementation must support the same objective and
deterministic mode; otherwise it cannot be included as a protocol-matched arm.
Its historical P0 numbers are not substituted for a missing Stage A run.

## Required implementation outputs

Each arm/seed JSON must contain three lambda results and record:

- arm ID, seed, lambda key, and lambda;
- dataset and split-manifest paths plus dataset fingerprint;
- topology ID, node/edge counts, message-passing layer count;
- objective coefficients and early-stopping criterion;
- deterministic flag and relevant runtime versions;
- 121 test queries;
- P, C, R, prompt counts, model counts, and task counts;
- checkpoint path and a serialized configuration snapshot.

The aggregator must require 15 source JSON files, 45 seed-level results, and
nine arm/lambda aggregates. It must verify unique arm/seed/lambda keys, seeds
1--5, 121 queries, shared manifests, exact reward algebra, count totals, source
hashes, and mean/sample-standard-deviation recomputation.

## Execution gates

1. Static and synthetic tests: graph membership, exact relation counts,
   split isolation, no forbidden features, all 36 scores, and no-MP bypass.
2. CPU smoke: one small synthetic batch for all three arms; finite loss and
   gradients; output schema validation.
3. Determinism gate: two independent seed-1 repetitions per arm must match
   after excluding only output-path fields.
4. Seed-1 integrity gate: 121 queries, reward algebra, non-empty selection
   counts, and no total fixed-action collapse. This is diagnostic, not a
   criterion for preferring an arm.
5. Five-seed sweep and aggregation only after all earlier gates pass.

## Interpretation gate

Stage A may support an architectural message-passing claim only if the matched
`edgegnn_v2_full_mp` arm improves over `edgegnn_v2_no_mp` consistently enough
to justify the claim. Flat-MLP provides a broader external control but is not a
substitute for the matched comparison. No significance, v2 advantage, or
inductive-generalization claim exists before execution and analysis.

Only after Stage A is complete may a new recorded decision authorize Stage B
(`L_reg + beta L_rank`), graph-density variants, a factorized router, or the
separate inductive evaluation.
