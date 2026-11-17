# ProMoRouter v2 calibration pilot

The existing Flat MLP scores each query–prompt–model action. This pilot adds
four features from support interactions: mean quality, log-transformed mean
total tokens, log-transformed support query count, and profile availability.

Both arms use the same support query IDs and score the same joint actions.
The global arm pools outcomes across prompts and uses the same profile for
every prompt of that model. The per-prompt arm aggregates only observations
for the candidate prompt. The global count is expressed in support queries
rather than calls so that both arms encode eight observations per prompt when
coverage is complete. The profile uses raw token counts, not query-normalized
costs.

Run on the experiment host with the complete original 9-model × 4-prompt log
and the original query, prompt, and model embeddings:

```bash
python train_router_behavior_profiles.py \
  --held-out-model mistral-7b \
  --calibration-queries 8 \
  --seed 1
```

The runner reserves the support queries from the training query split. The
held-out model is excluded from fitting and checkpoint selection. Fit and
validation rewards are recomputed using per-query min-max normalization of
the seen candidates' monetary cost proxies. Test uses the original full-pool
cost labels as outcomes after selection. Both arms use these same conventions.
The held-out model enters test scoring using only its description embedding
and support profile, with frozen MLP weights. Eight support queries correspond
to 32 logged calls for that model. The seen models are profiled on the same
queries.

Outputs are grouped by held-out model, support size, and seed under
`outputs/v2_behavior_profiles/`. They include six checkpoints (two arms
across three reward weights), support query IDs, quality, cost, reward, and
selection counts. This pilot compares profile granularity. The model-only
comparison, missing-profile evaluation, and full model-holdout sweep are
subsequent experiments.
