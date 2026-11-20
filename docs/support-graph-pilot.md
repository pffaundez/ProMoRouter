# Frozen support-graph pilot

One implementation, one external fold, one seed and lambda initially. No
architecture sweep. This implements D-028 using prompt-conditioned parallel
query--model edges. Task embeddings are concatenated with query embeddings;
prompt embeddings condition the messages. Two rounds update model states
separately for each prompt using mean support aggregation. These states are
computed compositionally, not from trainable pair-ID lookups. This is an
adaptation inspired by GraphRouter, not an exact reproduction of its code.

## Information and training contract

- Reserve eight train qids using the same seeded sampling as the behavior-profile
  pilot. They never serve as supervised targets. Validation and test are disjoint.
- Exclude the external LLM from every training and validation candidate tensor.
- Train on episodes: sample support from training qids outside the target batch;
  preserve observed quality, cost and prompt for every seen-model interaction.
- Fit with MSE(reward) + 0.05 CE(best action); one fixed objective, no search.
- Select the epoch on seen-only validation reward. Freeze parameters afterward.
- Support encoding depends only on support, never on targets. Targets are scored
  independently. No target outcomes or target-to-target message paths exist.
- Calibration uses eight queries x four prompts = 32 held-out executions.
  Historical evidence for seen models uses the same support queries.

## Three conditions from the same checkpoint

1. seen_only: eight seen models, observed seen support.
2. description_only: add the held-out embedding, mask all its support outcomes.
3. calibrated: reveal the held-out observations, without updating any parameter.

Seen scores are invariant to enabling/calibrating the new model in this minimal
design. Thus switches are attributable to its scores rather than changes to the
seen policy. The new model's conditioned states are the only added evidence.

## Costs and outputs

Training labels and support costs use query min-max over seen models, removing
the external model's influence on fitting normalization. Seen support has the
same normalization in all three conditions. Held-out support costs use those
same seen endpoints (may exceed [0,1]); a zero seen range maps costs to zero.
Evaluation P/C/R uses stored original full-pool query-normalized C, fixed across
all conditions; these target costs are outcomes, never routing inputs. Fit and
evaluation cost scales differ, as in the preceding held-out profile protocol;
this limitation must be disclosed when comparing or interpreting R.

The JSON stores histories, chosen epoch, partitions, dataset SHA256, support
outcomes, P/C/R, per-query selections, model/prompt/pair counts, pair entropy,
largest pair share, incorporation gain, observed oracle opportunities and
beneficial/harmful/tied selections. Collapse is a diagnostic, not a failure gate.
No monetary calibration cost or net deployment savings are claimed.

## First real-data command (Morel)

```bash
git pull --ff-only
v1_data="$HOME/repos/graph-router-2/data"
python train_router_support_graph.py \
  --data-path "$v1_data/interaction_logs/grpp_il_v1/router_bipartite_qnorm_complete.jsonl" \
  --query-emb-path "$v1_data/router/query_embeddings.pt" \
  --task-emb-path "$v1_data/router/task_embeddings.pt" \
  --prompt-emb-path "$v1_data/router/prompt_embeddings.pt" \
  --model-emb-path "$v1_data/router/model_embeddings.pt" \
  --held-out-model qwen2.5-14b --calibration-queries 8 \
  --reward-weight 0.9 --seed 1 --device cpu
```

Qwen seed 1/lambda .9 is a development case already inspected in prior work,
not independent confirmation. Output goes to
`outputs/support_graph/qwen2.5-14b/k8_seed1_lam0.9/` and existing results are
not overwritten. Read this pilot before launching more folds or changing the
method. Conclude on calibration gain, incorporation gain, P/C tradeoff and
selection concentration. The next action is interpreting that result, not
opening another architecture branch.


## Audit the existing checkpoint (no training)

After the first real pilot, run:

```bash
python audit_support_graph.py \
  outputs/support_graph/qwen2.5-14b/k8_seed1_lam0.9 \
  --device cuda:0
```

The audit verifies the dataset fingerprint, disjoint partitions, support
outcomes, checkpoint configuration, and reproduction of the original choices.
It writes score_audit.json without modifying the checkpoint or results.json.
For Qwen it reports changes in each prompt's scores, best-pair ranks among 36
candidates, margins relative to the best seen score, and summaries restricted
to observed oracle opportunities. Ranks use strictly-greater scores (ties share
a rank). A positive margin means Qwen strictly beats the best seen candidate.
A score change alone is not evidence of useful incorporation.

A best fixed seen pair per task is selected from validation using seen-only
normalized costs, matching checkpoint selection. It is then evaluated on test
with the same fixed full-pool evaluation costs as the router. Missing validation
tasks use the validation-global best pair and are explicitly flagged. No test
outcome is used to select or tune this policy.
