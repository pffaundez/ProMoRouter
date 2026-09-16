# Inductive unseen-candidate evaluation protocol

_Last updated: 2026-09-16_

This protocol extends the repaired closed-pool evaluation to candidates that have no training interactions.

## Candidate visibility

The target configuration contains 9 seen models and 4 seen prompting strategies for training, 3 external unseen models with distinct parameter scales, and 1 or 2 external unseen prompting strategies. The concrete external candidates remain to be selected and recorded in configs/inductive_candidates.yaml.

## Evaluation conditions

### Closed-pool control

Train and evaluate with the existing 9-model/4-prompt pool. This is the reference condition for the repaired results.

### Unseen-model condition

Remove all interactions involving the external models from training and validation. Add their nodes at inference using description-derived embeddings. Evaluate them together with the seen models and known prompts.

### Unseen-prompt condition

Remove all interactions involving the external prompting strategies from training and validation. Add their nodes at inference using description-derived embeddings. Evaluate them together with seen prompts and models.

### Joint unseen condition

Evaluate actions combining unseen models and unseen prompts. These pairs must not occur in the training population.

### Masked-pair condition

Mask selected prompt-model interactions from training while retaining the corresponding nodes. Use these pairs to test compositional generalization separately from node-level novelty.

## Zero-shot and few-shot variants

Report two variants when data permits:

- Zero-shot: unseen nodes contribute only static descriptions, embeddings, and permitted metadata.
- Few-shot: a calibration set provides a fixed number of interactions for unseen candidates; calibration queries are disjoint from test queries and do not trigger parameter updates.

GraphRouter's few-shot protocol is a reference point, not a requirement for the zero-shot variant.

## Required artifacts

- static descriptions for every candidate;
- description-embedding metadata and encoder identifier;
- train, calibration, and test manifests;
- interaction outcomes for test actions;
- a validator proving that unseen candidates have no training interactions.

## Metrics

Report, for each condition and lambda, P, C, and R; selection frequency of seen and unseen candidates; regret relative to the action oracle; calibration-versus-test performance; and results for unseen-model, unseen-prompt, unseen-pair, and closed-pool subsets.

## Non-claims

This protocol does not establish unrestricted generalization to arbitrary models or prompts. Results apply only to the selected candidates, descriptions, calibration budget, and test distribution.
