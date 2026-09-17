# Inductive unseen-candidate evaluation protocol

_Last updated: 2026-09-16_

This protocol extends the repaired closed-pool evaluation to candidates that have no training interactions.

## Candidate visibility

The target configuration contains 9 seen models and 4 seen prompting strategies for training, plus three external unseen models and two external unseen prompting strategies. The confirmed candidates are `HuggingFaceTB/SmolLM2-1.7B-Instruct`, `allenai/OLMo-2-1124-7B-Instruct`, `microsoft/phi-4`, `step_back`, and `self_consistency`; record them in `configs/inductive_candidates.yaml`.

## Evaluation conditions

### Closed-pool control

Train and evaluate with the existing 9-model/4-prompt pool. This is the reference condition for the repaired results.

### Unseen-model condition

Remove all interactions involving the external models from training and validation. Add their nodes at inference using description-derived embeddings. Evaluate them together with the nine seen models and the known prompts.

### Unseen-prompt condition

Remove all interactions involving the external prompting strategies from training and validation. Add their nodes at inference using description-derived embeddings. Evaluate them together with the four seen prompts and seen models.

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


## Prompt execution contracts

The unseen strategies use the following fixed contracts:

- `step_back`: issue a static instruction to identify the general principle
  relevant to the query before applying it; no external demonstrations are
  required.
- `self_consistency`: issue exactly three independent samples with the same
  strategy template and aggregate final answers by majority over normalized
  answers. Decoding parameters must be reported.

The templates are defined in `configs/prompt_templates.yaml`. Demonstration
content, sample count, decoding parameters, and their input/output token costs
must be recorded before evaluating the router.


## Concrete execution configuration

The reproducible settings are stored in
`configs/inductive_prompt_execution.yaml`:

- `step_back` uses only its static instruction and query text; no calibration
  examples are required.
- `self_consistency` uses three samples with temperature 0.7, top-p 0.95,
  and a 256-token generation cap. Final answers are normalized before majority
  aggregation, with a deterministic lexicographic tie-break.


## Backend and dataset boundary

P0 remains the primary closed-pool result and is not modified. The inductive
evaluation uses a separate test manifest and regenerates every candidate action
(12 models x 6 prompts) with one sequential Transformers backend. Seen and
unseen actions are therefore comparable within the inductive table, while
their absolute values are not merged with the vLLM-generated P0 table.
