# ProMoRouter

This repository contains the implementation and experimental artifacts for **ProMoRouter**, a heterogeneous graph-based router for joint prompt--model selection. ProMoRouter treats prompting strategy as a first-class routing decision: for each input query, it selects both a prompting strategy and an LLM from a candidate pool under a deployment objective that trades off performance and cost.

The code is provided for anonymous review. Author-identifying metadata, private paths, and non-essential generated artifacts have been removed.

## Overview

ProMoRouter formulates routing as edge-aware decision making over a heterogeneous interaction graph with task, query, prompt, and model nodes. During training, observed query--prompt--model interactions define action edges. At inference time, the router scores candidate prompt--model actions and selects the highest-scoring pair for each query.

The main experimental setting evaluates three deployment regimes using a query-normalized reward:

- **Performance-first**: low cost penalty, `lambda = 0.1`
- **Balanced**: moderate cost penalty, `lambda = 0.5`
- **Cost-first**: high cost penalty, `lambda = 0.9`

The repository also includes static and adaptive baselines, including fixed-model, fixed-pair, model-only routing, and oracle upper bounds.

## Repository Structure

```text
.
├── data/
│   ├── interaction_logs/
│   │   └── grpp_il_v1/
│   │       └── router_bipartite_qnorm.jsonl
│   └── router/
│       ├── query_embeddings.pt
│       ├── task_embeddings.pt
│       ├── prompt_embeddings.pt
│       ├── model_embeddings.pt
│       └── node_description_embedding_metadata.json
├── scripts/
│   ├── build_node_description_embeddings.py
│   └── summarize_results.py
├── outputs/
├── train_router_edgegnn_qnorm.py
├── train_router_model_only_direct_qnorm.py
├── train_router_fixed_baselines_qnorm.py
└── README.md
```

Some generated files may be absent from the anonymized archive and can be reproduced with the commands below.

## Environment

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install the required packages:

```bash
pip install torch numpy pandas scikit-learn tqdm sentence-transformers
```

If your local setup uses CUDA-specific PyTorch wheels, install PyTorch following the instructions for your CUDA version before installing the remaining dependencies.

## Data Preparation

The main router data is expected at:

```text
data/interaction_logs/grpp_il_v1/router_bipartite_qnorm.jsonl
```

The expected embedding files are:

```text
data/router/query_embeddings.pt
data/router/task_embeddings.pt
data/router/prompt_embeddings.pt
data/router/model_embeddings.pt
```

If node-description embeddings are missing, regenerate them with:

```bash
python scripts/build_node_description_embeddings.py
```

## Running the Main Method

Run ProMoRouter Edge-GNN for one seed:

```bash
python train_router_edgegnn_qnorm.py --seed 1
```

Run the five seeds used in the paper:

```bash
python train_router_edgegnn_qnorm.py --seed 1
python train_router_edgegnn_qnorm.py --seed 2
python train_router_edgegnn_qnorm.py --seed 3
python train_router_edgegnn_qnorm.py --seed 4
python train_router_edgegnn_qnorm.py --seed 5
```

To select a GPU explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 python train_router_edgegnn_qnorm.py --seed 1
```

The script trains and evaluates the router for all three reward regimes:

```text
reward_qnorm_lam_01
reward_qnorm_lam_05
reward_qnorm_lam_09
```

Outputs are written to:

```text
outputs/router_edgegnn_qnorm/
```

## Running Baselines

### Static baselines

```bash
python train_router_fixed_baselines_qnorm.py
```

This evaluates:

- Largest LLM
- Smallest LLM
- Best Fixed Model
- Best Fixed Prompt--Model Pair
- Oracle Model-Only
- Oracle Prompt--Model

### Model-only routing baseline

```bash
python train_router_model_only_direct_qnorm.py --seed 1
python train_router_model_only_direct_qnorm.py --seed 2
python train_router_model_only_direct_qnorm.py --seed 3
python train_router_model_only_direct_qnorm.py --seed 4
python train_router_model_only_direct_qnorm.py --seed 5
```

This baseline routes only over models and does not explicitly select prompting strategies.

## Optional Ablations

If included in the archive, the following scripts reproduce graph-structure and neighborhood-size ablations:

```bash
python train_router_edgegnn_qnorm.py --seed 1
python train_router_edgegnn_qnorm_lattice.py --seed 1
python train_router_edgegnn_qnorm_topk5.py --seed 1
python train_router_edgegnn_qnorm_topk5_lattice.py --seed 1
```

Run each command for seeds 1--5 to reproduce the multi-seed summaries.

## Summarizing Results

After running all seeds, summarize results with:

```bash
python scripts/summarize_results.py
```

If the summarization script is not included, the result files can be inspected directly. Each run produces JSON files of the form:

```text
outputs/<experiment_name>/<experiment_name>_results_seed<N>.json
```

Each JSON file reports performance `P`, normalized cost `C`, reward `R`, selected prompt counts, and selected model counts for each reward regime.

## Main Metrics

The reported metrics are:

- `P`: normalized task performance
- `C`: normalized query-level inference cost
- `R`: deployment reward

The reward follows the query-normalized cost setting used in the experiments:

```text
R = P - lambda * C
```

where `lambda` controls the strength of the cost penalty.

## Reproducibility Notes

The experiments are stochastic and should be reported over multiple seeds. For the main table, use the mean over seeds. Standard deviations can be reported in an appendix or ablation table when available.

Recommended reporting format:

```text
mean ± standard deviation
```

For the compact main paper table, use three decimal places.

## Citation

This repository is anonymized for review. Citation information will be added after the review process.

## License

License information will be added in the public release.