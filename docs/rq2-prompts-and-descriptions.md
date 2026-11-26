# RQ2 · Prompts and text descriptions

This appendix records the four generation templates and the text inputs used to
construct the task, strategy, and model embeddings for the 6→9 benchmark.
The blocks preserve the English wording and placeholders in the repository.
The description blocks include the prefixes and metadata fields assembled by
the embedding builder.

## How the text enters the experiment

| Text | Role in the experiment |
|---|---|
| Generation template | Formats the LLM request used to obtain an interaction in the offline log. The four strategies define the prompt axis of the joint action space. |
| Query text | Comes from the benchmark example and is represented by a precomputed query embedding. The query embedding builder reads the stored `query_text` field. |
| Prompt strategy description | Provides a prompt embedding. The joint GNN combines it with the query embedding for each query–prompt node. ProMoRouter Lite also uses it when scoring pairs. |
| Task description | Provides a task embedding for both GNN routers. |
| Model description | Provides a model embedding for both GNN routers and Lite. The same description format is used for known and held-out models. |

Training and inference replay precomputed interactions and load embeddings from
`data/router/`. The routers do not call the LLMs during these offline runs.
GraphRouter with fixed Direct prompting uses the Direct subset of the log and
scores models from task, query, and model embeddings. Its graph has no prompt
embedding input. In the zero-support condition, the three held-out LLMs are
available through their description embeddings. With eight support queries,
the observed interactions enter the inference graph as edge attributes
`[R, C, P]`.

## Generation templates

Source: [prompt_templates.yaml](../configs/prompt_templates.yaml).
`{subquery}` holds the instance text. `{context_block}` is optional and is
empty when no context is supplied. The blocks preserve their adjacency.
The scope here is the four strategies in the 6→9 runner.

### `direct`


```text
You are a helpful assistant. Answer the user's request as directly and correctly as possible.

{context_block}{subquery}
```

### `cot`


```text
You are a helpful assistant. Think step by step, then provide the final answer.

{context_block}{subquery}

Final:
```

### `decompose`


```text
You are a helpful assistant. Break the problem into a short plan of 2 to 4 bullet points, then solve it.

{context_block}{subquery}

Plan:
-
Execution:
Final:
```

### `selfcheck`


```text
You are a careful assistant. First provide an answer. Then briefly check for mistakes or missing assumptions. If needed, correct the answer.

Output format:
Answer: ...
Check: ...
Final: ...

{context_block}{subquery}

Answer:
Check:
Final:
```

## Prompt strategy descriptions

Source: [prompt_strategies.yaml](../configs/prompt_strategies.yaml).

### `direct`


```text
Prompting strategy: Direct Answer. Description: A low-cost prompting strategy that asks the model to answer the user query directly and correctly without explicit intermediate reasoning, planning, or verification. It is suitable for simple factual questions, extractive QA, and straightforward instruction-following cases where concise responses are enough.
 Expected cost: low. Expected behavior: concise direct final answer.
```

### `cot`


```text
Prompting strategy: Chain-of-Thought. Description: A reasoning-oriented prompting strategy that encourages the model to think step by step before giving a final answer. It is useful for arithmetic, multi-hop reasoning, and questions that benefit from explicit intermediate inference, but it usually increases output length and inference cost.
 Expected cost: medium. Expected behavior: step-by-step reasoning followed by final answer.
```

### `decompose`


```text
Prompting strategy: Decomposition / Plan-and-Solve. Description: A high-cost prompting strategy that asks the model to break the problem into a short plan or smaller reasoning steps and then solve them before producing the final answer. It is useful for complex multi-step or compositional queries, but can substantially increase token usage.
 Expected cost: high. Expected behavior: short plan, execution, final answer.
```

### `selfcheck`


```text
Prompting strategy: Self-Check. Description: A verification-oriented prompting strategy in which the model first answers, then briefly checks for mistakes, missing assumptions, or inconsistencies, and finally outputs a corrected final answer. It can improve reliability but adds additional token cost.
 Expected cost: medium_high. Expected behavior: answer, check, corrected final answer.
```

## Task descriptions

Source: [task_descriptions.yaml](../configs/task_descriptions.yaml).

### `gsm8k`


```text
Task: GSM8K. Description: Grade-school math word problems requiring multi-step arithmetic reasoning, numerical computation, and extraction of a final numeric answer. The primary metric is exact-match accuracy after answer normalization.
 Primary metric: accuracy. Expected reasoning: arithmetic multi-step reasoning.
```

### `hotpotqa`


```text
Task: HotpotQA. Description: Multi-hop question answering over natural-language evidence, requiring the model to combine multiple supporting facts before producing a concise answer. The primary metric is token-level F1 against reference answers.
 Primary metric: f1. Expected reasoning: multi-hop factual reasoning.
```

### `squad`


```text
Task: SQuAD. Description: Extractive question answering over a provided context passage. The model must identify an answer span or concise answer supported by the passage. The primary metric is token-level F1 or exact match against annotated answers.
 Primary metric: f1. Expected reasoning: context-grounded extractive QA.
```

### `alpaca`


```text
Task: Alpaca-style instruction following. Description: General instruction-following examples in which the model must satisfy a user instruction, optionally conditioned on an additional input field. The task emphasizes helpfulness, relevance, and matching the expected response content.
 Primary metric: f1. Expected reasoning: general instruction following.
```

## LLM descriptions

Source: [model_descriptions.json](../configs/model_descriptions.json).

### `mistral-7b`

Model identifier: `mistralai/Mistral-7B-Instruct-v0.3`.

```text
Language model: Mistral-7B-Instruct. Family: Mistral. Size: 7B. Description: A compact 7B instruction-tuned general-purpose language model. It is relatively low cost and suitable for concise instruction following, factual question answering, and lightweight reasoning, but may be less reliable on difficult multi-step reasoning than larger models.
```

### `qwen2.5-7b`

Model identifier: `Qwen/Qwen2.5-7B-Instruct`.

```text
Language model: Qwen2.5-7B-Instruct. Family: Qwen2.5. Size: 7B. Description: A compact 7B instruction-tuned general-purpose model from the Qwen2.5 family. It is cost-efficient and often strong for mathematics, multilingual text, instruction following, and structured reasoning relative to its size.
```

### `llama3.1-8b`

Model identifier: `meta-llama/Llama-3.1-8B-Instruct`.

```text
Language model: Llama-3.1-8B-Instruct. Family: Llama 3.1. Size: 8B. Description: An 8B instruction-tuned general-purpose model from the Llama 3.1 family. It provides a low-to-moderate cost option for general QA, summarization-style responses, and instruction following, with limited capacity compared with larger models.
```

### `qwen2.5-14b`

Model identifier: `Qwen/Qwen2.5-14B-Instruct`.

```text
Language model: Qwen2.5-14B-Instruct. Family: Qwen2.5. Size: 14B. Description: A mid-sized 14B instruction-tuned model from the Qwen2.5 family. It balances cost and quality and is suitable for mathematical reasoning, question answering, and general instruction following when 7B models may be insufficient.
```

### `yi-34b`

Model identifier: `01-ai/Yi-34B-Chat`.

```text
Language model: Yi-34B-Chat. Family: Yi. Size: 34B. Description: A 34B chat-oriented general-purpose model. It is more capable but more expensive than compact 7B and 14B models, and is intended for harder instruction-following, reasoning, and question-answering cases where stronger language understanding is useful.
```

### `codellama-34b`

Model identifier: `codellama/CodeLlama-34b-Instruct-hf`.

```text
Language model: CodeLlama-34B-Instruct. Family: CodeLlama. Size: 34B. Description: A 34B instruction-tuned model specialized for code generation and code reasoning. It can handle programming-oriented prompts well, but may be less optimal than general-purpose chat models for ordinary factual QA, math word problems, or instruction-following tasks unrelated to code.
```

### `mixtral-8x7b`

Model identifier: `mistralai/Mixtral-8x7B-Instruct-v0.1`.

```text
Language model: Mixtral-8x7B-Instruct. Family: Mixtral. Size: MoE-8x7B. Description: A sparse mixture-of-experts instruction-tuned model with 8 experts of 7B parameters. It is a strong general-purpose option with higher capacity than dense small models and can perform well on reasoning, QA, and instruction-following while maintaining efficient sparse activation.
```

### `llama3.1-70b`

Model identifier: `meta-llama/Llama-3.1-70B-Instruct`.

```text
Language model: Llama-3.1-70B-Instruct. Family: Llama 3.1. Size: 70B. Description: A large 70B instruction-tuned general-purpose model. It is expected to be stronger on complex reasoning, multi-hop QA, and instruction following than smaller Llama models, but has substantially higher inference cost.
```

### `qwen2.5-72b`

Model identifier: `Qwen/Qwen2.5-72B-Instruct`.

```text
Language model: Qwen2.5-72B-Instruct. Family: Qwen2.5. Size: 72B. Description: A large 72B instruction-tuned general-purpose model from the Qwen2.5 family. It is a high-quality but high-cost option for difficult reasoning, mathematics, QA, and instruction-following tasks where smaller models may fail.
```

## Reproducibility

- [6→9 runner](../run_unseen_6x3.py) and [experiment protocol](unseen-6x3-comparison.md)
- [Description embedding builder](../scripts/build_node_description_embeddings.py).
  Its default is `sentence-transformers/all-MiniLM-L6-v2` with normalized embeddings.
- [Query embedding builder](../analysis/build_query_embeddings.py).
  Its configured encoder is `intfloat/e5-large-v2` with normalized embeddings.

The text above is verified against the repository configuration and builder.
Before treating this appendix as a record of the exact Morel run, match the
description blocks against the `texts` fields in the actual embedding packs
and verify the encoder metadata. The run JSON records action IDs and scores.
Exact rendered requests, dataset-specific context formatting, and backend chat
wrappers require the original generation logs or generator.
