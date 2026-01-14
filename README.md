# GraphRouter**

GraphRouter** is an extension of GraphRouter that performs joint routing over
(1) decomposed sub-queries,
(2) reasoning prompts (e.g., CoT, ToT, Self-Ask, Reflexion),
and (3) heterogeneous LLMs.

Unlike prior routers that assign a single model to an entire query,
GraphRouter** decomposes complex queries into atomic sub-queries and routes
each sub-query to the most suitable (prompting strategy, LLM) pair using an inductive
heterogeneous graph neural network.

## Key Idea

Query → {Sub-Queries} → (Prompt, LLM)

Each sub-query is treated as an indivisible reasoning unit, and the router
learns:
- which prompting strategy best fits each sub-query,
- which LLM best executes that strategy,
- how to compose the resulting answers into a final solution.

## Contributions (IN THEORY)

- Hierarchical routing over sub-queries.
- Explicit modeling of reasoning strategies as graph nodes.
- Joint prompt–model selection via inductive message passing.
- Generalization to unseen prompts and unseen LLMs.

## Supported Prompting Strategies TBD

- Direct Answer
- Chain-of-Thought (CoT)
- Tree-of-Thought (ToT)
- Self-Ask
- Reflexion
- Program-of-Thought
- ...

## Datasets TBD

- GSM8K (multi-step arithmetic reasoning)
- HotpotQA (multi-hop QA)
- BIG-Bench Hard (compositional reasoning)

## Baselines TBD

- GraphRouter
- AgentRouter
- Best single LLM
- Best single prompt
- among others
