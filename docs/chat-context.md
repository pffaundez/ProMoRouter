# ProMoRouter — Chat Context

Lee este archivo primero cuando retomes el proyecto en un chat nuevo. Después
lee, en este orden:

1. `docs/experiment-status.md`
2. `docs/decisions.md`
3. `docs/next-steps.md`
4. `docs/cost-information-contract.md`
5. `docs/inductive-evaluation-protocol.md`

## Estado operativo

- Repositorio: `pffaundez/ProMoRouter`
- Rama de trabajo: `fix/p0-routing-integrity`
- P0 está congelado y no debe modificarse.
- Los artefactos locales de datos y embeddings viven principalmente en
  `~/repos/graph-router-2`, no necesariamente en GitHub.
- El entorno disponible tiene PyTorch con CUDA y Transformers; no tiene vLLM ni
  Ollama activo.
- Los embeddings inductivos ya fueron generados y validados:
  modelos unseen `(3,384)`, prompts unseen `(2,384)`.

## Pool inductivo confirmado

Modelos unseen:

- `smollm2-1.7b`
- `olmo2-7b`
- `phi-4`

Prompts unseen:

- `step_back`
- `self_consistency` con 3 muestras y mayoría sobre respuestas normalizadas.

Los tres modelos pasaron `AutoConfig` y generación mínima con Transformers.

## Contrato experimental

P0 conserva los resultados cerrados generados con vLLM. La evaluación inductiva
es un protocolo separado: nuevo test, 12 modelos x 6 prompts, todas las 72
acciones regeneradas con un único backend Transformers secuencial. Sus valores
no se mezclan numéricamente con P0.

Los resultados de acciones incluyen tokens de entrada/salida, latencia,
backend, modelo, prompt y performance. El coste realizado se calcula después
de la generación; nunca se usa como feature de routing.

Las gold answers se obtienen de Hugging Face `datasets`, usando la
configuración existente en `configs/rq2_dataset_builder_smoke.yaml`.
Alpaca usa token-level F1; esta elección fue verificada mediante un join de
7.200 acciones raw/qnorm sin discrepancias.

## Comandos base

Activar entorno:

```bash
source ~/repos/graph-router-2/.venv/bin/activate
```

Validar candidatos y embeddings:

```bash
python analysis/validate_inductive_candidates.py
```

Smoke test del generador:

```bash
python experiments/generate_inductive_transformers.py \
  --source-qnorm ~/repos/graph-router-2/data/interaction_logs/grpp_il_v1/router_bipartite_qnorm_complete.jsonl \
  --split-manifest ~/repos/graph-router-2/data/router/splits/qnorm_complete_seed1.json \
  --output outputs/inductive_smoke.jsonl \
  --max-queries 1 \
  --dry-run
```

## Regla de continuidad

No abrir otra variante metodológica sin actualizar `docs/decisions.md`.
Después de cada cambio significativo actualiza también
`docs/experiment-status.md` y `docs/next-steps.md`. Mantén una única
tarea en la sección `Now` de `docs/next-steps.md`.


## Estado al cierre de este chat (2026-09-18)

La generación inductiva completa terminó en el host del experimento con 8,712 filas (121 test qids × 12 modelos × 6 prompts), usando Transformers para todos los candidatos. La auditoría encontró dos respuestas vacías causadas por truncación al límite de 256 tokens. Se regeneraron correctamente de forma dirigida con un límite de 512 tokens:

- `gsm8k-train-000072`, `llama3.1-8b`, `cot`: response `54`, performance 1.0.
- `alpaca-train-000102`, `llama3.1-70b`, `step_back`: response POS-tagged, performance 0.5.

Los archivos de reparación fueron recibidos, pero todavía no se han fusionado en `outputs/inductive_full.jsonl` ni se ha ejecutado la validación final. La única tarea activa es T-023 en `docs/next-steps.md`.

## Prompt listo para el siguiente chat

> Estamos continuando el proyecto ProMoRouter en la rama `fix/p0-routing-integrity`. Lee primero `docs/chat-context.md`, luego `docs/experiment-status.md`, `docs/decisions.md` y `docs/next-steps.md`. No reconstruyas el contexto desde la conversación. La generación inductiva completa ya produjo 8,712 filas en `outputs/inductive_full.jsonl`; dos filas fueron truncadas y ya fueron regeneradas en `repair_gsm8k_000072.jsonl` y `repair_alpaca_000102.jsonl`. Tu única tarea inicial es ejecutar T-023: fusionar esas dos filas en una copia corregida del JSONL y validar exactamente 8,712 claves únicas `(qid, model, prompt)`, 121 qids, 12 modelos, 6 prompts, cero respuestas vacías y presencia de las dos reparaciones con performance 1.0 y 0.5. No agregues nuevos experimentos ni cambies el protocolo hasta que esta validación pase. Después actualiza los tres documentos de continuidad y propone el siguiente paso basado solo en evidencia verificada.


## Handoff update after repair merge (2026-09-18)

T-023 is complete. The canonical inductive artifact is now `outputs/inductive_full_corrected.jsonl` with 8,712 validated rows. The original `outputs/inductive_full.jsonl` is retained as a pre-repair audit artifact. The next chat should begin with T-024: aggregate the corrected inductive artifact and analyze seen/unseen model and prompt conditions. Do not mix inductive values with P0 values or draft claims before producing machine-readable aggregates.

### Updated handoff prompt

> Continue ProMoRouter on `fix/p0-routing-integrity`. Read `docs/chat-context.md`, `docs/experiment-status.md`, `docs/decisions.md`, and `docs/next-steps.md`. T-023 is complete: `outputs/inductive_full_corrected.jsonl` is the canonical 8,712-row inductive artifact, validated for 121 queries, 12 models, 6 prompts, unique action keys, non-empty responses, and repaired truncation cases. Execute only T-024 first: build a machine-readable aggregation of performance and token-cost proxy by seen/unseen model, seen/unseen prompt, and masked prompt–model conditions. Keep inductive results separate from P0, distinguish verified facts from hypotheses, and update the continuity documents after analysis.


## Latest handoff update (2026-09-18)

T-024 is complete and persisted in GitHub. T-010 is the sole active task. The
reproducible P0 aggregator is
`analysis/aggregate_p0_closed_pool_results.py`; it has passed syntax compilation
and CLI parsing but has not been executed against the five-seed outputs. Those
outputs remain on the `morel` experiment host under the four `outputs/p0_*` and
`outputs/p1_router_flat_mlp_qnorm/` directories. Run the documented T-010
command on `morel`, then inspect and share the generated JSON, CSV, and LaTeX
artifacts. Do not mix these closed-pool results with the inductive aggregates.


## Latest handoff update after T-010 (2026-09-18)

T-010 is complete. The aggregator ran successfully on `morel`, validating 20
seed source files and producing 135 normalized seed-level rows and 27 aggregate
rows for nine methods. The canonical artifacts are
`analysis/p0_closed_pool_aggregates.json`,
`analysis/p0_closed_pool_table.csv`, and
`analysis/p0_closed_pool_table.tex`. Their internal consistency was verified in
the continuation workspace, but the 20 source JSON files were not independently
rerun there. P0 and inductive results remain separate. The sole active task is
now T-011.


## Latest handoff update for T-011 (2026-09-18)

T-011 is prepared but not yet complete. D-023 defines the historical routing
ablation as a controlled 2x2 matrix over observed top-k `{3,5}` and the complete
prompt-model lattice `{off,on}`. The action space remains 36 candidates for
every configuration. The manifest, runner, and aggregator are implemented and
passed compilation/dry-run checks. Execute the seed-1 gate on `morel`, validate
its aggregate, and only then launch seeds 1--5. Do not call this matrix
GraphRouter-style or Edge-GNN v2: the current lattice flag adds prompt-model
edges only and does not connect evaluation queries to every prompt/model.


## Latest handoff update after T-011 (2026-09-18)

T-011 is complete. Two deterministic seed-1 repetitions matched exactly after
excluding only `model_path`. The complete deterministic 2x2 sweep validated 20
source JSON files, 60 seed-level rows, and 12 aggregates for seeds 1--5. The
canonical artifacts are `analysis/p0_routing_ablation_aggregates.json` and
`analysis/p0_routing_ablation_table.csv`. No consistent advantage was found for
top-k 5 or for adding the prompt--model lattice; this historical ablation is
not GraphRouter-style Edge-GNN v2 and remains separate from frozen P0.

D-026 freezes a staged v2 design. The sole active task is T-025: specify and
review the Stage A implementation contract for a GraphRouter-style full
message-passing arm, a matched two-layer self-only arm, and the existing
Flat-MLP, all with the same 36 actions and current objective. Do not implement
or run v2 until that contract and leakage audit are complete. Objective,
density, factorized-router, and inductive variants remain later gated stages.


## Latest handoff update after T-025 (2026-09-18)

T-025 is complete. The binding pre-implementation contract is
`docs/edgegnn-v2-stage-a.md`, mirrored by the machine-readable manifest
`configs/edgegnn_v2_stage_a.json`. D-027 requires independent per-query ego
graphs to prevent cross-query transductive coupling and makes full message
passing versus a matched self-only arm the primary causal comparison.
The Flat-MLP architecture must be retrained under the shared Stage A objective;
its frozen P0 outputs are not reused as a matched run.

The sole active task is T-026: implement Stage A and its static/synthetic/CPU
smoke validations without launching GPU training. Preserve P0, keep inductive
evaluation separate, and do not add Stage B objectives, density variants, or a
factorized router during T-026.
