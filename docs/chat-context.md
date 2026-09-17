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
