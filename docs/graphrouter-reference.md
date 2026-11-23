# Referencia few-shot de GraphRouter: código y datos de los autores

Fecha: 2026-11-23. D-039. Estado: implementación, pruebas y una ejecución CPU
completa terminadas. No se usa ningún dataset, prompt o scorer de ProMoRouter.
Esta referencia reconstruye la evaluación few-shot ausente del driver público;
**no se declara una réplica exacta de todos los resultados del paper**.

## Fuentes fijadas

Repositorio: ulab-uiuc/GraphRouter, commit
`0edfb8758d45acd93c8029354867ffb49bb8a97f`.

- Paper ICLR 2025, secciones 3.2, 3.5 y 4.2, Tabla 5:
  https://proceedings.iclr.cc/paper_files/paper/2025/file/41b6674c28a9b93ec8d22a53ca25bc3b-Paper-Conference.pdf
- Encoder y entrenamiento públicos:
  https://github.com/ulab-uiuc/GraphRouter/blob/0edfb8758d45acd93c8029354867ffb49bb8a97f/model/graph_nn.py
- Construcción de datos, particiones y etiquetas:
  https://github.com/ulab-uiuc/GraphRouter/blob/0edfb8758d45acd93c8029354867ffb49bb8a97f/model/multi_task_graph_router.py
- Configuración pública:
  https://github.com/ulab-uiuc/GraphRouter/blob/0edfb8758d45acd93c8029354867ffb49bb8a97f/configs/config.yaml
- CSV precomputado enlazado por el README de los autores:
  https://drive.google.com/file/d/1YYn-BV-5s2amh6mKLqKMR0H__JB-CKU4/view

CSV descargado y verificado, SHA-256:
`df6fd3e7cd932b85801d8cb4d782188250266dad5f53eff5ed95253134a6349c`.
Las descripciones, embeddings LLM y configuración se descargan de URLs fijadas
al commit y se verifican con sus Git blob SHAs. No hay llamadas a LLMs, costes
de regeneración ni sustitución automática por nuestros logs.

## Lo que reveló la descarga

El CSV tiene **240 bloques de consulta y 2400 interacciones**: por cada una de
Alpaca, GSM8K, Multi-News y SQuAD hay 60 consultas por diez modelos. La Tabla 2
dice 600 Cases por tarea; el artefacto tiene 600 filas de interacción por tarea.
No se resuelve aquí si Cases se refiere a consultas o interacciones. Se reportan
las dos unidades y no se infiere que el artefacto sea la población completa
que generó todas las tablas. `paper_population_match=false` expresa que no hay
600 consultas por tarea bajo esa interpretación literal.

El nombre del cuarto modelo está vacío en 240 filas. El driver original ignora
esa etiqueta y asigna modelos por posición según LLM_Descriptions.json.
El lector conserva esa semántica: solo el cuarto campo vacío se asigna a
LLaMA-2 (7b), verificando todos los demás nombres y el orden en cada bloque.
No se reescribe el CSV. El informe registra esta resolución explícitamente.
Todos los embeddings de consulta/tarea se comprobaron iguales a los obtenidos
por el parser original de los autores, en las 240 consultas.

## Matriz de fidelidad y decisiones

| Elemento | Paper / código público | Referencia implementada |
|---|---|---|
| Candidatos few-shot | Primeros seis conocidos, últimos cuatro nuevos | Misma lista y orden de las descripciones públicas |
| Datos | Cuatro tareas y diez modelos; CSV enlazado por README | CSV publicado verificado por hash, sin logs PR |
| Soporte nuevo | 80 consultas de train; no driver few-shot ni IDs auxiliares publicados | 80 índices uniformes con seed1881; cuatro outcomes por consulta, 320 acciones |
| Consultas de soporte conocidas | Las consultas pueden pertenecer al train conocido | No retirar esas 80 consultas del entrenamiento conocido; solo ocultar sus outcomes nuevos |
| Split | Paper 70/10/20; código corta bloques contiguos por tarea | Igual corte público: 168 train, 24 val, 48 test |
| Reward | Few-shot en Balance | 0.5*effect - 0.5*cost |
| Normalización | CSV del autor ya normaliza coste por tarea | Reusar effect/cost almacenados sin normalizar con nuestra receta |
| Arquitectura | Paper describe GAT/32; código usa GeneralConv/common_dim8, ancho16 | Código público fijado, sin editar encoder/decoder |
| Representación de tarea | Texto menciona grafo heterogéneo; FeatureAlign público concatena proyecciones task/query | Mantener exactamente FeatureAlign, no inventar nodos/tipos nuevos |
| Optimizador | Paper Adam/LR decreciente; código AdamW 0.0003/wd0.0001 constante | Configuración pública, no una mezcla silenciosa |
| Máscaras | Código usa 32 máscaras/paso, tasa0.5 | Igual layout completo de aristas y un paso sobre la pérdida media |
| Features durante train | Código pasa embeddings de todas las consultas al encoder/BatchNorm | Mantenerlo y declararlo: val/test features presentes, sus outcomes no son visibles |
| Historia para val | Solo train conocido | Igual |
| Historia para test | Código usa train+val | Solo train+val conocidos, más soporte nuevo cuando corresponde |
| Estado de evaluación | Código guarda mejor F1 pero test() no lo recarga | Primario: estado final de época1000; guardar aparte mejor estado val, sin elegir con test |
| Test durante train | Código registra test cada época | Omitir esas llamadas: test solo después de entrenar; no altera gradientes ni selección |
| Modelo nuevo | Descripciones y soporte con parámetros congelados | Añadir cuatro nodos y sus 320 aristas; pesos y BN congelados |

Estos puntos distinguen datos publicados, implementación pública y reconstrucción.
La partición de soporte exacta de los autores no está disponible; el seed se fija
antes del resultado y no se busca otro para mejorar la cifra. El uso de todas
las features de consulta en BatchNorm se conserva por fidelidad al código: no
se presenta esta ejecución como entrenamiento que nunca vio features de test.

## Implementación

- `router/graphrouter_reference.py`: lectura/validación del CSV, verificación de
  artefactos, split, grafo y evaluación. Usa el EncoderDecoderNet ya vendorizado,
  sin modificarlo y sin una cabeza de scoring nueva.
- `prepare_graphrouter_reference.py`: descarga pública, informes y prepared.pt.
  El pickle NumPy de embeddings se verifica por hash y usa un unpickler limitado
  a las clases necesarias para arrays, no una carga arbitraria de objetos.
- `train_graphrouter_reference.py`: entrenamiento conocido, descripción y
  fewshot80 del mismo estado congelado, más un modelo fijo conocido elegido
  solo con validación. No aprende prompts ni modifica la agregación.
- `scripts/run_graphrouter_reference.sh`: preparación y una repetición opcional.

Un cambio en hashes o alineación detiene el proceso. No hay fallback a otro
benchmark, a Qwen como único retenido o a un presupuesto k8.

## Pruebas

52 pruebas CPU pasan, incluidas diez nuevas para esta referencia. Cubren:
partición/soporte, alineación CSV y etiqueta vacía controlada, formatos NumPy,
restricción de pickle, outcomes ocultos, inclusión deliberada de features de
consulta, equivalencia de un paso de entrenamiento con el loop público,
BatchNorm congelado, sensibilidad al soporte, invariancia a orden/lote y
entrenamiento sintético de dos épocas con las tres condiciones.

Además se descargaron/validaron los artefactos reales y se completaron **1000
épocas con los datos publicados**. Esto es una ejecución real, no un smoke
sintético. Runtime: Torch 2.15.0+cpu, PyG 2.8.1, seed1881.

## Resultado medido

| Condición | P | C | R | Modelos elegidos |
|---|---:|---:|---:|---|
| Solo seis conocidos | 0.44531814 | 0.03062116 | 0.20734849 | LLaMA-3 (8b): 48 |
| Añadir descripciones de cuatro nuevos | 0.44531814 | 0.03062116 | 0.20734849 | LLaMA-3 (8b): 48 |
| Añadir soporte80 de cuatro nuevos | 0.44531814 | 0.03062116 | 0.20734849 | LLaMA-3 (8b): 48 |
| Modelo fijo conocido elegido en validación | 0.44531814 | 0.03062116 | 0.20734849 | LLaMA-3 (8b): 48 |

El R few-shot redondea al **0.207** publicado en Tabla5. **Es una coincidencia
escalar a tres decimales, no validación suficiente de la reproducción integral**.
Las predicciones originales de los autores no están disponibles para comprobar
que su política fuese la misma. No se afirma que el paper seleccionara siempre
un modelo conocido.

En esta ejecución sí podemos comprobar que ninguno de los cuatro modelos nuevos
fue seleccionado. El soporte cambia sus scores (máximo cambio absoluto 0.33966564),
los scores conocidos no cambian y las decisiones tampoco. Hay 14/48 oportunidades
oráculo para que un modelo nuevo supere a los conocidos; R oráculo=0.23101927.
Por tanto, no es ausencia total de oportunidad ni soporte desconectado.

El mejor estado por macro-F1 conocido se registró en época157 y se conserva
separadamente; el estado evaluado primario es época1000, como el test público.
No se ha elegido entre ambos mirando cuál resulta mejor en test.

La referencia **no valida todavía la premisa de que reproducimos una adaptación
útil a modelos nuevos**. Impide inferir esa propiedad solo porque aparezca R=0.207.
ProMoRouter permanece pausado: no se lanza una extensión a prompts ni otra
variante automáticamente.

## Artefactos persistidos

En `analysis/graphrouter_reference/seed1881_balance/`:

- `results.json`: historia de 1000 épocas, scores, splits y métricas.
- `data_report.json`, `artifact_sources.json`, `source_audit.json`, `manifest.json`:
  trazabilidad, hashes y coincidencia estructural del encoder con upstream.

Los pesos se conservan en el archivo entregado por separado
`graphrouter_reference_seed1881_weights.zip`: `router.pt` contiene el estado
final y `best_validation_router.pt` el mejor por F1, no usado en el primario.
Sus hashes están en el manifest del repositorio y en la procedencia del archivo.

Hash SHA-256 del resultado primario:
`5e11700e3c41df0752e3dba2efa46a1f05b5d43e1f62f4a82d8f95d1ee1665cc`.

## Repetición opcional en Morel

Los resultados y checkpoints ya están guardados; no hace falta un nuevo fit
para revisarlos. Para repetir deliberadamente la referencia en GPU0:

```bash
source ~/repos/graph-router-2/.venv/bin/activate
cd ~/repos/ProMoRouter
git pull --ff-only
bash scripts/run_graphrouter_reference.sh
```

El primer uso descarga el CSV público y artefactos fijados; no necesita clave
Together/OpenAI ni ejecuta LLMs. Dependencias: el entorno PyTorch/PyG ya usado.
Resultados nuevos: `outputs/graphrouter_reference/seed1881_balance/`.
Se rechaza sobrescribir una ejecución existente. GPU1 es un argumento opcional
`cuda:1`, no una segunda ejecución recomendada.

Si Drive devuelve permisos/cuota/HTML, el programa se detiene explicándolo;
no intenta regenerar interacciones. Se puede proporcionar una copia del mismo
CSV con `python prepare_graphrouter_reference.py --router-data-path RUTA`.
El hash sigue siendo obligatorio. Una repetición GPU puede variar por versión
PyTorch/dispositivo; su resultado se conserva separado del primario CPU.


## Dictamen posterior de fidelidad — 2026-11-23

La auditoría acotada terminó con **partial_reproduction**. Se ejecutó el método
original train_validate en un fixture y se verificó el resultado canónico;
[graphrouter-fidelity-verdict.md](graphrouter-fidelity-verdict.md) contiene la
evidencia y límites. Se corrigió únicamente el origen CPU del RNG para futuras
máscaras GPU; la referencia CPU permanece válida en los bloques auditados,
sin repetir las 1000 épocas. best_f1 sí se actualiza en el upstream: se guarda
el mejor estado, aunque test() evalúa el actual sin recargarlo.
No se inicia una extensión de prompts ni otro barrido.
