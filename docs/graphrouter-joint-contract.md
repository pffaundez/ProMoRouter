# Contrato: extensión conjunta mínima del GraphRouter oficial

Fecha: 2026-11-23. Decisión D-034. Estado: implementación validada con pruebas sintéticas;
ejecución del experimento conjunto con datos reales pendiente en Morel.

## Pregunta única

¿El mecanismo de incorporación del GraphRouter oficial, extendido mínimamente
para distinguir prompts, mejora la recompensa al añadir un modelo excluido del
entrenamiento, manteniendo congelados encoder, alineadores, decoder y BatchNorm?

Los pilotos SupportGraphRouter y el umbral están cerrados, pero NO responden
esta pregunta: usaban otro encoder/scorer. El control oficial anterior solo
respondía a selección de modelo con Direct fijo. Su resultado negativo no
sustituye este experimento conjunto. No se añade un gate externo ni otro scorer.

## Representación elegida y cambios exactos

Se elige una representación única para evitar abrir variantes: un nodo fuente
por consulta–prompt y un nodo destino compartido por LLM. Los cuatro nodos
fuente de una consulta comparten su embedding de consulta y tarea, pero reciben
el embedding de su prompt correspondiente. No hay nodos independientes de
prompt, de tarea ni de pareja prompt–modelo. Los modelos NO se duplican por
prompt. Esta es una decisión de extensión mínima, no la afirmación de que el
GraphRouter original ya representa prompts de esta manera.

| Componente | Control Direct oficial adaptado | Extensión conjunta |
|---|---|---|
| Fuente | Consulta q | Consulta–prompt (q,p) |
| Entrada de consulta | e_q | Concatenación [e_q,e_p] |
| Tarea | Embedding por consulta en FeatureAlign | Mismo embedding repetido para sus cuatro fuentes |
| Destino | Un nodo por LLM m | Un nodo por LLM m, compartido entre prompts |
| Arista histórica dirigida | q → m | (q,p) → m |
| Atributos de arista | [R,C,P] | Los mismos, específicos de (q,p,m) |
| Encoder | Dos GeneralConv y BatchNorm | Sin modificar |
| Decoder | Sigmoid del producto interno medio | Sin modificar |
| Salida | Score por modelo | Score por pareja prompt–modelo |

Reutilizar sin editar `router/graphrouter_official.py`, fijado en upstream
`ulab-uiuc/GraphRouter` commit `0edfb8758d45acd93c8029354867ffb49bb8a97f`.
La proyección query_transform admite la dimensión de entrada concatenada;
se vuelve a entrenar con esa dimensión. No agregar una cabeza MLP de scoring.
El resto del alineador mantiene task_transform y llm_transform oficiales.
Si los embeddings son de dimensión 384, query_transform pasa de 384 a 768
entradas; esto añade parámetros y debe contarse, no presentarse como capacidad
idéntica al control Direct. El ancho interno permanece en 16 (common_dim=8).

Las dos GeneralConv agregan interacciones de todos los prompts en el estado
del mismo modelo. No son dos saltos de ida y vuelta: las aristas continúan
siendo dirigidas de fuentes a modelos. No se agregan aristas inversas,
relaciones prompt–modelo adicionales ni relaciones entre consultas.

Para cada fuente y candidato, el decoder oficial calcula:

`s(q,p,m) = sigmoid(mean(x_ini(q,p) * x_GNN(m)))`.

Son scores de compatibilidad supervisados para elegir una acción, NO una
regresión numérica de R ni probabilidades calibradas de utilidad.
El target de test no aporta resultados ni aristas visibles; solo se predicen
sus aristas candidatas. El embedding fuente del target lo usa el decoder.

## Entrenamiento único y exclusión real

Piloto de desarrollo: seed=1, lambda=0.9, retenido qwen2.5-14b; cuatro prompts,
ocho modelos conocidos. Se conserva la misma población completa de 797
consultas y el manifiesto seed1: 557 train, 119 validación, 121 test.
Reservar los mismos ocho qids de train como soporte; 549 qids generan aristas
supervisadas y los 557 aportan historia conocida, como en el control oficial.

Entrenar conjuntamente el FeatureAlign completo, edge_mlp y las dos GeneralConv
con sus BatchNorm; decoder sin parámetros adicionales. Embeddings descriptivos
fijos y ningún ajuste de LLM. No reutilizar los pesos del SupportGraphRouter.

Mantener BCE, AdamW lr=0.0003/wd=0.0001, 1000 épocas, 32 máscaras aleatorias
por paso y probabilidad de enmascarar arista elegible 0.5. Una etiqueta positiva
por CONSULTA ORIGINAL corresponde a la mejor pareja entre sus 4x8 acciones
conocidas; no una positiva por consulta–prompt. Esto cambia el espacio de
acciones y la proporción positiva de 1/8 a 1/32 respecto al control Direct;
es una consecuencia explícita de la extensión, no un efecto aislado de prompt.
La identidad de clase es (prompt,modelo), con orden semántico documentado.

Seleccionar checkpoint por macro-F1 de la mejor acción en validación conocida,
con la misma convención de clases presentes del adaptador Direct. No elegir
checkpoint con datos del modelo retenido ni con test. Reportar también R de
validación para evidenciar posibles desacuerdos con macro-F1, sin cambiar el
criterio a posteriori. Cargar el mejor estado antes de incorporar al retenido.

Excluir por completo las observaciones de Qwen del entrenamiento, las máscaras,
la selección de checkpoint y el ajuste de costes. Enmascarar outcomes retirados
antes de construir atributos, no solo después del message passing. Las aristas
predichas de entrenamiento no son visibles al encoder, como en el adaptador
oficial; las demás aristas históricas conocidas sí pueden ser visibles.

## Coste común para interpretar el contraste

Reutilizar la referencia del control Direct: mínimo/máximo de cost_proxy_money
entre las acciones Direct de los 549 qids de ajuste y los ocho modelos conocidos.
Aplicarla sin reajuste a TODOS los prompts, soportes, candidatos y condiciones:
`C=(cost_proxy_money-lo)/(hi-lo)` y `R=P-0.9*C`. No recortar extrapolaciones.

Así el Direct histórico y la extensión comparten unidades primarias si coinciden
hash del dataset, manifiesto, soporte, candidatos y extremos. Verificar esa
igualdad antes de comparar resultados. No reusar resultados Direct si no se
puede verificar; una eventual discrepancia es un bloqueo de comparabilidad,
no permiso para mezclar escalas. No usar valores del retenido para ajustar lo/hi.
Qnorm original se reporta solo como métrica secundaria, igual que en el control.

Exigir coste monetario proxy válido para todas las acciones conjuntas; fallar
sin fallback a qnorm si faltan campos. Costes fuera de [0,1] son esperables con
la referencia Direct y deben reportarse, no corregirse usando test.

## Una adaptación, tres condiciones del mismo checkpoint

1. **Solo conocidos:** cuatro prompts por cada uno de los ocho modelos vistos.
2. **Descripción:** añadir un nodo Qwen con su embedding, sin aristas observadas
   hacia él. Puntuar sus cuatro parejas con el decoder oficial.
3. **Calibrado k8:** mismo nodo Qwen; añadir únicamente las 8x4 aristas históricas
   (q,p) → Qwen de los ocho qids reservados, con sus atributos [R,C,P].

Conservar íntegra la historia de los modelos conocidos. No volver a entrenar,
recalibrar BatchNorm ni modificar el decoder al añadir el nodo/aristas.
El presupuesto es ocho consultas, pero 32 acciones del modelo nuevo; el control
Direct k8 usa ocho acciones. No afirmar comparación a igual coste de calibración.
No k80, umbrales, ajustes externos, otras temperaturas ni varios folds en este piloto.

La comparación primaria es calibrado menos solo conocidos dentro del checkpoint
conjunto. La diferencia calibrado menos descripción mide el aporte del soporte.
Comparar con el control Direct previo de manera secundaria: cambian espacio de
acciones, dimensión de entrada y datos históricos, no solo una pieza del grafo.
Opcionalmente reportar la restricción a Direct del mismo checkpoint como lectura
descriptiva sin entrenamiento adicional; no llamarla baseline Direct entrenada.

## Verificaciones exigidas antes de Morel

- Identificar todas las acciones por (qid,prompt,modelo), sin depender de orden
  incidental; verificar el espacio completo y ausencia de duplicados.
- Confirmar exclusión del retenido de labels, contexto y normalizador de ajuste.
- Atributos de target ausentes: perturbar sus outcomes no cambia sus scores.
- Modificar soporte Qwen puede cambiar su score; modificar outcomes no observados
  no puede hacerlo. Cambiar el prompt de una fuente cambia solo su identidad de
  entrada correspondiente, sin reasignar modelos o etiquetas.
- Params y buffers BN idénticos antes/después de incorporación. En eval, añadir
  Qwen no cambia los scores de candidatos conocidos en este grafo dirigido.
- Los scores de una consulta son invariantes al lote de otros targets y al orden
  de candidatos, tras reordenar las salidas.
- Con un único prompt Direct y parámetros de proyección compatibles (bloque de
  entrada prompt a cero), el adaptador conjunto debe reproducir los scores del
  adaptador Direct en un fixture sintético usando pesos copiados, sin entrenar.
- Conteos: 557x4 fuentes históricas, ocho/nueve destinos según condición;
  557x4x8 aristas conocidas visibles en evaluación; 32 aristas Qwen adicionales
  solo en calibrado. Targets jamás aparecen como fuentes visibles.
- Ejecutar un smoke CPU sintético con checkpoint y las tres condiciones;
  preservar resultados y código de experimentos anteriores.

## Lectura y cierre del piloto

Guardar P/C/R primarios y qnorm secundarios, scores por acción, selecciones y
perfiles por tarea/prompt/modelo/pareja, concentración, ganancia de incorporación,
ganancia de calibración, oportunidades oráculo y selecciones beneficiosas y
perjudiciales frente a la base conjunta. Guardar estado, config, hashes,
referencia de costes y qids para auditar sin nuevo entrenamiento.

Si la ganancia calibrado–base no es positiva, cerrar el piloto como ausencia de
mejora útil bajo esta configuración. Si es positiva, reportarla como evidencia
de desarrollo que requiere confirmación; no escalar automáticamente a un barrido.
En ambos casos examinar oportunidad oráculo y perfiles sin cambiar retrospectivamente
el contrato. No hay una afirmación de significación o generalización a partir
de un único fold observado previamente.

Esta fase prueba una extensión mínima específica del mecanismo oficial, no todas
las posibles arquitecturas de grafo con prompts. La incorporación física no basta:
la pregunta principal sigue siendo si mejora las decisiones de inferencia.

## Implementación y ejecución — 2026-11-23

Contrato implementado en `router/graphrouter_joint.py` y
`train_graphrouter_joint_unseen.py`; módulo oficial sin cambios.
Pasaron 35 pruebas CPU: 14 de GraphRouter (10 nuevas conjuntas) y 21 de
regresión de soporte/auditoría/umbral. La integración entrena dos épocas
sintéticas y evalúa las tres condiciones; no es evidencia de mejora real.
No se ha ejecutado el piloto con datos de Morel.

Comando desde la rama v2-behavior-profiles en Morel:

```bash
source ~/repos/graph-router-2/.venv/bin/activate
cd ~/repos/ProMoRouter
git pull --ff-only
bash scripts/run_graphrouter_joint_unseen.sh
```

Usa las dependencias de `requirements-graphrouter.txt` ya empleadas por el
control Direct. GPU0 por defecto. Para GPU1, ejecutar en su lugar:

```bash
bash scripts/run_graphrouter_joint_unseen.sh ~/repos/graph-router-2/data cuda:1
```

Es un único entrenamiento; no ejecutar ambos comandos. Guarda checkpoint,
historia, normalizador, hashes, scores, selecciones, conteos del grafo y métricas
en `outputs/graphrouter_joint_unseen/qwen2.5-14b/seed1_lam0.9/`.
Enviar `results.json` al terminar; se rechaza sobrescribir una carpeta existente.

La referencia Direct se busca en
`outputs/graphrouter_unseen/qwen2.5-14b/seed1_lam0.9/results.json`.
`direct_reference` informa `verified`, `unavailable` o `mismatch`.
Solo `verified` permite comparación numérica histórica en la misma escala;
no implica igualdad de capacidad, datos históricos ni presupuesto de soporte.
Una incompatibilidad no invalida el contraste primario entre condiciones del
mismo checkpoint; bloquea su interpretación como contraste histórico alineado.
No se elige checkpoint ni hiperparámetros usando resultados de test Direct.
