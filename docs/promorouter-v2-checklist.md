# ProMoRouter v2 — checklist

La v1 es el manuscrito [ProMoRouter: Joint Prompt and Model Routing for Cost-Efficient LLM Inference](https://openreview.net/pdf?id=e2yVbty4We) que no superó la revisión. Se verificaron sus RQ y su protocolo en el PDF facilitado por el autor. La v2 conserva su pregunta sobre
enrutamiento conjunto como pregunta principal y añade la incorporación de
modelos no vistos. El [registro de decisiones](decisions.md) documenta los
acuerdos. El [protocolo 6→9](unseen-6x3-comparison.md) describe las corridas
existentes.

## 1. Correcciones metodológicas

### 1.1. Volver a GraphRouter y extenderlo a `(p, m)`

- [x] Auditar la implementación pública de GraphRouter. El resultado es una
  [reproducción parcial](graphrouter-fidelity-verdict.md): se verificó la
  equivalencia del bucle público y se recuperó la recompensa agregada, sin
  verificar la política exacta publicada ni una mejora debida al soporte.
- [x] Implementar la extensión que selecciona `(p, m)` y el control
  `graphrouter_direct` sobre el encoder oficial fijado. El runner 6→9 usa
  etiquetas de mejor acción y máscaras agrupadas por consulta en ambos brazos.
- [x] Verificar sintéticamente la separación de ajuste, soporte y test, las
  observaciones consumidas y la invariancia de pesos y buffers en los grafos.
- [ ] Cerrar la auditoría metodológica de v2. Confirmar qué información está
  disponible al puntuar una consulta nueva, cómo se enmascaran sus resultados
  y qué diferencias frente a GraphRouter deben declararse. Verificar que el
  presupuesto de soporte, el conjunto de acciones y la recompensa se describan
  tal como los usa cada método.
- [x] Fijé la arquitectura, la escala de coste y el criterio de checkpoint
  de RQ1. Mantengo el encoder fijado de GraphRouter con fuentes consulta–prompt,
  un nodo por LLM, dos capas GeneralConv, máscaras por consulta y el decoder
  actual. Entrenaré con la pérdida binaria de pareja ganadora, por separado
  para cada λ. Seleccionaré checkpoints por recompensa media de validación
  y usaré macro-F1 de selección como diagnóstico.
- [x] Verifiqué el cálculo implementado del proxy: tokens de entrada por
  tarifa de entrada más tokens de salida por tarifa de salida. El valor
  almacenado no divide por un millón. La escala min–max es invariante a
  esa conversión común. Reportaré el valor como proxy de coste ponderado
  y usaré unidades monetarias solo si están documentadas las tarifas.
- [x] Fijé la selección del checkpoint. Maximizaré la recompensa media de
  validación por λ. Una mejora efectiva supera el mejor valor guardado por
  más de 1e-6. Las diferencias dentro de esa tolerancia conservan el primer
  checkpoint. La parada temprana ocurre tras 18 épocas consecutivas sin
  mejora efectiva. El tope es 1000 épocas para los grafos y 80 para Flat-MLP.
- [x] Ejecuté [la auditoría de costes](../analysis/audit_rq1_cost.py) en
  Morel durante la corrida corta. Las 28.692 acciones coinciden con el
  log de origen, con error absoluto máximo cero.
- [ ] Verificar cobertura de tokens de las estrategias multillamada en el
  código de generación original o en los registros por llamada.
- [x] Implementé la normalización común y la selección de checkpoint en el
  [runner de RQ1](../run_rq1_closed_pool.py). Validé los tres λ en CPU con
  datos sintéticos, incluyendo 1.296 costes y la invariancia de scores y
  checkpoints al modificar P y C de test.
- [x] Validé la corrida corta de RQ1 en GPU 0 de Morel con la semilla 1.
  Completó los tres λ y nueve métodos. Las 121 consultas de test son
  comunes y se verificó R = P − λC por consulta y en los agregados.
- [ ] Analizar perfiles con el log completo. Comparar prompts fijando
  consulta y modelo, y modelos fijando consulta y prompt. Reportar tokens,
  coste bruto, calidad e interacción entre prompt y modelo.

### 1.2. Respuesta a la revisión de ARR

- [x] Revisé la metarrevisión y los cuatro informes, junto con las respuestas
  de autor. Unifiqué las observaciones repetidas en las tareas de abajo y en
  las dos RQ. [Discusión de ARR](https://openreview.net/forum?id=e2yVbty4We#discussion)

| Prioridad | Observación | Qué haré para cerrarla |
|---|---|---|
| Alta | Posicionamiento y novedad (metarrevisión, gY5T y cuarto informe) | Explicaré qué reutilizo de GraphRouter, cómo extiendo la acción a (prompt, LLM) y cómo se compara con Route-to-Reason y el routing de esfuerzo de razonamiento. Situaré TensorOpera Router en trabajos relacionados. |
| Alta | Coste antes del routing, arquitectura y Algoritmo 1 (metarrevisión, gY5T y segundo informe) | Reescribiré el algoritmo y añadiré figuras separadas de entrenamiento e inferencia. Identificaré los embeddings, las aristas visibles y ocultas, las etiquetas, el decoder, la información disponible antes de decidir y el papel del soporte. Eliminaré las definiciones duplicadas de φ. |
| Alta | Coherencia de recompensa (gY5T y metarrevisión) | Cerraré la arquitectura y el protocolo de v2 y construiré la tabla principal con nuevas corridas del router y sus comparadores bajo ese mismo protocolo. Comprobaré R = P − λC por consulta y antes del redondeo. Las discrepancias de las tablas 2 y 3 de v1 quedan como antecedente de auditoría. Si conservo la ablación fijo/adaptativo, la ejecutaré con el protocolo final. |
| Alta | Comparabilidad de baselines (metarrevisión e informes 2 y 3) | Explicitaré acciones, prompt fijo, soporte y actualizaciones por método. Distinguiré la comparación de sistemas con Direct de los controles con el mismo espacio de acciones, como Flat-MLP. Documentaré ambos oráculos y consideraré un control con prompt fijo fuerte seleccionado en validación si hace falta aislar el efecto del prompt. |
| Alta | Reproducibilidad y robustez (metarrevisión e informes 2 y 3) | Incluiré tamaños de splits, semillas, hiperparámetros, selección de checkpoint, versiones, hardware y procedencia de logs. Reportaré resultados por semilla y diferencias pareadas cuando estén disponibles. Distinguiré la incertidumbre por consulta de la variación entre semillas y folds. |
| Alta | Incorporación de candidatos (metarrevisión e informes 2 y 3) | Evaluaré modelos nuevos en RQ2. Reportaré selecciones por modelo y prompt, concentración de decisiones, comparación con políticas fijas y el contraste de soporte con pesos congelados. La extensión a prompts nuevos queda como limitación. |
| Media | Despliegue y escalabilidad (metarrevisión e informes 2 y 3) | Separaré coste de generación, construcción del grafo, scoring, incorporación y latencia extremo a extremo. Explicaré el crecimiento del espacio de acciones y el alcance de los tiempos offline. |
| Media | Alcance de las conclusiones (informes 2 a 4) | Acotaré las afirmaciones a los benchmarks en inglés, los LLM abiertos y la evaluación offline. Explicitaré el riesgo de diferencias de calidad por variedad lingüística. Evaluar modelos propietarios, razonamiento adaptativo o familias adicionales de baselines queda condicionado a que aclare una RQ. |

El coste realizado se usa para evaluar las decisiones o como atributo de
interacciones observadas permitidas. La prueba de perturbación del runner 6→9
ya mostró que alterar resultados retenidos de ajuste y resultados de test deja
intactos los scores. Quedan por cerrar la redacción, la comprobación del
protocolo final y la regeneración de las tablas. Las respuestas del hilo
documentan compromisos de la v1 y las tareas anteriores registran su estado
para la v2.

## 2. RQ1 — enrutamiento conjunto de prompt y modelo

**¿Supera el enrutamiento conjunto de prompt y modelo al enrutamiento solo de modelo?** Es la RQ2 de la v1 (§4) y la pregunta principal de la v2.
La antigua RQ1 sobre el efecto del prompting por sí solo se elimina.

- [x] Conservar la estructura de la Tabla 2 de v1. Reportar P, C y R para
  cada escenario: λ = 0.1, 0.5 y 0.9. R es el criterio principal y P y C
  explican el intercambio entre rendimiento y coste.
- [x] Registrar el antecedente en pool cerrado: cinco semillas con las tres
  penalizaciones. Flat-MLP conjunto lideró los routers aprendidos en 0.1 y
  0.5 y quedó cerca de Edge-GNN en 0.9. El baseline histórico llamado
  GraphRouter-direct era un MLP que fijaba Direct (D-038).
- [ ] Completar la comparación de v2 en pool cerrado. Los resultados 6→9
  documentados abajo corresponden a generalización con candidatos retenidos.
- [ ] Entrenar y evaluar las políticas de v2 para λ = 0.1, 0.5 y 0.9 tras
  cerrar la arquitectura y el protocolo. Usaré esas corridas de pool cerrado
  para la tabla principal de RQ1. Las corridas 6→9 quedan como evidencia de RQ2.
- [ ] Completar la tabla con GraphRouter Direct, Flat-MLP, los controles
  fijos actuales y ambos oráculos. Usar el mismo test y la misma definición
  de recompensa en cada comparación. Documentar el alcance de los baselines
  históricos de v1 que no estén en el runner actual.
- [ ] Calcular ambos oráculos por consulta y por λ. Usar Direct para el
  oráculo solo de modelo y todas las parejas elegibles para el oráculo
  conjunto. Cuantificar el aumento de techo y cuánto captura el router en
  test. Reportar diferencias pareadas y el espacio de acciones de cada fila.

## 3. RQ2 — adaptación sin reentrenar

**¿Qué método se adapta mejor sin reentrenar?**

- [x] Comparé el GNN conjunto y GraphRouter Direct con 0 y 8 consultas de
  soporte en A, B y C. Cada contraste usa el mismo checkpoint congelado,
  nueve candidatos y el mismo test.
- [x] Verifiqué el cambio de recompensa y de decisiones. La recompensa del
  GNN conjunto aumenta en los tres folds y cambia la pareja global en
  todos. GraphRouter Direct cambia 85 de 121 decisiones en C.
- [ ] Preparar la tabla del manuscrito con R sin y con soporte, ΔR,
  incertidumbre y cambios de decisión. Explicitar 36 frente a 9 acciones
  y 96 frente a 24 observaciones con las mismas ocho consultas.

## 4. RQ3 — recompensa al incorporar modelos nuevos

**¿Qué sistema obtiene mayor recompensa al incorporar modelos nuevos con
consultas de soporte?**

- [x] Evalué los sistemas con el pool ampliado y ocho consultas de soporte
  en los tres folds. Lite actualizado supera al GNN conjunto en los tres.
  El GNN conjunto supera a GraphRouter Direct en los tres.
- [ ] Preparar la tabla con la recompensa absoluta y el protocolo de
  incorporación de cada sistema, incluyendo los controles fijos y el
  tiempo registrado. Identificar que Lite actualiza sus pesos y que los
  dos GNN los conservan congelados.

## Evidencia de la auditoría del coste

El [script histórico](../analysis/add_query_normalized_lambda_rewards.py)
define nueve tarifas y calcula el proxy a partir de `cost.tokens_in` y
`cost.tokens_out`. El [constructor del grafo](../analysis/build_bipartite_router_dataset.py)
copia el proxy y el total de tokens. El JSON de T-039 permite invertir su
normalización para una comprobación indirecta. Para las 3.872 acciones de
test de los ocho modelos con tarifa simétrica, el proxy recuperado equivale
a un número entero de tokens con error máximo inferior a 0,000025 tokens.
El modelo con tarifas asimétricas requiere el desglose de entrada y salida.
La comprobación indirecta no acredita la cobertura de llamadas.

El auditor compara todas las aristas del dataset de routing con el log de
origen por ID de consulta, prompt y modelo. Verifiqué con datos sintéticos
que acepta un cálculo correcto con tarifas asimétricas y rechaza un proxy
alterado. La corrida corta de Morel auditó las 28.692 acciones con error máximo cero. La referencia de ajuste de semilla 1 fue [7.6, 517.2].


## Corrida corta de RQ1 en Morel

El runner evalúa el GNN conjunto, GraphRouter Direct, Flat-MLP, controles
fijos y ambos oráculos con el mismo manifiesto. Recibe una semilla por
ejecución y evalúa los tres λ. Para la corrida corta usaré la semilla 1.

```bash
cd ~/repos/ProMoRouter
git switch experiment/unseen-6x3
git pull --ff-only origin experiment/unseen-6x3
source ~/repos/graph-router-2/.venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python run_rq1_closed_pool.py \
  --smoke --seed 1 \
  --data-root ~/repos/graph-router-2/data \
  --raw-log ~/repos/graph-router-2/data/interaction_logs/grpp_il_v1/train_clean.jsonl \
  --split-manifest data/router/splits/qnorm_complete_seed1.json \
  --output outputs/rq1_closed_pool/smoke_seed1.json
```

La auditoría coteja las aristas contra el log original antes de ajustar los
routers. El modo smoke hace una época por método y λ. La corrida de tabla
requiere el log original y GPU 0. La cobertura de llamadas se verifica
aparte con el código de generación o registros por llamada.
