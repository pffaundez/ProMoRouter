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
- [ ] Resolver el contrato de coste y selección de checkpoint para la tabla
  principal. La v1 usa coste normalizado por consulta y selecciona por reward
  de validación. El runner actual usa coste monetario normalizado con una
  referencia ajustada en Direct de los modelos vistos y elige checkpoints de
  grafo por macro-F1. Los campos qnorm que exporta son métricas secundarias
  de las decisiones ya tomadas. Registrar la decisión antes de ampliar corridas.

### 1.2. Revisiones de OpenReview

- [ ] Recuperar los comentarios y la decisión de
  [OpenReview](https://openreview.net/forum?id=e2yVbty4We). El artículo se pudo
  identificar, pero la página del foro exige verificación y los comentarios
  aún no se han inspeccionado.
- [ ] Registrar cada observación pertinente sobre arquitectura, validez de
  la comparación o presentación de resultados, con enlace o ID del comentario,
  respuesta propuesta y evidencia de cierre. Incorporar las tareas concretas
  a este checklist.

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
- [ ] Preparar y entrenar las políticas aprendidas por λ una vez cerrado el
  contrato. Completar 0.1 y 0.5 y decidir si 0.9 requiere repetición.
- [ ] Completar la tabla con GraphRouter Direct, Flat-MLP, los controles
  fijos actuales y ambos oráculos. Usar el mismo test y la misma definición
  de recompensa en cada comparación. Documentar el alcance de los baselines
  históricos de v1 que no estén en el runner actual.
- [ ] Calcular ambos oráculos por consulta y por λ. Usar Direct para el
  oráculo solo de modelo y todas las parejas elegibles para el oráculo
  conjunto. Cuantificar el aumento de techo y cuánto captura el router en
  test. Reportar diferencias pareadas y el espacio de acciones de cada fila.

## 3. RQ2 — adaptación a modelos no vistos

**¿Cómo se compara ProMoRouter v2 con otros routers al incorporar LLM no vistos con soporte limitado, y qué efecto tiene el soporte con el checkpoint congelado?** La recompensa en el pool ampliado es la comparación principal entre métodos. El contraste emparejado 0 vs. 8 aísla el efecto del soporte.

- [x] Implementar el protocolo 6 vistos + 3 nuevos. A valida el pipeline y
  B/C cambian el trío retenido. Los folds comparten consultas, por lo que
  sus resultados tienen dependencia.
- [x] Registrar el soporte actual: ocho consultas. La GNN conjunta consume
  hasta 96 observaciones (8 × 3 modelos nuevos × 4 prompts) y GraphRouter
  Direct consume 24 (8 × 3 × 1). Las corridas A, B y C entregadas tienen la condición con soporte para los grafos.
- [x] Cerrar T-039 (8→9): el control y el entrenamiento episódico obtuvieron
  R = 0.30704 y eligieron Qwen2.5-14B + decompose en 121/121 consultas.
  La ganancia episódica aparente frente a su propia base provenía de una
  base más débil. El margen calibrado apenas discriminó la ventaja por consulta.
- [x] Validar las tres corridas 6→9 (semilla 1, λ = 0.9). Comparten dataset,
  manifiesto, soporte y consultas de test. La normalización se ajusta por
  fold y se comparte entre métodos.

  | Método | A | B | C | Media |
  |---|---:|---:|---:|---:|
  | GNN conjunto congelado | 0.32229 | 0.27460 | 0.23576 | 0.27755 |
  | GraphRouter Direct congelado | 0.22145 | -0.00426 | 0.13823 | 0.11847 |
  | Flat-MLP actualizado | 0.37057 | 0.34011 | 0.32714 | 0.34594 |
  | Pareja fija conocida | 0.35724 | 0.28122 | 0.32101 | 0.31982 |
  | Pareja fija adaptada | 0.25761 | 0.34256 | 0.32101 | 0.30706 |

  El GNN supera a Direct en los tres folds, con espacios de acciones de
  36 y 9 respectivamente. Queda por debajo de Flat-MLP y de la pareja
  fija conocida en los tres. El GNN elige un LLM distinto por fold
  (Qwen2.5-14B, Qwen2.5-7B y Llama3.1-8B), siempre con decompose en
  las 121 consultas. La incorporación técnica está verificada. La
  adaptación útil por consulta sigue siendo un criterio abierto.
- [ ] Añadir el control pareado de 0 vs. 8 consultas en un split fijo. Para
  cada GNN evaluar el mismo checkpoint, pool ampliado, prompts, información
  estática, métricas y consultas de test. En 0 shots omitir las observaciones
  de soporte de los modelos nuevos. Verificar que los tres modelos se puedan
  puntuar en frío con la representación disponible.
- [ ] Medir ΔR por consulta, con ΔP y ΔC, incertidumbre y frecuencia de
  selección de nuevos modelos. Distinguir el efecto de añadir candidatos
  del efecto de revelar soporte. Documentar el presupuesto de observaciones
  al comparar métodos y aplicar un control con acciones equivalentes si se
  quiere atribuir la diferencia al mecanismo de adaptación.
- [ ] Tras validar el contraste principal, comprobar si el efecto se mantiene
  con otros tríos retenidos. El referente histórico de GraphRouter usó
  6 vistos, 4 nuevos y 80 consultas en Balance. Mantener las 8 consultas
  para la primera comprobación de v2 y reconsiderar el presupuesto solo
  si responde a una pregunta experimental explícita.
