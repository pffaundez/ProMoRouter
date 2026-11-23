# Memoria de cálculo: intervenciones para incorporar modelos no vistos

Fecha de consolidación: 2026-11-23. Rama: `v2-behavior-profiles`.
Estado: intervenciones de esta etapa cerradas; no se programa otro entrenamiento.

## 1. Objetivo y alcance

Evaluar si ProMoRouter puede seleccionar conjuntamente prompt y modelo y obtener
utilidad al incorporar un LLM excluido del entrenamiento, manteniendo congelados
los parámetros del router durante la incorporación. Distinguir tres propiedades:
aceptar técnicamente un candidato, seleccionarlo y mejorar la recompensa.

Esta memoria consolida el piloto uniforme, su auditoría, la intervención de
agregación por similitud, el control oficial GraphRouter y el umbral global.
Los resultados anteriores proceden de la revisión de artefactos del usuario
registrada en `experiment-status.md`; el umbral se recalculó directamente desde
el JSON entregado en esta conversación. No se ejecutaron nuevos entrenamientos
al redactar este documento. No incluye resultados numéricos del MLP ni de Stage A.

## 2. Contrato de cálculo y comparabilidad

Recompensa: `R = P - lambda*C`; aquí lambda = 0.9. Para una intervención,
`ganancia = R_intervención - R_base`, con consultas y referencia de coste iguales.

Los pilotos conjuntos usan cuatro prompts: direct, cot, decompose, selfcheck.
Su coste de evaluación es el `cost_norm_query` original, fijo entre condiciones.
Entrenamiento y soporte se renormalizan por consulta usando solo acciones de
modelos conocidos. La diferencia entre escala de ajuste y evaluación es una
limitación del protocolo, no una mejora demostrada.

El control oficial usa exclusivamente Direct y coste monetario proxy normalizado
con extremos de entrenamiento conocido: mínimo 7.6, máximo 502.8. No son importes
en dólares demostrados. Sus R/P/C primarios no deben compararse numéricamente
con los del piloto conjunto como si fueran un contraste arquitectónico pareado.

Los experimentos son de desarrollo con semilla 1 sobre el mismo benchmark.
Reutilizar consultas y observar una sola inicialización limita la confirmación.
Los intervalos bootstrap condicionados al checkpoint no sustituyen varias semillas
ni un dataset independiente. El coste de calibración no permite afirmar ahorro
monetario neto amortizado.

## 3. Intervenciones y resultados

### I-01. Soporte uniforme y auditoría del checkpoint (T-028 / T-029)

Modelo retenido: qwen2.5-14b. Soporte: ocho consultas, cuatro prompts
(32 observaciones del modelo nuevo). Particiones: 549 ajuste, ocho soporte,
119 validación y 121 test. Mejor época: 5. La arquitectura es el
SupportGraphRouter simplificado con estados por pareja, no el encoder oficial.

| Condición | P | C_qnorm | R_qnorm | Selecciones Qwen |
|---|---:|---:|---:|---:|
| Solo conocidos | 0.42107605 | 0.07948213 | 0.34954213 | 0 |
| Con descripción nueva | 0.42107605 | 0.07948213 | 0.34954213 | 0 |
| Con soporte nuevo | 0.42107605 | 0.07948213 | 0.34954213 | 0 |

Había 54 consultas con oportunidad oráculo y ganancia oráculo media 0.03445720.
Todas las selecciones usaban selfcheck: Mistral-7B 62, Qwen-7B 59.

La auditoría mostró respuesta al soporte: en esas 54 oportunidades, aumento
medio del mejor score Qwen +0.15176394, mediana de rango 13 a 9 y margen medio
respecto al mejor conocido -0.25701923 a -0.10525529. El prompt Qwen preferido
pasó uniformemente de selfcheck a decompose; los scores conocidos no cambiaron.
Eso no produjo selecciones ni ganancia. Una pareja fija por tarea elegida en
validación obtuvo R=0.32354404 en test: diferencia del router +0.02599808,
sin demostrar por sí sola adaptación por instancia o significación.

**Cierre:** el soporte afecta la representación, pero no demuestra incorporación útil.

### I-02. Agregación condicionada por similitud de consulta (T-030)

Cambiar únicamente la media uniforme por pesos softmax de similitud coseno
entre embeddings de texto de consultas, temperatura fija 0.1. Aplicarlo también
al entrenamiento; no es una edición de inferencia del checkpoint anterior.
Mantener el resto de la configuración; sin barrido de temperaturas.

Resultado: P=0.45073089, C_qnorm=0.12270050, R_qnorm=0.34030043.
Ganancia de incorporación y de calibración: cero; Qwen seleccionado cero veces.
Selfcheck en todas las consultas y selección prácticamente fija por tarea.
Diferencia descriptiva de R respecto al uniforme: -0.00924169.

**Cierre:** no se observó mejora útil con esta única intervención de agregación;
no se continúa buscando temperaturas o variantes de agregación.

### I-03. Control con encoder oficial GraphRouter (T-031)

Código fijado en upstream `ulab-uiuc/GraphRouter`, commit
`0edfb8758d45acd93c8029354867ffb49bb8a97f`. GeneralConv/BatchNorm,
atributos de interacción y decodificador oficiales; bucle adaptado para elegir
checkpoint en validación conocida y congelar pesos y buffers antes de test.
Solo Direct, Qwen retenido, una trayectoria de entrenamiento y cuatro condiciones.
Mejor época 993; macro-F1 de validación conocida 0.2125562.

| Condición | P | C primario | R primario | Ganancia vs conocidos | Qwen elegido |
|---|---:|---:|---:|---:|---:|
| Solo conocidos | 0.33158374 | 0.11815411 | 0.22524504 | 0 | 0 |
| Descripción | 0.33383565 | 0.12370519 | 0.22250097 | -0.00274406 | 24 |
| Soporte k8 | 0.33396743 | 0.12329268 | 0.22300401 | -0.00224102 | 22 |
| Soporte k80 | 0.31791216 | 0.12396566 | 0.20634307 | -0.01890197 | 55 |

Oráculo Direct en la escala primaria: 19 oportunidades, ganancia 0.00817815.
k8: ocho selecciones beneficiosas, 14 perjudiciales; captura 5/19 oportunidades.
k80: 27 beneficiosas, 28 perjudiciales; captura 11/19. Los intervalos de ganancia
incluyen cero. La mayor selección con k80 no prueba mayor utilidad, y las
estimaciones negativas no demuestran daño general o estadísticamente significativo.

**Cierre:** GraphRouter sí incorpora y selecciona técnicamente Qwen en este fold,
pero no demuestra ganancia agregada. No se asume que su éxito publicado se
transfiere automáticamente a nuestro benchmark y protocolo adaptado.

### I-04. Umbral global de incorporación (T-032)

Implementación: commit `f35824a2d8137eca3b6d3af231fd16e7f37cee31`.
Arquitectura uniforme fija, hidden 128, AdamW lr=0.0005 / wd=0.0001,
batch 32, MSE + 0.05 CE, clip 5, máximo 80 épocas, paciencia 18,
semilla 1, lambda 0.9, soporte ocho consultas.

Retenidos internos: llama3.1-8b y codellama-34b. Retenidos externos:
yi-34b y mixtral-8x7b, excluidos de todo ajuste interno. Cada entrenamiento
interno excluye además a su retenido: seis modelos conocidos. Se realizaron
dos entrenamientos, ambos con mejor checkpoint en época 10 y parada en 28.
No se entrenó el router externo compartido.

Particiones verificadas disjuntas: 549 ajuste, ocho soporte, 59 consultas de
selección del umbral, 60 de comprobación y 121 test. Los registros internos
son 118 de selección y 120 de comprobación porque cada consulta aparece en
los dos folds de modelos; no son 238 consultas independientes.

Para cada consulta:

- Elegir el mejor prompt nuevo mediante scores, sin consultar su recompensa.
- Elegir la mejor pareja conocida mediante scores.
- `margen = score_nuevo - score_conocido`.
- `delta_R = R(prompt_nuevo_predicho, modelo_nuevo) - R(pareja_conocida_predicha)`.
- Incorporar si `margen > umbral`; `null` significa rechazar siempre.

Enumerar todas las políticas distintas de umbral estricto, incluyendo umbrales
negativos y rechazar siempre. Maximizar ganancia media en selección; desempatar
por menos incorporaciones. Checkpoint y umbral usan solo las 59 consultas de
selección. La comprobación independiente exige AUC agrupada y por fold >0.5,
ganancia agrupada positiva y ninguna ganancia por fold negativa.

| Fold interno | Beneficios posibles en selección | Beneficios posibles en comprobación | AUC comprobación |
|---|---:|---:|---:|
| Llama-3.1-8B | 7/59 | 9/60 | 0.64488017 |
| CodeLlama-34B | 1/59 | 2/60 | 0.66379310 |

Estos beneficios se refieren al prompt predicho para el modelo nuevo frente a
la decisión conocida, no al oráculo de todos los prompts/modelos.
AUC agrupada=0.61718098. Todos los márgenes observados fueron negativos.
El margen ordena parcialmente los casos beneficiosos frente a los perjudiciales;
eso no garantiza una región de incorporación con recompensa positiva.

**Resultado exacto:** `threshold=null`, `status=stopped_internal_gate`.
Ningún umbral con al menos una selección tuvo ganancia positiva en selección.
La mejor política no vacía tuvo umbral -0.11554968357086182, una selección
entre los 118 registros y ganancia media -0.0041693584393646755.
Rechazar siempre ganó con cero selecciones y ganancia cero.

En comprobación, esa política obtuvo P=0.37059523, C=0.05019217,
R=0.32542227 y ganancia cero. Son promedios de los dos folds, no resultados
externos. El intervalo bootstrap [0,0] es consecuencia mecánica de no cambiar
ninguna decisión; no acredita precisión poblacional sobre otras políticas.

El filtro falló por elegir rechazar siempre y no producir ganancia positiva,
**no por AUC insuficiente**. Yi y Mixtral no fueron evaluados; no hay resultados
externos negativos que reportar. Tampoco se eligió un segundo umbral tras observar
la comprobación.

**Cierre:** un único umbral global sobre estos scores no resolvió la incorporación
útil bajo esta configuración. Se cierra la intervención sin nuevo barrido.

## 4. Trazabilidad y recálculo del umbral

Artefacto en Morel: `outputs/support_threshold/seed1_lam0.9/results.json`.
Copia analizada: adjunto `results(3).json`.
SHA-256 del adjunto:
`386878350b11888bfe3ba485a2026dc3bdc225d38b9067f51e880400bfb370c5`.
SHA-256 del dataset registrado:
`d7bfa5946d7cd737ff01770fd616f69a0ded237c06e10d6a89d1e65e1ffea715`.
SHA-256 del umbral congelado registrado:
`81731e30b737fbfbe8a4b1a863d02c54e89355a9b0971fe7d33bdd41034ad923`.
El JSON original permanece en Morel; esta memoria no lo reemplaza.

Recálculo sin GPU ni nuevo entrenamiento, desde la raíz del repositorio:

```python
import json
from pathlib import Path
r = json.loads(Path('outputs/support_threshold/seed1_lam0.9/results.json').read_text())
a = r['internal_records']['tune']
margins = sorted({x['margin'] for x in a})
thresholds = margins + [margins[0] - max(1., abs(margins[0]))]
policies = []
for t in thresholds:
    selected = [x for x in a if x['margin'] > t]
    if selected:
        gain = sum(x['delta_R'] for x in selected) / len(a)
        policies.append((gain, t, len(selected)))
print('Mejor ganancia no vacía, umbral, selecciones:', max(policies))
print('Umbral elegido:', r['threshold'])
print('Comprobación:', r['internal_gate'])
```

El criterio de desempate completo está en `train_support_threshold.py`;
el fragmento anterior reproduce el mejor valor no vacío de esta ejecución.
Otros contratos: [piloto](support-graph-pilot.md),
[control oficial](graphrouter-unseen-control.md), [umbral](support-threshold.md).
Evidencia histórica y decisiones: [estado](experiment-status.md),
[decisiones](decisions.md), [cola de trabajo](next-steps.md).

## 5. Redacción reutilizable y decisión de cierre

> En una evaluación de desarrollo con modelos retenidos, la calibración modificó
> los scores del router conjunto sin producir incorporación útil del modelo
> nuevo. La agregación por similitud no corrigió este comportamiento. Un control
> adaptado con el encoder oficial de GraphRouter seleccionó el modelo retenido,
> pero no mejoró la recompensa agregada frente al conjunto conocido. Finalmente,
> un umbral global elegido mediante retención interna prefirió no incorporar
> candidatos; aunque los márgenes tenían AUC superior a 0.5, ninguna política
> de umbral no vacía obtuvo ganancia positiva durante la selección. El protocolo
> se detuvo antes de la evaluación externa. Estos resultados delimitan las
> intervenciones examinadas y no demuestran imposibilidad general de adaptación.

No hay evidencia para afirmar que un grafo extendido con prompts, mayor soporte
o un cambio global de umbral resuelva ya la hipótesis del paper. La causa del
problema no queda identificada causalmente por estos contrastes. Se conserva
la evidencia negativa y se cierran estas intervenciones; no se autoriza por
este registro un nuevo entrenamiento o una nueva rama experimental.

## 6. Aclaración posterior de alcance — 2026-11-23

El cierre anterior corresponde a las intervenciones ejecutadas, no al objetivo
de extender GraphRouter a selección conjunta. El usuario señaló correctamente
que el control oficial solo usaba Direct y los pilotos conjuntos cambiaban el
encoder y el scorer. Por tanto, no se ha probado todavía la combinación del
mecanismo oficial de incorporación con routing prompt–modelo.

D-034 y [el contrato conjunto](graphrouter-joint-contract.md) restablecen esa
pregunta mediante una extensión mínima especificada: fuentes consulta–prompt,
un nodo compartido por LLM y encoder/decoder oficiales. Es un diseño pendiente
de implementación, no un nuevo resultado ni una reinterpretación positiva de
las intervenciones cerradas. No introduce otra regla externa de utilidad.
