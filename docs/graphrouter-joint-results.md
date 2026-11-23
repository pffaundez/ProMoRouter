# Resultado del piloto conjunto con GraphRouter oficial

Fecha de revisión: 2026-11-23. Implementación: commit
`78e3314f38c9e5c9fd218e2976da03ddfb2d6888`. Contrato: D-034 y
[graphrouter-joint-contract.md](graphrouter-joint-contract.md).

## Resultado principal

La extensión mínima oficial sí obtuvo una mejora descriptiva de incorporación:
R calibrado menos R de la base conjunta = **+0.0297694573**. Sin embargo, la
política calibrada seleccionó **decompose + Qwen-14B en las 121 consultas**.
Esto demuestra incorporación y mejora agregada en este checkpoint/fold, pero
no ventaja de routing adaptativo por instancia. El intervalo bootstrap pareado
condicionado al checkpoint [-0.04242451, 0.10897677] incluye cero.

## Protocolo y verificaciones

Protocolo registrado: official_graphrouter_joint_unseen_v1. Qwen2.5-14B excluido,
seed1, lambda0.9, cuatro prompts, 1000 épocas y GPU0. El checkpoint elegido fue
época 44: macro-F1 de validación 0.01758658, R validación 0.30288675. La métrica
de selección no debe confundirse con la recompensa ni describirse como alta
exactitud de selección. Runtime: Torch 2.14.0+cu130 y PyG 2.8.1.

Particiones disjuntas verificadas: 549 ajuste, ocho soporte, 119 validación,
121 test. La referencia Direct histórica figura como verified: coinciden
población/splits/soporte/modelos, semilla/lambda y normalización primaria.
Esto alinea unidades, no capacidad, datos históricos o presupuesto de soporte.

Coste primario: cost_proxy_money normalizado con extremos de acciones Direct
conocidas de ajuste, mínimo 7.6 y máximo 502.8, sin clipping. Tres costes de
acciones de test superan 1 (máximo 1.00726974), de forma permitida por el contrato.
El coste qnorm guardado sigue siendo una salida secundaria separada.

El grafo tiene 2228 fuentes históricas consulta–prompt. Base y descripción tienen
17824 aristas visibles conocidas. Calibración añade exactamente 32 aristas
Qwen (ocho consultas por cuatro prompts), total 17856. Hay ocho/nueve nodos LLM
según condición. Los scores conocidos cambiaron exactamente **cero** entre las
tres condiciones, verificado desde candidate_scores. Las verificaciones de
congelación de pesos/BN del programa se ejecutaron antes de escribir el resultado;
el JSON por sí solo no permite una auditoría independiente de todos los pesos.

## R/P/C y efecto del soporte

| Condición | P | C primario | R primario | Ganancia vs base |
|---|---:|---:|---:|---:|
| Solo conocidos | 0.34977802 | 0.08056349 | 0.27727088 | 0 |
| Descripción Qwen | 0.26279095 | 0.08044757 | 0.19038813 | -0.08688275 |
| Qwen con soporte k8 | 0.34419288 | 0.04128061 | 0.30704034 | +0.02976946 |

Calibración menos descripción: **+0.11665220**. En comparación con la base,
P cambia -0.00558514 y C cambia -0.03928288 (reducción relativa del C normalizado
48.76%). La mejora de R procede del menor coste, no de mayor P promedio.
Este porcentaje no es ahorro monetario neto de despliegue ni amortiza soporte.

R_qnorm secundario: base 0.23930008, descripción 0.14236013, calibrado 0.31080880.
No mezclar estos valores con R primario ni presentar una comparación numérica
simple con pilotos anteriores que optimizaban otra referencia de coste.

## Selecciones y concentración

| Condición | Parejas seleccionadas |
|---|---|
| Base | decompose/Qwen-7B: 106; decompose/Llama-8B: 15 |
| Descripción | direct/Qwen-14B: 121 |
| Calibrado | decompose/Qwen-14B: 121 |

El soporte cambia la decisión observable de prompt de Direct a decompose en
todas las consultas, sin cambiar el modelo seleccionado. La política calibrada
es exactamente equivalente, en este test, a usar siempre Qwen-14B/decompose.
No es evidencia de que la mejor pareja se identifique individualmente.

Hay 81 cambios beneficiosos y 40 perjudiciales frente a la política base.
Oportunidades oráculo de ampliar el pool: 46; se selecciona el nuevo modelo en
todas y se elige una pareja nueva óptima en 29. Esos 46 casos comparan contra
la mejor acción conocida oráculo; los 81 beneficios comparan contra la acción
real de la base, que puede ser subóptima. Por ello tampoco es contradictorio
que la ganancia vs base (+0.02977) supere la ganancia de expansión oráculo
(+0.01610833): sus referencias son distintas.

| Tarea | Ganancia R calibrado vs base |
|---|---:|
| Alpaca | +0.01450320 |
| GSM8K | +0.14103681 |
| HotpotQA | -0.02521033 |
| SQuAD | -0.00280028 |

La mejora agregada no es uniforme por tarea; GSM8K aporta la mayor diferencia.

## Flag de comparabilidad de scores

Rango de scores conocidos: [0.00064843, 0.00279123], idéntico en todas las
condiciones. Rango Qwen sin soporte: [0.50234407, 0.50256163]. Con soporte:
[0.48018536, 0.48179933]. Todos los márgenes Qwen vs conocidos son positivos.

El candidato nuevo domina el ranking por un amplio desfase de scores tanto
antes como después de recibir soporte. Los scores son compatibilidades, no R
predicho ni probabilidades calibradas de utilidad. Esta diferencia de escala
es un flag observado, no una causa demostrada. No se ha identificado si su
origen es la cantidad de historia, el estado del nodo, BatchNorm, optimización
u otro componente; no se adopta otra modificación para explicarlo aquí.

## Comparación sencilla disponible sin entrenamiento adicional

Lectura exploratoria posterior al piloto: usar únicamente las ocho consultas
de soporte Qwen para elegir su prompt por R medio daría:

| Prompt | R medio en soporte Qwen |
|---|---:|
| Direct | 0.39496962 |
| CoT | 0.49904168 |
| Decompose | 0.37911084 |
| Selfcheck | 0.43717999 |

Esa regla elige CoT sin consultar test y, al evaluarla, obtiene R_test=0.24274546.
No reproduce la elección decompose del GNN. La política calibrada la supera
en este test en +0.06429488. Es un comparador exploratorio, no prerregistrado,
y no demuestra que el grafo aporte valor frente a todas las políticas fijas.
En particular, falta la pareja fija del pool conocido elegida SOLO con validación conocida.
El modelo retenido no puede aportar outcomes de validación para esta selección;
sus prompts solo pueden elegirse con el soporte permitido. No se elige
retrospectivamente la mejor pareja usando test.

## Cierre de lectura y siguiente comprobación acotada

T-035 se completa como piloto con señal positiva descriptiva, incertidumbre y
concentración total. No se descarta este mecanismo ni se da por resuelta la
hipótesis del paper. Antes de más entrenamiento, la comprobación pertinente es
comparar con una pareja fija seleccionada usando únicamente información
permitida de validación/soporte, explicitando qué candidatos vio cada selector.
Esto debe hacerse con los artefactos existentes y sin reentrenar la GNN.
No se programa un nuevo encoder, umbral, k80 ni barrido de semillas a raíz de
este resultado.

## Trazabilidad

Artefacto Morel:
`outputs/graphrouter_joint_unseen/qwen2.5-14b/seed1_lam0.9/results.json`.
Adjunto analizado: results(4).json; SHA-256:
`54b96da267ee6dc7e4fdc5368b086555ab56e5799d2b950477180a4c37f219cc`.
Los cálculos de esta nota usan results, candidate_scores, support_outcomes y
target_outcomes del JSON. Se verificó álgebra R=P-0.9C, disjunción de qids,
selecciones, valores del comparador de soporte y scores conocidos invariables.
La revisión no ejecutó nuevos entrenamientos ni modificó el artefacto original.


## Final comparator result — 2026-11-23

The pending comparison has completed. A known-pool fixed pair selected only on
known validation, selfcheck/Qwen-7B, obtains primary R=0.34255881, exceeding the
calibrated GNN's 0.30704034 by 0.03551847. The GNN-minus-control conditional
interval [-0.10844570,0.03267945] includes zero. The support-selected adapted
fixed policy Qwen/CoT obtains R=0.24274546; the GNN beats that point estimate
but does not beat both main fixed controls. All GNN calibrated choices remain
Qwen/decompose. D-037 closes this stage without another experiment.
See [graphrouter-joint-closure.md](graphrouter-joint-closure.md) for the final
calculation record and reporting limits; earlier pending-control text above
is superseded by this result.
