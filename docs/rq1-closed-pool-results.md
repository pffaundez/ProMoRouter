# RQ1 · Resultados en pool conocido

Verifiqué las cinco corridas completas con 557 consultas de ajuste, 119 de validación y 121 de test por semilla. Recalculé P, C y R a partir de las decisiones guardadas. Los 28.692 proxies de coste coinciden con los totales de tokens registrados. Queda por verificar la cobertura de llamadas en las estrategias multillamada.

Los valores se reportan como media ± desviación estándar muestral entre cinco semillas para P, C y R. La desviación estándar describe la variación entre corridas. Los splits reutilizan el mismo corpus.

## λ = 0.1

| Método | P media ± DE | C media ± DE | R media ± DE |
|---|---:|---:|---:|
| ProMoRouter v2 · GNN conjunto | 0.43822 ± 0.03229 | 0.15901 ± 0.09619 | 0.42232 ± 0.02837 |
| GraphRouter Direct | 0.35265 ± 0.15110 | 0.23778 ± 0.05778 | 0.32887 ± 0.14546 |
| Flat-MLP conjunto | 0.56265 ± 0.05133 | 0.24855 ± 0.02924 | 0.53779 ± 0.05227 |
| Mejor pareja fija | 0.54541 ± 0.04092 | 0.25846 ± 0.01447 | 0.51957 ± 0.04123 |
| Mejor modelo fijo Direct | 0.48805 ± 0.05857 | 0.28844 ± 0.01128 | 0.45920 ± 0.05853 |
| Mayor LLM Direct | 0.36355 ± 0.04497 | 0.39758 ± 0.01934 | 0.32380 ± 0.04557 |
| Menor LLM Direct | 0.21036 ± 0.01547 | 0.05234 ± 0.00267 | 0.20512 ± 0.01566 |
| Oráculo conjunto | 0.76826 ± 0.04210 | 0.11745 ± 0.00836 | 0.75651 ± 0.04168 |
| Oráculo Direct | 0.64079 ± 0.05873 | 0.17254 ± 0.01389 | 0.62354 ± 0.05886 |

## λ = 0.5

| Método | P media ± DE | C media ± DE | R media ± DE |
|---|---:|---:|---:|
| ProMoRouter v2 · GNN conjunto | 0.42186 ± 0.02413 | 0.13774 ± 0.07168 | 0.35299 ± 0.04349 |
| GraphRouter Direct | 0.30444 ± 0.12337 | 0.16790 ± 0.07642 | 0.22049 ± 0.12132 |
| Flat-MLP conjunto | 0.51463 ± 0.05447 | 0.14674 ± 0.03833 | 0.44126 ± 0.05530 |
| Mejor pareja fija | 0.54541 ± 0.04092 | 0.25846 ± 0.01447 | 0.41618 ± 0.04295 |
| Mejor modelo fijo Direct | 0.43943 ± 0.09096 | 0.22112 ± 0.08871 | 0.32887 ± 0.06264 |
| Mayor LLM Direct | 0.36355 ± 0.04497 | 0.39758 ± 0.01934 | 0.16476 ± 0.04869 |
| Menor LLM Direct | 0.21036 ± 0.01547 | 0.05234 ± 0.00267 | 0.18419 ± 0.01643 |
| Oráculo conjunto | 0.76314 ± 0.04167 | 0.09925 ± 0.00762 | 0.71351 ± 0.04009 |
| Oráculo Direct | 0.63404 ± 0.05875 | 0.14917 ± 0.01066 | 0.55946 ± 0.05916 |

## λ = 0.9

| Método | P media ± DE | C media ± DE | R media ± DE |
|---|---:|---:|---:|
| ProMoRouter v2 · GNN conjunto | 0.37433 ± 0.08145 | 0.11676 ± 0.08018 | 0.26925 ± 0.06614 |
| GraphRouter Direct | 0.30581 ± 0.12310 | 0.16767 ± 0.08079 | 0.15491 ± 0.12971 |
| Flat-MLP conjunto | 0.47891 ± 0.06724 | 0.09440 ± 0.02208 | 0.39395 ± 0.04881 |
| Mejor pareja fija | 0.42961 ± 0.02253 | 0.07975 ± 0.00452 | 0.35783 ± 0.02173 |
| Mejor modelo fijo Direct | 0.37248 ± 0.07992 | 0.13139 ± 0.03587 | 0.25423 ± 0.05106 |
| Mayor LLM Direct | 0.36355 ± 0.04497 | 0.39758 ± 0.01934 | 0.00573 ± 0.05277 |
| Menor LLM Direct | 0.21036 ± 0.01547 | 0.05234 ± 0.00267 | 0.16325 ± 0.01724 |
| Oráculo conjunto | 0.75774 ± 0.04225 | 0.09165 ± 0.00792 | 0.67526 ± 0.03853 |
| Oráculo Direct | 0.62502 ± 0.06266 | 0.13705 ± 0.01314 | 0.50168 ± 0.05889 |

## Contrastes emparejados

Cada celda indica la diferencia media de recompensa y el número de semillas con diferencia estrictamente positiva. Los splits reutilizan el mismo corpus.

| Contraste | λ = 0.1 | λ = 0.5 | λ = 0.9 |
|---|---:|---:|---:|
| Flat-MLP frente a GraphRouter Direct | +0.20892 (5/5) | +0.22077 (5/5) | +0.23904 (5/5) |
| Flat-MLP frente a la mejor pareja fija | +0.01823 (4/5) | +0.02508 (4/5) | +0.03612 (4/5) |
| GNN conjunto frente a GraphRouter Direct | +0.09345 (2/5) | +0.13250 (5/5) | +0.11434 (5/5) |
| GNN conjunto frente a Flat-MLP | −0.11548 (0/5) | −0.08827 (0/5) | −0.12470 (0/5) |
| GNN conjunto frente a la mejor pareja fija | −0.09725 (0/5) | −0.06319 (0/5) | −0.08858 (0/5) |
| Oráculo conjunto frente a oráculo Direct | +0.13298 (5/5) | +0.15405 (5/5) | +0.17358 (5/5) |

Flat-MLP conjunto obtiene la mayor recompensa media entre los métodos evaluables en los tres regímenes. El GNN conjunto selecciona de una a tres parejas por semilla y régimen. ProMoRouter v2 y GraphRouter Direct disponen de 36 y 9 acciones, respectivamente. La comparación con Flat-MLP usa las mismas 36 acciones. Los oráculos cuantifican el techo de los dos espacios de acciones.
