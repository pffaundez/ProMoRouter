# Comparación de generalización con seis modelos conocidos y tres nuevos

## Pregunta principal

¿Qué router obtiene mayor recompensa cuando incorpora tres LLM con el mismo conjunto de consultas de soporte? El contraste primario compara métodos en el pool ampliado. La diferencia frente al pool conocido de cada método se registra como diagnóstico secundario.

## Folds

| Fold | Modelos retenidos | Papel |
|---|---|---|
| A | qwen2.5-14b, mistral-7b, llama3.1-70b | Validación del pipeline. Qwen2.5-14B ya se usó en desarrollo. |
| B | qwen2.5-7b, yi-34b, mixtral-8x7b | Evaluación con el protocolo fijado tras A. |
| C | llama3.1-8b, codellama-34b, qwen2.5-72b | Evaluación con el protocolo fijado tras A. |

El complemento de cada trío constituye el conjunto conocido. Cada LLM se retiene una vez. Los folds comparten consultas, por lo que los resultados entre folds tienen dependencia. B y C amplían la cobertura de modelos retenidos sobre este benchmark.

## Particiones y recompensa

- Dataset completo de 797 consultas, manifiesto de seed 1 y los mismos ocho IDs de soporte de T-039. Compartir 549 consultas de ajuste, ocho de soporte, 119 de validación y 121 de test entre métodos y folds.
- La validación y el entrenamiento inicial solo usan resultados de los seis conocidos. Las ocho consultas de soporte quedan fuera de los targets de ajuste. Para los métodos inductivos conservan el papel de contexto conocido del piloto T-039.
- Cada fold ajusta una sola referencia monetaria con los extremos de las acciones Direct de los seis conocidos en las 549 consultas de ajuste. Todos los métodos aplican esa misma referencia a todas sus acciones, sin clipping. Recompensa primaria R = P − 0.9 C.
- En incorporación se revelan las 96 observaciones disponibles de los tres nuevos LLM en las ocho consultas y cuatro prompts. Direct consume las 24 observaciones de su espacio de acciones. El test aporta recompensas para evaluar las decisiones ya emitidas.

## Comparadores

| Método | Ajuste inicial | Incorporación | Acciones en test |
|---|---|---|---:|
| Router conjunto con encoder oficial | Brazo control del piloto T-039 adaptado a seis conocidos | Congelar pesos y buffers. Añadir los tres nodos y sus aristas de soporte. | 36 |
| GraphRouter Direct | Control basado en el encoder oficial, con seis LLM y Direct | Congelar pesos y buffers. Añadir tres nodos y las aristas Direct del soporte. | 9 |
| Flat-MLP conjunto con soporte | Flat-MLP de P0, ajustado a seis LLM y cuatro prompts | Comparador con actualización supervisada usando únicamente los resultados nuevos del soporte y la historia de ajuste conocida. Fijar el presupuesto de actualización antes de evaluar A. | 36 |
| Pareja fija adaptada | Elegir la mejor pareja conocida por recompensa en validación conocida | Compararla con las 12 parejas nuevas en las ocho consultas de soporte. Elegir una pareja para todo el test. En empate conservar la conocida. | 1 emitida |

Registrar también la pareja fija conocida. Para Flat-MLP, la selección de hiperparámetros y presupuesto de actualización debe quedar definida con candidatos conocidos. Su fila requiere indicar que actualiza los pesos. La fila conjunta con encoder oficial corresponde al candidato de adaptación de T-039. ProMoRouter v1 es el antecedente de pool cerrado.

## Tabla de resultados y salida del primer fold

- Por método y fold, reportar P, C y R en el pool ampliado, diferencias pareadas por consulta, selecciones por LLM y por pareja, observaciones de soporte consumidas, tiempo de ajuste inicial, tiempo de incorporación/actualización y tiempo de scoring. Los tiempos con logs precalculados excluyen la generación de las observaciones de soporte.
- El fold A valida la alineación de IDs, los pools y la escala de recompensa compartida, la exclusión de resultados retenidos del ajuste inicial y la invariancia del estado en los métodos congelados. Revisar comparaciones por consulta y guardar resultados junto con el manifiesto de ejecución.
- Los runners actuales aceptan un único modelo retenido. Implementar la entrada de tres IDs, unificar evaluadores y agregar la actualización del MLP antes de lanzar A. T-039 es un antecedente 8→9 y sus métricas no ocupan una fila de esta comparación 6→9.
- Ejecutar primero A en GPU 0 de Morel. Ejecutar B y C con el mismo contrato una vez validado el pipeline.
