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
| Flat-MLP conjunto con soporte | Flat-MLP de P0, ajustado a seis LLM y cuatro prompts | Comparador con actualización supervisada usando únicamente los resultados nuevos del soporte y la historia de ajuste conocida. Una época de actualización con replay de las 549 consultas conocidas y las ocho de soporte. | 36 |
| Pareja fija adaptada | Elegir la mejor pareja conocida por recompensa en validación conocida | Compararla con las 12 parejas nuevas en las ocho consultas de soporte. Elegir una pareja para todo el test. En empate conservar la conocida. | 1 emitida |

Registrar también la pareja fija conocida. Para Flat-MLP, la selección de hiperparámetros y presupuesto de actualización debe quedar definida con candidatos conocidos. Su fila requiere indicar que actualiza los pesos. La fila conjunta con encoder oficial corresponde al candidato de adaptación de T-039. ProMoRouter v1 es el antecedente de pool cerrado.

## Tabla de resultados y salida del primer fold

- Por método y fold, reportar P, C y R en el pool ampliado, diferencias pareadas por consulta, selecciones por LLM y por pareja, observaciones de soporte consumidas, tiempo de ajuste inicial, tiempo de incorporación/actualización y tiempo de scoring. Los tiempos con logs precalculados excluyen la generación de las observaciones de soporte.
- El fold A valida la alineación de IDs, los pools y la escala de recompensa compartida, la exclusión de resultados retenidos del ajuste inicial y la invariancia del estado en los métodos congelados. Revisar comparaciones por consulta y guardar resultados junto con el manifiesto de ejecución.
- El runner compartido ejecuta la comparación 6→9. T-039 queda como antecedente 8→9.
- Ejecutar primero A en GPU 0 de Morel. Ejecutar B y C con el mismo contrato una vez validado el pipeline.

## Ejecución en Morel

El ajuste de los dos grafos usa 1000 épocas, 32 máscaras por paso y selección
por macro-F1 en validación conocida. Ambos agrupan las máscaras por consulta.
El baseline Direct reutiliza el encoder oficial con este protocolo común.
El MLP usa hasta 80 épocas de ajuste inicial con selección por recompensa en
validación conocida y una época de actualización de 557 pasos. Para los grafos,
el tiempo de incorporación mide la preparación del contexto. Construcción del
grafo de inferencia y forward se cuentan en el tiempo de scoring.

```bash
cd ~/repos/ProMoRouter
git fetch origin experiment/unseen-6x3
git switch experiment/unseen-6x3
git pull --ff-only origin experiment/unseen-6x3
source ~/repos/graph-router-2/.venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python run_unseen_6x3.py \
  --fold A --data-root ~/repos/graph-router-2/data \
  --split-manifest data/router/splits/qnorm_complete_seed1.json \
  --output outputs/unseen_6x3/fold_A/results.json
```

El manifiesto debe ser el archivo existente de T-039. Si se guardó en otro
directorio, pasar esa ruta mediante `--split-manifest`.

La prueba sintética CPU del fold A ejecutó los cinco métodos y comprobó las
96 y 24 observaciones de soporte, el orden de consultas, R = P − 0.9 C y la
invariancia de pesos, buffers y scores conocidos al añadir candidatos en los
grafos. Sustituir por NaN los resultados de los retenidos en ajuste/validación
y alterar los resultados de test dejó idénticos todos los scores. La ejecución
completa en GPU de Morel sigue pendiente.
