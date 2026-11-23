# Dictamen de fidelidad de GraphRouter

Fecha: 2026-11-23. Referencia examinada: commit ProMoRouter
`06dbb6bb9b38b6cb8a257fc56aa6db4015a32076`, con fuentes de GraphRouter fijadas
en `0edfb8758d45acd93c8029354867ffb49bb8a97f`.

**Dictamen: reproducción parcial.** Se verifica la correspondencia de los
bloques auditados con el código público y la coherencia del resultado CPU.
No se verifica una reproducción integral del experimento few-shot original ni
su política por consulta. No se repitió el entrenamiento del benchmark ni se
modificó ProMoRouter para perseguir el resultado.

## 1. Evidencia ejecutable

Se ejecutó el método **original** `GNN_prediction.train_validate` de la fuente
pública fijada, no otra implementación manual de su loop. Se comparó con nuestra
referencia en un fixture pequeño: una época y tres máscaras, mismos pesos
iniciales, datos y RNG. Es una prueba de ingeniería, no otro experimento de
rendimiento. La fuente original se conserva sin cambios como fixture MIT.

| Comprobación | Resultado |
|---|---|
| AST de FeatureAlign y EncoderDecoderNet | Idénticos al upstream |
| Máscaras del entrenamiento | Iguales |
| Scores de entrenamiento | Error máximo 0 |
| Parámetros y buffers BatchNorm tras el paso | Error máximo 0 |
| Scores de validación | Error máximo 0 |
| Scores de test con el orden de aristas del adaptador | Error máximo 1.1920929e-7 |
| Scores de test conservando el orden de aristas del upstream | Error máximo 0 |

La diferencia pequeña de test procede del orden de acumulación de mensajes:
el adaptador concatena train+val, mientras el original enmascara las aristas
en el orden del CSV. Ordenarlas como el original elimina esa diferencia en el
fixture. No se modifica la ejecución primaria para elegir otra política.
Estas comprobaciones cubren el fixture y el runtime registrado, no garantizan
igualdad bit a bit en todos los dispositivos, versiones o 1000 pasos.

También se verificó de nuevo el resultado guardado contra el CSV publicado:
identidad de fuente, split, orden de consultas, P/C/R, promedios y argmax de
cada condición. Error máximo de las métricas comprobadas: **0**. El resultado
canónico no se reescribió; conserva SHA-256
`5e11700e3c41df0752e3dba2efa46a1f05b5d43e1f62f4a82d8f95d1ee1665cc`.

## 2. Hallazgos y correcciones

### Máscaras en una futura ejecución GPU

El original usa `torch.rand(mask.size())` con el dispositivo predeterminado CPU.
Nuestro runner generaba los números en `eligible.device`, cambiando al RNG CUDA
cuando se elegía GPU. Se corrigió para generar siempre la máscara en CPU y
transferirla después al dispositivo de cálculo.

**La ejecución canónica fue CPU:** su secuencia ya era la del original. La
corrección no cambia su resultado ni requiere otro fit. No se afirma haber
validado numéricamente una repetición GPU; se corrige el origen del RNG.

### Guardado y evaluación del checkpoint

La fuente **sí actualiza best_f1** y guarda el mejor estado conocido. El método
`test()` tiene comentada la carga del checkpoint y evalúa el estado actual.
Nuestra referencia evalúa el estado final y conserva aparte el mejor por F1;
eso coincide con la semántica pública usada como primaria.

Durante la revisión hubo una sospecha manual de que best_f1 no se actualizaba.
El análisis AST y la línea real de actualización la refutaron. Se corrigió esa
lectura; no se atribuye al código un defecto de guardado inexistente.

## 3. Qué es verificable y qué está reconstruido

| Elemento | Estado |
|---|---|
| Código fijado, embeddings y CSV público | Verificados por hashes |
| Orden de modelos y campo vacío del cuarto modelo | Resuelto según asignación posicional pública, registrado |
| Split público contiguo por tarea y proporciones 70/10/20 | Verificado frente al código |
| Encoder, decoder, atributos, BCE, optimizador y máscaras | Verificados en los bloques/fixture descritos |
| Visibilidad de features de todas las consultas en BatchNorm | Conservada como en código, declarada |
| Historia conocida train+val para test | Conservada como en código |
| Exclusión de cuatro modelos y adición de 80 ejemplos | Reconstrucción conforme a sección3.2; no hay driver few-shot explícito en el release auditado |
| Índices exactos de las 80 consultas originales | No identificados en el release |
| Checkpoint/configuración/predicciones específicos de Tabla5 | No identificados como un conjunto verificable |
| Runtime original completo | No identificado; requirements fija Torch2.1.0 y no fija PyG |
| Correspondencia paper/código sobre arquitectura y optimizador | Hay discrepancias publicadas, sin resolver |

Hay checkpoints públicos, pero no se ha establecido cuál corresponde a Tabla5,
a qué configuración/partición se asocia ni sus decisiones originales. No se
los prueba selectivamente hasta obtener la cifra deseada.

El paper describe GAT/ancho32 y Adam con schedule; el código fijado emplea
GeneralConv/ancho16 y AdamW constante. Los datos publicados contienen 60
bloques de consulta y 600 interacciones por tarea; la Tabla2 dice 600 Cases.
Esa unidad continúa ambigua. No se cambia arquitectura, normalización ni datos
para resolver unilateralmente esas discrepancias.

## 4. Resultado que podemos sostener

La ejecución canónica mantiene R=0.20734849 en las tres condiciones y elige
LLaMA-3 (8b), conocido, en 48/48 consultas. Coincide con el 0.207 redondeado de
Tabla5, pero el soporte no cambia decisiones. Sí modifica scores nuevos
(máximo delta 0.33966564); existen 14 oportunidades oráculo nuevas.

Por tanto:

- Hay concordancia escalar a tres decimales y una implementación de referencia
  parcialmente contrastada con el código público.
- No hay evidencia de incorporación útil incremental en nuestra ejecución.
- No podemos inferir que las predicciones originales de los autores fueran
  iguales, ni concluir que GraphRouter no se adapta en general.
- La Tabla5 compara recompensa entre métodos; no reporta directamente nuestro
  contraste marginal con/sin nuevos modelos. Esas afirmaciones son distintas.

## 5. Información externa mínima para resolver lo pendiente

Solicitud técnica que puede remitirse a los autores, **no enviada**:

> Para reproducir Tabla5, necesitamos el commit/entrypoint few-shot exacto,
> IDs de las 80 consultas auxiliares y estado/semilla del muestreo; configuración
> de capas, ancho, optimizador y versiones Torch/PyG; construcción del grafo
> durante entrenamiento y evaluación, incluido si los nodos nuevos ya estaban
> presentes sin aristas; checkpoint asociado y predicciones por consulta. También
> agradeceríamos confirmar qué unidad representan los 600 Cases por tarea.

Son datos para cerrar una reproducción, no una nueva búsqueda de modelos o
hiperparámetros. Esta revisión no contacta a terceros ni autoriza mensajes.

## 6. Cierre y artefactos

Auditoría completada, estado **partial_reproduction**. No se encontró un error
que invalide el resultado CPU primario en los bloques comprobados. No hay
motivo verificado para repetir las 1000 épocas bajo la regla acordada.
ProMoRouter sigue pausado y no se dispara otra variante.

- Código: `audit_graphrouter_fidelity.py`.
- Evidencia: `analysis/graphrouter_reference/fidelity_audit_final.json`.
- Fixture: `tests/fixtures/graphrouter_upstream_graph_nn.py`, fuente pública
  sin cambios, con atribución y licencia MIT.
- Pruebas: 18 de referencia/auditoría/control Direct pasan en CPU; cuatro son
  nuevas de auditoría y diez corresponden a la referencia. No se presenta esta
  corrida selectiva como repetición de toda la suite histórica de 52 pruebas.
- Runtime de auditoría: Torch2.15.0+cpu, PyG2.8.1, scikit-learn1.8.0. El archivo
  opcional requirements-graphrouter-fidelity.txt usa la dependencia métrica
  scikit_learn1.6.1 de los autores; no se cambió el entorno auditado.

Para repetir la auditoría deliberadamente, sin un fit de benchmark:

```bash
python prepare_graphrouter_reference.py
python audit_graphrouter_fidelity.py
```

Requiere las dependencias ya usadas y scikit-learn para ejecutar el loop original.
Si falta, instalar requirements-graphrouter-fidelity.txt en el entorno elegido.
La evidencia repetida se escribe en outputs/graphrouter_reference/fidelity_audit.json
sin sobrescribir el informe canónico. El dataset y las fuentes se descargan de
ubicaciones públicas fijadas/verificadas. No hay llamadas a LLMs ni búsquedas.
