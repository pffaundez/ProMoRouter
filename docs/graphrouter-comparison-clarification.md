# Aclaración: qué comparaciones con GraphRouter están respaldadas

Fecha de revisión: 2026-11-23. Esta nota corrige la interpretación de los nombres
de baselines, sin cambiar números históricos ni abrir otra ejecución.

## Hallazgo verificable en el código

El agregador `analysis/aggregate_p0_closed_pool_results.py` etiqueta como
GraphRouter-direct los resultados de `outputs/p0_router_model_only_direct_qnorm`,
archivos `router_model_only_direct_qnorm_results_seed{seed}.json`.
El entrenador `train_router_model_only_direct_qnorm.py` implementa
`DirectModelOnlyRouter`: concatena embeddings de consulta y modelo y aplica un
MLP con Linear/LayerNorm/GELU/Dropout. No usa el encoder oficial GraphRouter ni
message passing de una GNN.

Por tanto, al referirnos a esa comparación P0, el nombre descriptivo correcto es
**MLP de selección de modelo con Direct fijo (etiqueta histórica GraphRouter-direct)**.
La superioridad P0 de ProMoRouter/Edge-GNN frente a ese baseline NO demuestra
superioridad frente al GraphRouter oficial del paper. El baseline continúa
siendo una comparación válida con un router model-only; su atribución al método
publicado es la que debe corregirse. No se renombra aquí toda la cadena de
artefactos ni se invalidan las métricas verificadas.

Fuentes del hallazgo: los dos archivos anteriores, revisados en la rama
v2-behavior-profiles (head 203a58cd729f1b073f28dd6d23e0688d6c5b4102).

## Tres comparaciones distintas

| Familia | Arquitectura / escenario | Qué evidencia aporta |
|---|---|---|
| P0 histórico | ProMoRouter frente a MLP model-only; pool conocido | Comparación cerrada; no prueba adaptación ni superioridad sobre GraphRouter oficial |
| Control oficial Direct reciente | Encoder y decoder oficiales, bucle/protocolo adaptados; modelo retenido | Incorporación técnica en nuestro benchmark; no mejora agregada demostrada en ese fold |
| Extensión conjunta reciente | Encoder oficial con fuentes consulta–prompt y más acciones/historia | Mejora vs su base, concentración total y ausencia de ventaja frente al control fijo conocido |

La extensión reciente tampoco es el ProMoRouter v1 original al que se añade
solamente un modelo: cambia la representación y el entrenamiento. No se puede
encadenar la ventaja P0 con la adaptación del paper como si fueran propiedades
medidas de una única implementación bajo condiciones constantes.

## Qué acredita el paper

Referencia primaria: GraphRouter, ICLR 2025,
https://proceedings.iclr.cc/paper_files/paper/2025/file/41b6674c28a9b93ec8d22a53ca25bc3b-Paper-Conference.pdf

El apartado de nuevos LLM usa ejemplos auxiliares obtenidos de consultas de
entrenamiento, excluidos del entrenamiento del router y utilizados al evaluar.
La Tabla 5 compara GraphRouter few-shot con otros métodos y con una variante
entrenada. Eso respalda sus resultados bajo su protocolo; no demuestra que
agregar cualquier candidato siempre mejore cualquier base existente, ni es
idéntico a nuestro contraste marginal contra una política fija seleccionada
en validación conocida.

## Flag pendiente de explicación, sin diagnóstico causal

En la extensión conjunta, los scores conocidos quedan aproximadamente entre
0.00065 y 0.00279, y los de Qwen calibrado entre 0.48019 y 0.48180. El nuevo
modelo gana en todas las consultas. Este desfase y la concentración requieren
prudencia al atribuir el resultado a incapacidad de generalización. No se ha
demostrado su causa ni que sea un bug, una propiedad del método, o un efecto de
la extensión/datos/entrenamiento. Copiar el encoder no reproduce por sí solo
toda la metodología experimental del paper.

## Consecuencia para reportar

Se mantienen el cierre del piloto y sus resultados. Se retira la interpretación
amplia de que la comparación P0 demostraba que ProMoRouter v1 superaba al
GraphRouter oficial. Tampoco se concluye que el método publicado no se adapta.
Son cuestiones de atribución y comparabilidad, no evidencia nueva de desempeño.
No hay nuevo entrenamiento, diagnóstico causal resuelto ni variante autorizada
por esta aclaración.
