# Incorporación episódica: un contraste emparejado 8→9

Fecha: 2026-11-23. D-041. Implementación lista y pruebas CPU aprobadas;
entrenamiento real pendiente en GPU0 de Morel. No se ejecutó un benchmark CPU
como sustituto del uso de GPU solicitado.

## Hipótesis única

Simular durante el entrenamiento la escasez de historia de un candidato ayuda
a puntuarlo frente a los conocidos cuando se incorpora un noveno modelo real.
Se compara un control con historia conocida disponible frente a episodios que
restringen la historia de un modelo pseudo-nuevo. No se entrena con el noveno.
No se exige reproducir un resultado positivo de GraphRouter para este contraste.

## Arquitectura que permanece fija

Se usa la extensión conjunta del último piloto oficial:
`router/graphrouter_official.py` + `router/graphrouter_joint.py`.
Fuentes consulta–prompt, un nodo compartido por LLM, FeatureAlign, dos
GeneralConv/BatchNorm de ancho16 (common_dim8), atributos [R,C,P] y decoder
sigmoid del producto interno medio. No se cambia scorer, agregación, embeddings,
normalización, arquitectura ni regla argmax. No es el antiguo SupportGraphRouter
ni una reproducción de Edge-GNN v1.

## Dos brazos, misma información y presupuesto

Los dos brazos se entrenan en un único proceso sobre GPU0 y parten de una copia
exacta de los mismos parámetros y buffers. Ambos usan los ocho modelos conocidos,
cuatro prompts, seed1, lambda0.9, AdamW lr0.0003/wd0.0001, BCE, 32 muestras
por paso, un paso por época, 1000 épocas. No hay barrido.

| Aspecto | Control | Episódico |
|---|---|---|
| Inicialización y arquitectura | Idénticas | Idénticas |
| Consultas objetivo y etiquetas por muestra | Idénticas | Idénticas |
| Resultados objetivo visibles al encoder | Ninguno | Ninguno |
| Historia de los siete candidatos ordinarios | Toda la permitida por la máscara | La misma |
| Historia del octavo candidato pseudo-nuevo | Toda la permitida por la máscara | Solo ocho consultas de soporte |
| Noveno modelo real en ajuste/validación | Excluido | Excluido |

En cada muestra se eligen aproximadamente la mitad de las 549 consultas de
ajuste como objetivos. Se ocultan **todas** sus aristas de resultados, agrupadas
sobre prompts y modelos, en ambos brazos. Sus embeddings de entrada permanecen;
no se eliminan features disponibles antes de ejecutar un LLM.

El candidato pseudo-nuevo rota entre los ocho conocidos: cuatro veces por
modelo y por época. Sus ocho consultas auxiliares se muestrean exclusivamente
del contexto de train no objetivo. Se conservan sus 8x4=32 interacciones y se
ocultan las demás; para los otros siete modelos no cambia ninguna arista.
El control y el episodio reciben las mismas etiquetas objetivo. Hay una positiva
por consulta original: mejor pareja entre las 32 acciones conocidas.

Los outcomes objetivo del pseudo-nuevo pueden usarse como etiquetas del loss:
es uno de los ocho modelos de entrenamiento. No entran al grafo visible.
Esto no es entrenamiento con el noveno modelo externo, cuyos outcomes permanecen
sin leer numéricamente hasta congelar los dos modelos.

### Por qué se vuelve a entrenar el control

El checkpoint histórico enmascaraba aristas individualmente y podía conservar
otras acciones de la misma consulta objetivo. Para cumplir la separación por
consulta, la máscara agrupada se aplica **a ambos brazos nuevos**. Así el único
tratamiento entre ellos es restringir la historia del pseudo-nuevo. El resultado
histórico queda como antecedente, no como comparador primario emparejado.
No se atribuye la diferencia contra el checkpoint antiguo solo al episodicismo.
No es un contraste de entrenar 8x4 frente a 9x4: ambos entrenamientos son 8x4.

## Datos y evaluación congelada

Mismo dataset completo de 797 consultas y manifiesto seed1: 549 de ajuste,
ocho reservadas para soporte externo, 119 validación y 121 test. Las ocho
reservadas aportan historia de modelos conocidos durante fit, pero no targets
supervisados, como en el piloto conjunto anterior. El modelo externo es
qwen2.5-14b; no aparece como pseudo-nuevo ni en ningún label/contexto de ajuste.

Coste primario: extremos de cost_proxy_money en acciones Direct de los 549 qids
de ajuste y modelos conocidos, aplicados sin clipping a todos los prompts.
R=P-0.9C. Qnorm almacenado es secundario. No se cambian unidades ni se estima
ahorro amortizado de calibración.

Checkpoint por brazo: máximo macro-F1 en validación con solo los ocho conocidos,
exactamente el mismo criterio. No se ajusta con validación del externo ni test.
Ambos checkpoints se cargan y congelan antes de leer sus outcomes externos.

Desde cada checkpoint se evalúan las mismas tres condiciones:

1. Solo conocidos: 8x4 acciones.
2. Descripción: 9x4, sin resultados históricos del nuevo.
3. Calibrado: 9x4, añadiendo las 32 interacciones del nuevo sobre los mismos
   ocho qids reservados. Ninguna actualización de pesos ni de BatchNorm.

Las 32 observaciones externas se reutilizan entre brazos: no hay otro presupuesto
para favorecer uno. Los outcomes de test solo evalúan decisiones ya puntuadas.
Los scores conocidos deben permanecer iguales al añadir el nodo nuevo.

## Controles y lectura predefinida

El mismo runner calcula los controles anteriores sin otra ejecución:
pareja conocida fija elegida solo en validación conocida, prompt nuevo fijo
seleccionado solo en soporte, y política fija adaptada que compara ambos en
ese soporte. No usa outcomes de validación del modelo externo ni test para
seleccionar estos controles.

Guardar y leer conjuntamente:

- R/P/C calibrados del episódico frente al control emparejado.
- Ganancia de añadir el externo frente a la base de cada brazo.
- Diferencia entre esas dos ganancias de incorporación (no confundirla con
  mejora de routing cerrado).
- Contrastes con controles fijos, márgenes del nuevo, oportunidades oráculo,
  selecciones beneficiosas/perjudiciales y concentración por pareja/tarea.

El runner marca una señal descriptiva favorable solo si el episódico mejora R
calibrado frente al control, tiene ganancia propia de incorporación positiva,
mejora esa ganancia frente al control y supera ambos controles fijos principales.
Los intervalos bootstrap pareados se reportan sin convertir una señal puntual
en significación general. La política constante se marca aparte y no acredita
adaptación por instancia aunque mejore una recompensa promedio.

Es un único fold Qwen previamente observado, con una semilla: piloto de
desarrollo. Interpretar este resultado antes de cualquier cambio posterior;
ningún resultado dispara automáticamente otro fit, variante o barrido.

## Integridad y pruebas

Ocho pruebas nuevas cubren planes reproducibles y roles balanceados, exclusión
de consultas objetivo del soporte/grafo, presupuesto exacto, masking antes del
encoder, inicialización común, dos épocas sintéticas completas con todos los
controles, congelación y ausencia de fallback GPU→CPU.

Una de esas pruebas altera a NaN TODOS los outcomes del
noveno modelo real y compara los dos ajustes antes de la evaluación: los hashes
de los checkpoints y del plan de episodios no cambian. Eso comprueba exclusión
numérica externa en la trayectoria de entrenamiento, no solo los nombres.
Pasaron también las diez pruebas conjuntas previas: 18 pruebas únicas de este
mecanismo y la regresión conjunta. No son resultados de rendimiento real.

La salida registra hashes iniciales/seleccionados, plan de episodios, ejemplos
con qids, conteos de roles, validación por brazo y los scores/decisiones de las
seis condiciones. No se sustituyen artefactos históricos.

## Una ejecución en Morel: GPU0 exclusivamente

```bash
source ~/repos/graph-router-2/.venv/bin/activate
cd ~/repos/ProMoRouter
git pull --ff-only
bash scripts/run_graphrouter_joint_episodic.sh
```

El runner fija CUDA_VISIBLE_DEVICES=0 y usa cuda:0; no toca GPU1 ni cambia a CPU
si CUDA no está disponible. Entrena los dos brazos emparejados en ese proceso.
No requiere nuevas dependencias respecto del piloto conjunto anterior.

Al terminar, devolver:
`outputs/graphrouter_joint_episodic/qwen2.5-14b/seed1_lam0.9/results.json`.
En la misma carpeta quedan control.pt y episodic.pt. El directorio existente
se rechaza para preservar resultados. Este entorno no tiene CUDA ni conexión
remota habilitada a Morel: aquí solo se ejecutaron verificaciones CPU.
