# Comprobación final y cierre del piloto conjunto

Fecha: 2026-11-23. T-036. El objetivo es cerrar la interpretación del piloto,
no añadir otra familia experimental. El entrenamiento y el resultado permanecen
intactos. No hay inferencia, GPU, nuevo checkpoint ni dependencias de PyTorch.

## Una auditoría de artefactos existentes

`audit_graphrouter_joint_fixed.py` usa la biblioteca estándar y el dataset
original en Morel, cuyo hash debe coincidir con results.json. Evalúa tres
políticas constantes, elegidas antes de consultar los outcomes de test:

1. **fixed_known:** mejor pareja del pool conocido por recompensa media en
   validación conocida. Nunca usa outcomes de validación del modelo retenido.
2. **fixed_new_support:** mejor prompt Qwen por recompensa media en las ocho
   consultas de soporte (ya disponibles), sin usar validación Qwen ni test.
3. **fixed_adapted:** compara fixed_known y fixed_new_support en las mismas
   ocho consultas de soporte; elige una sola pareja para todas las consultas.
   En empate conserva la conocida. No añade llamadas de calibración.

Comparaciones principales: GNN calibrada menos fixed_known y menos fixed_adapted.
fixed_new_support es una lectura secundaria. No escoger el control más débil
según test ni elegir retrospectivamente la pareja fija de mayor R_test.
Se conserva la referencia de coste del resultado original. El resultado de
selección se registra mediante un hash; cambiar test no puede cambiarlo.

## Regla de cierre, sin búsqueda posterior

- Si alguna de las dos ganancias principales no es positiva: no hay ventaja
  frente a ambos controles fijos. Cerrar este piloto sin otra variante.
- Si ambas son positivas pero algún intervalo incluye cero: mejora puntual
  incierta. Cerrar y reportar esa incertidumbre; no lanzar un barrido automático.
- Si ambas son positivas y sus límites inferiores son mayores que cero:
  registrar ventaja condicional en este fold de desarrollo. Tampoco demuestra
  routing por instancia: las 121 decisiones actuales son la misma pareja.

En todos los casos se cierra la etapa actual. La hipótesis amplia del paper no
se declara resuelta por una política constante. Cualquier continuación futura
requiere un objetivo nuevo explícito; no forma parte de esta auditoría.
Los comparadores son posteriores a la observación del piloto; los intervalos
son condicionados al checkpoint y no constituyen confirmación independiente.

## Ejecutar una sola vez, sin GPU

```bash
cd ~/repos/ProMoRouter
git pull --ff-only
bash scripts/audit_graphrouter_joint_fixed.sh
```

Usa results.json en la ruta por defecto del piloto. Produce al lado:
`outputs/graphrouter_joint_unseen/qwen2.5-14b/seed1_lam0.9/fixed_policy_audit.json`.
No cambia resultados originales y rechaza sobrescribir la auditoría.
Enviar ese archivo para registrar el cierre; no ejecutar de nuevo el entrenamiento.

Siete pruebas CPU, sin PyTorch, comprueban separación de información, desempate,
orden de outcomes/qids, independencia de selección respecto a test, integridad
de particiones y clasificación de cierre. La auditoría real está pendiente en
Morel porque el dataset original de validación no está adjunto aquí.


## Resultado final y cierre efectivo — 2026-11-23

Auditoría real recibida: fixed_policy_audit.json. Fuente y dataset coinciden con
el resultado conjunto; se recalcularon las ganancias pareadas y el álgebra R.

| Política | Selección | P | C primario | R primario |
|---|---|---:|---:|---:|
| GNN base conocida | decompose/Qwen-7B o Llama-8B | 0.34977802 | 0.08056349 | 0.27727088 |
| GNN calibrada | decompose/Qwen-14B siempre | 0.34419288 | 0.04128061 | 0.30704034 |
| fixed_known | selfcheck/Qwen-7B siempre | 0.41206791 | 0.07723234 | **0.34255881** |
| fixed_new_support | CoT/Qwen-14B siempre | 0.30190279 | 0.06573037 | 0.24274546 |
| fixed_adapted | CoT/Qwen-14B siempre | 0.30190279 | 0.06573037 | 0.24274546 |

La pareja fija conocida fue elegida en validación conocida (R_val=0.33557548).
En soporte, esa pareja obtiene 0.47191818 y el mejor prompt Qwen, CoT,
0.49904168. Por ello la regla adaptada elige Qwen/CoT. El test no elige las
políticas y no se utiliza validación del modelo retenido.

Ganancia GNN calibrada menos fixed_known: **-0.03551847**; intervalo bootstrap
condicionado al checkpoint **[-0.10844570, 0.03267945]**. Hay 79 diferencias
positivas y 42 negativas; las negativas tienen mayor magnitud agregada.
No es evidencia de inferioridad estadísticamente establecida.

Ganancia menos fixed_adapted (también fixed_new_support): **+0.06429488**;
intervalo **[-0.00222867, 0.13043471]**, 58 beneficios, 22 perjuicios y 41 empates.
Tampoco establece una ventaja independiente con este único fold.

Veredicto predefinido: **no_advantage_over_both_fixed_controls**.
El router recupera +0.02976946 frente a su propia base, pero esa base queda
0.06528793 por debajo de la pareja fija conocida. Por ello recuperar parte de
esa diferencia no demuestra una ventaja del router frente al control sencillo.
Las 121 decisiones calibradas siguen siendo una sola pareja; no hay evidencia
de routing adaptativo por consulta.

**T-036 completada. Etapa cerrada. No quedan entrenamientos, auditorías,
variantes ni barridos pendientes de esta etapa.** La incorporación técnica y
su mejora descriptiva frente a la base se conservan como resultados; no se
presentan como validación de la hipótesis amplia. La causa de la concentración
y del desfase de scores no quedó identificada. Otra etapa requerirá un objetivo
nuevo explícito, sin reabrir automáticamente las intervenciones anteriores.

Texto reutilizable:

> La extensión de GraphRouter incorporó el modelo retenido con parámetros
> congelados y mejoró su propia base en 0.02977 de recompensa. Sin embargo,
> seleccionó la misma pareja prompt–modelo en todas las consultas y no superó
> una política fija elegida exclusivamente en validación conocida
> (0.30704 frente a 0.34256). El control final no respalda una ventaja del
> routing adaptativo bajo este protocolo de desarrollo. Los resultados se
> reportan con su incertidumbre y no establecen una imposibilidad general de
> adaptación a nuevos modelos.

Trazabilidad:
- Auditoría en Morel: outputs/graphrouter_joint_unseen/qwen2.5-14b/seed1_lam0.9/fixed_policy_audit.json.
- SHA-256 auditoría: 802c4ab02141a55ddd44cca96336d2ab39141288651a8358e8b1f74887dd83b1.
- SHA-256 resultado fuente: 54b96da267ee6dc7e4fdc5368b086555ab56e5799d2b950477180a4c37f219cc.
- Hash políticas congeladas: cbdd6976ad1f2959abeac83b7631ec6a9360c4ba692763e3dd45b51923549dd7.
