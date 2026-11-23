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
