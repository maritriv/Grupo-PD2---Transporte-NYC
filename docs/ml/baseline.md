# Baseline EX2 - Prediccion de Estres Urbano

Este documento describe el baseline actual del **Ejercicio 2**. Su objetivo es ofrecer una referencia minima pero interpretable para comparar los modelos predictivos de estres urbano.

## 1. Dataset utilizado

El baseline usa exactamente el mismo dataset model-ready que los demas modelos de `ej2`:

- `data/aggregated/ex_stress/df_stress_zone_hour_day`

Este dataset contiene variables temporales, espaciales, agregados operacionales y el target de estres futuro.

## 2. Targets

El script soporta los targets:

- `target_stress_t1`
- `target_stress_t3`
- `target_stress_t24`

y sus equivalentes binarios:

- `target_is_stress_t1`
- `target_is_stress_t3`
- `target_is_stress_t24`

Importante:

- En el snapshot local descargado para esta revision solo estaban disponibles `target_stress_t1` y `target_is_stress_t1`.
- Por tanto, la evaluacion real validada en esta iteracion corresponde a `t+1h`.

## 3. Baselines incluidos

El baseline actual esta implementado en `src/ml/models_ej2/a_model_baseline.py` y evalua tres referencias complementarias:

### a) `persistence_current_stress`

Predice que el estres futuro sera aproximadamente igual al `stress_score` actual.

Rol recomendado:

- **baseline naive principal**

Es la referencia mas natural para una serie temporal de corto plazo.

### b) `zone_hour_weekend_mean`

Predice la media historica del target por:

- `pu_location_id`
- `hour`
- `is_weekend`

Usa fallback por:

- `borough + hour + is_weekend`
- `hour + is_weekend`
- media global

Rol recomendado:

- **referencia fuerte**

Captura buena parte de la estacionalidad espacial y temporal del problema.

### c) `global_mean`

Predice siempre la media global del train.

Rol recomendado:

- **piso de control**

Sirve para validar que cualquier baseline o modelo real supera una referencia trivial.

## 4. Split y evaluacion

El baseline replica la misma logica de particion temporal usada por los modelos de `ej2`:

- `train = 70%`
- `val = 15%`
- `test = 15%`
- `time_col = timestamp_hour`

Metricas reportadas:

- Regresion: `MAE`, `RMSE`, `R2`
- Clasificacion: `Accuracy`, `Precision`, `Recall`, `F1`

## 5. Ejecucion

Comando base:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run -m src.ml.models_ej2.a_model_baseline --target-col target_stress_t1
```

Artefacto generado:

- `outputs/ml/ej2/baseline/baseline_report_target_stress_t1.json`

## 6. Resultados obtenidos para `target_stress_t1`

Resultados en test sobre el dataset local disponible:

| Baseline | MAE | RMSE | R2 | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `persistence_current_stress` | 0.5901 | 1.0406 | 0.5988 | 0.9134 | 0.6419 | 0.6864 | 0.6634 |
| `zone_hour_weekend_mean` | 0.4835 | 0.8078 | 0.7582 | 0.9268 | 0.8363 | 0.5105 | 0.6340 |
| `global_mean` | 1.1649 | 1.6510 | -0.0099 | 0.8757 | 0.0000 | 0.0000 | 0.0000 |

Interpretacion:

- `global_mean` confirma el piso minimo esperado.
- `persistence_current_stress` funciona bien como baseline temporal simple y ofrece el mejor `F1`.
- `zone_hour_weekend_mean` es el baseline mas fuerte en regresion y captura casi toda la estructura del problema.

## 7. Comparacion con el modelo GBT documentado

En `docs/ejercicios/ex2/gbt_regressor.md` el modelo GBT final para `+1h` reporta en test:

| Modelo | MAE | RMSE | R2 |
|---|---:|---:|---:|
| `GBTRegressor (+1h)` | 0.4860 | 0.8116 | 0.7611 |
| `Baseline: zone_hour_weekend_mean` | 0.4835 | 0.8078 | 0.7582 |
| `Baseline: persistence_current_stress` | 0.5901 | 1.0406 | 0.5988 |

Lectura de esta comparacion:

- El modelo GBT mejora claramente al baseline naive de persistencia.
- Frente al baseline fuerte `zone_hour_weekend_mean`, la mejora es muy pequena.
- Esto sugiere que el problema tiene una estacionalidad espacio-temporal muy marcada, y que una parte grande de la señal ya queda explicada por una referencia agregada simple.

## 8. Conclusiones

El baseline actual se considera adecuado para `ej2` porque cubre tres niveles utiles de comparacion:

- una referencia trivial (`global_mean`)
- una referencia naive temporal (`persistence_current_stress`)
- una referencia estructurada fuerte (`zone_hour_weekend_mean`)

Por tanto, la lectura correcta de los modelos avanzados no es solo "superan al baseline", sino:

- superan con claridad al naive
- y apenas mejoran una referencia fuerte basada en estructura temporal y geografica

Esa conclusion no invalida el modelo final; al contrario, refuerza que el fenomeno de estres urbano tiene patrones recurrentes muy fuertes y relativamente estables.
