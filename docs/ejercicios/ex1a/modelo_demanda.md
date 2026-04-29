# Prediccion de Zona de Maxima Demanda — EX1(a)

## 1. Contexto y objetivo

El objetivo del ejercicio **EX1(a)** es predecir que **zona de recogida** (`pu_location_id`) concentrara la **maxima demanda de viajes en la siguiente hora**. A diferencia de otros problemas de forecasting donde se predice una magnitud continua, aqui el resultado final es una **clase**: la zona ganadora en terminos de numero de viajes.

La tarea se formula como un problema de **clasificacion multiclase** con horizonte corto:

| Elemento | Definicion |
|---|---|
| Unidad de prediccion | Una hora (`timestamp_hour`) |
| Target base | `target_n_trips_t1` |
| Etiqueta final | Zona con mayor `target_n_trips_t1` en esa hora |
| Regla de desempate | Menor `pu_location_id` |

Esta formulacion permite responder a una pregunta operativa concreta: **si tuvieramos que anticipar la zona mas tensionada del sistema en la proxima hora, cual deberiamos vigilar primero**.

## 2. Arquitectura del pipeline

El pipeline sigue una cadena de transformacion por capas:

```text
TLC estandarizado (Capa 2)
        +
meteo + eventos + rent + restaurantes
        |
        v
Capa 3 EX1(a) - dataset zona-hora enriquecido
(data/aggregated/ex1a/df_demand_zone_hour_day/)
        |
        v
Reformulacion multiclase por hora
(1 muestra = 1 hora, 1 etiqueta = zona ganadora)
        |
        v
Split temporal train / val / test
        |
        v
Entrenamiento de modelos de clasificacion
        |
        v
Artefactos y metricas
(outputs/ml/max_demand_zone/)
```

La construccion del dataset enriquecido se realiza en `src/procesamiento/capa3/builders/demand_zone.py`, mientras que el entrenamiento de modelos se implementa en `src/ml/models_ej1/model_a_demanda.py`.

## 3. Construccion del dataset de entrada

### 3.1 Fuente base

El punto de partida es la capa estandarizada de TLC (`data/standarized/`), que se agrega a nivel de:

- `date`
- `hour`
- `pu_location_id`

Cada fila del dataset EX1(a) representa por tanto una combinacion **zona-hora**, y el valor principal observado es `num_trips`, es decir, el numero de viajes iniciados en esa zona durante esa hora.

### 3.2 Variables generadas en Capa 3

Sobre esa base se construye `data/aggregated/ex1a/df_demand_zone_hour_day/`, que contiene:

- Variables temporales: `year`, `month`, `hour`, `hour_block_3h`, `day_of_week`, `is_weekend`, `week_of_year`
- Variable geografica: `pu_location_id`
- Señal actual de demanda: `num_trips`
- Variables historicas por zona:
  - `lag_1h`
  - `lag_24h`
  - `lag_168h`
  - `rolling_mean_3h`
  - `rolling_mean_24h`
- Variables externas:
  - `temp_c`
  - `precip_mm`
  - `city_n_events`
  - `city_has_event`
- Variables estaticas por zona:
  - `n_restaurants_zone`
  - `n_cuisines_zone`
  - `rent_price_zone`
- Targets multi-horizonte:
  - `target_n_trips_t1`
  - `target_n_trips_t3`
  - `target_n_trips_t24`

El builder completa explicitamente la rejilla temporal observada por zona, de modo que cuando una combinacion hora-zona no aparece en los datos originales se rellena con `num_trips = 0`. Esto es importante para que los lags y las medias moviles representen horas reales consecutivas y no solo horas con actividad.

### 3.3 Reformulacion multiclase

El modelo final no entrena directamente sobre filas zona-hora. Primero convierte el panel a un problema multiclase por hora:

1. Para cada `timestamp_hour`, se identifica la zona con mayor `target_n_trips_t1`.
2. Esa zona pasa a ser la etiqueta `target_zone_id`.
3. Las variables globales de calendario, meteorologia y eventos se conservan una sola vez por hora.
4. Las señales historicas por zona se pivotan a formato ancho, generando columnas del tipo:
   - `zone_79__lag_1h`
   - `zone_138__lag_168h`
   - `zone_161__rolling_mean_3h`

Con ello, cada muestra final representa una hora concreta y resume tanto el contexto global como el estado reciente de las zonas candidatas.

## 4. Variables utilizadas por el clasificador final

Aunque el dataset enriquecido contiene tambien variables estaticas de restaurantes y renta, la version final del clasificador se entreno con tres bloques de variables:

### 4.1 Variables globales

- `month`
- `hour`
- `hour_block_3h`
- `day_of_week`
- `is_weekend`
- `temp_c`
- `precip_mm`
- `city_n_events`
- `city_has_event`

### 4.2 Variables historicas por zona

Para cada zona candidata se utilizaron las siguientes señales:

- `num_trips`
- `lag_1h`
- `lag_24h`
- `lag_168h`
- `rolling_mean_3h`
- `rolling_mean_24h`

### 4.3 Seleccion de variables

El pivotado completo generaba **1.324 columnas de entrada** potenciales. Para reducir ruido y dimensionalidad, se aplico una seleccion de features basada en las zonas ganadoras observadas en entrenamiento (`feature_scope = train_winner_zones`), quedando finalmente **249 variables**.

Esta estrategia evita llenar la matriz con señales de zonas que nunca llegan a ganar en train y que, por tanto, aportan muy poca informacion para la decision multiclase.

## 5. Split temporal

El split no se hace de forma aleatoria, sino **estrictamente temporal**, lo que evita *data leakage* entre pasado y futuro:

```text
Train: 70%  ->  Val: 15%  ->  Test: 15%
```

Fechas de corte observadas en el dataset final:

| Split | Inicio | Fin |
|---|---|---|
| Train | 2023-01-08 00:00:00 | 2025-02-08 06:00:00 |
| Val | 2025-02-08 07:00:00 | 2025-07-21 14:00:00 |
| Test | 2025-07-21 15:00:00 | 2025-12-31 23:00:00 |

El conjunto final contiene:

- `18.295` muestras de entrenamiento
- `3.920` muestras de validacion
- `3.921` muestras de test

En total, el problema se entreno sobre **26.136 horas observadas**, derivadas de `6.873.768` filas validas del panel zona-hora.

## 6. Modelos empleados

Se evaluaron tres clasificadores multiclase:

### 6.1 Regresion logistica

Pipeline:

- `SimpleImputer(strategy="median")`
- `StandardScaler()`
- `LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")`

Actua como baseline lineal y sirve para medir hasta donde se puede llegar con una frontera de decision mas simple.

### 6.2 Random Forest

Pipeline:

- `SimpleImputer(strategy="median")`
- `RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample")`

Permite modelar relaciones no lineales y capturar interacciones entre las señales temporales y los lags de demanda por zona.

### 6.3 XGBoost

Pipeline:

- `SimpleImputer(strategy="median")`
- `XGBClassifier(objective="multi:softprob", n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)`

Fue el modelo mas potente del ejercicio y el que obtuvo el mejor rendimiento final en validacion y test.

## 7. Metricas de evaluacion

Dado que se trata de clasificacion multiclase con fuerte desbalance, se utilizaron varias metricas:

- `accuracy`: porcentaje de aciertos exactos
- `f1_macro`: media no ponderada de F1 por clase
- `f1_weighted`: F1 ponderada por frecuencia de clase
- `top_3_accuracy`: porcentaje de veces que la zona correcta aparece entre las 3 primeras propuestas
- `top_5_accuracy`: porcentaje de veces que la zona correcta aparece entre las 5 primeras propuestas

Las metricas `top-k` son especialmente relevantes en este caso, porque desde el punto de vista operativo puede ser suficiente identificar un pequeño conjunto de zonas criticas candidatas, aunque no siempre se acierte exactamente la primera.

## 8. Resultados

### 8.1 Resultados en test

| Modelo | Accuracy | F1 macro | F1 weighted | Top-3 | Top-5 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5542 | 0.1754 | 0.5907 | 0.8878 | 0.9533 |
| Random Forest | 0.6386 | 0.1901 | 0.6080 | 0.9156 | 0.9686 |
| XGBoost | **0.7133** | **0.2713** | **0.6916** | **0.9419** | **0.9758** |

El mejor modelo fue **XGBoost**, que supero claramente a la regresion logistica y tambien a Random Forest en todas las metricas clave de generalizacion.

### 8.2 Resultados por split

| Modelo | Split | Accuracy | F1 macro | F1 weighted |
|---|---|---:|---:|---:|
| Logistic Regression | Train | 0.7085 | 0.6411 | 0.7238 |
| Logistic Regression | Val | 0.6010 | 0.2476 | 0.6286 |
| Logistic Regression | Test | 0.5542 | 0.1754 | 0.5907 |
| Random Forest | Train | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | Val | 0.7196 | 0.2221 | 0.6943 |
| Random Forest | Test | 0.6386 | 0.1901 | 0.6080 |
| XGBoost | Train | 0.9866 | 0.7906 | 0.9863 |
| XGBoost | Val | 0.7288 | 0.2561 | 0.7129 |
| XGBoost | Test | 0.7133 | 0.2713 | 0.6916 |

La lectura de estos resultados deja tres mensajes claros:

1. **La regresion logistica se queda corta** para capturar la estructura no lineal del problema.
2. **Random Forest sobreajusta con fuerza**, ya que logra ajuste perfecto en train pero cae de forma importante en test.
3. **XGBoost mantiene el mejor equilibrio entre capacidad y generalizacion**.

## 9. Analisis de dificultad del problema

### 9.1 Fuerte desbalance de clases

Aunque el dataset contiene **51 clases observadas**, la distribucion del target esta muy concentrada:

| Zona | Horas ganadoras | % del total |
|---|---:|---:|
| 138 | 8.470 | 32.41% |
| 132 | 8.029 | 30.72% |
| 79 | 2.760 | 10.56% |
| 161 | 2.720 | 10.41% |
| 236 | 898 | 3.44% |

Algunos hechos relevantes:

- Las **3 zonas mas frecuentes concentran el 73.69%** de las muestras.
- Las **5 primeras concentran el 87.53%**.
- **24 de las 51 clases** aparecen 5 veces o menos.
- **11 clases** aparecen una sola vez.

Esto explica por que la `accuracy` puede ser relativamente alta mientras que la `f1_macro` sigue siendo baja: el modelo aprende bien las zonas dominantes, pero tiene mucha mas dificultad para acertar clases raras.

### 9.2 Clases no vistas en entrenamiento

Existe otra limitacion estructural: algunas zonas aparecen como ganadoras en validacion o test, pero **no aparecen nunca como ganadoras en train**. En los artefactos actuales esto ocurre con las zonas:

- `89`
- `145`
- `216`

Es imposible que un modelo acierte exactamente esas clases si nunca ha visto ejemplos positivos de ellas durante el entrenamiento. Esto penaliza especialmente la `f1_macro`.

### 9.3 Empates en la zona ganadora

El dataset contiene `99` horas con empate entre varias zonas ganadoras, con un tamaño maximo de empate de `5` zonas. Para garantizar reproducibilidad se resolvio el empate escogiendo la zona con menor `pu_location_id`.

## 10. Interpretacion de variables importantes

En el modelo XGBoost, las variables mas influyentes fueron:

| Feature | Importancia |
|---|---:|
| `hour_block_3h` | 0.1211 |
| `hour` | 0.0480 |
| `zone_79__lag_1h` | 0.0295 |
| `zone_161__lag_1h` | 0.0295 |
| `zone_138__lag_168h` | 0.0233 |
| `zone_161__lag_168h` | 0.0183 |
| `zone_148__rolling_mean_3h` | 0.0180 |
| `is_weekend` | 0.0151 |
| `zone_79__rolling_mean_3h` | 0.0146 |
| `day_of_week` | 0.0138 |

La interpretacion es bastante clara:

- La estructura **horaria** y **semanal** pesa mucho.
- Los lags por zona capturan fuerte **inercia temporal**.
- Las zonas dominantes reutilizan patrones muy estables en el tiempo.
- Las variables externas aportan contexto, pero no dominan la decision tanto como la historia reciente de demanda.

## 11. Conclusiones

El ejercicio EX1(a) demuestra que la zona de maxima demanda a una hora vista se puede predecir razonablemente bien usando una combinacion de calendario, contexto urbano ligero y memoria historica de la demanda por zona.

Las conclusiones principales son:

1. **XGBoost es el mejor modelo del ejercicio**, con `accuracy = 0.7133` y `top-5 accuracy = 0.9758` en test.
2. El problema esta dominado por **estacionalidad temporal e inercia espacial**, mas que por factores externos complejos.
3. El fuerte desbalance de clases limita la capacidad de generalizacion sobre zonas raras, lo que se refleja en una `f1_macro` modesta incluso en el mejor modelo.
4. Para uso operativo, las metricas `top-3` y `top-5` son especialmente valiosas: aunque no siempre se acierte la primera zona exacta, el modelo casi siempre deja la respuesta correcta dentro de un conjunto muy pequeño de candidatas.

En conjunto, el modelo final resulta util como herramienta de apoyo a la decision, especialmente en escenarios donde se quiera anticipar en que zonas conviene vigilar o reforzar la operativa en la siguiente hora.

## 12. Artefactos generados

Los resultados y artefactos de entrenamiento se guardan en:

```text
outputs/ml/max_demand_zone/
├── dataset_profile.json
├── training_summary.json
├── logistic_regression_report.json
├── random_forest_report.json
├── xgboost_report.json
├── *_feature_importance.csv
├── *_confusion_matrix_test.csv
└── xgboost.pkl
```

El dataset de entrada del ejercicio se construye en:

```text
data/aggregated/ex1a/df_demand_zone_hour_day/
```

## 13. Comandos utiles

Generar el dataset de EX1(a):

```bash
uv run -m src.procesamiento.capa3.ejercicios.ex1a_demand --mode overwrite
```

Entrenar los modelos:

```bash
uv run -m src.ml.models_ej1.model_a_demanda
```

Entrenar solo XGBoost:

```bash
uv run -m src.ml.models_ej1.model_a_demanda --models xgboost
```
