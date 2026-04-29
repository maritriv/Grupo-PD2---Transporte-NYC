# Random Forest Regressor — Predicción de Estrés


El objetivo es predecir el nivel de **estrés futuro** en una zona geográfica, agregado por hora y día. Se definen **tres targets** distintos que representan el estrés en diferentes horizontes temporales:
 
| Target | Horizonte |
|---|---|
| `target_stress_t1` | +1 hora |
| `target_stress_t3` | +3 horas |
| `target_stress_t24` | +24 horas |
 
Se entrena **un modelo independiente por target**, todos con la misma arquitectura: **RandomForestRegressor sobre Spark ML**.


Los datos provienen de un dataset de parquets particionados en `data/aggregated/ex_stress/df_stress_zone_hour_day`. Se filtran filas donde el target o el timestamp sean nulos.
 
La partición es **temporal estricta** (no aleatoria), lo cual es crítico para evitar data leakage en series temporales:
 
```
Train: 70%  →  Val: 15%  →  Test: 15%
```
 
El split se hace por `timestamp_hour`, preservando el orden cronológico. Los bounds (fechas de corte) se guardan en el report JSON.


## Preprocesamiento y pipeline
 
El pipeline de Spark ML tiene las siguientes etapas en orden:
 
### a) Encoding de categóricas
- `StringIndexer` convierte cada columna string a índice numérico (`col__idx`)
- Por defecto **no se aplica OneHotEncoder** (mejor para árboles: evita explosión de memoria y dimensionalidad)
- Si se activa `--one-hot-cats`, se añade un `OneHotEncoder` tras el indexer
### b) Ensamblado de features
- `VectorAssembler` une todas las columnas numéricas + las categóricas codificadas en un único vector `features`
- `handleInvalid="keep"` en todas las etapas → los nulos se gestionan sin romper el pipeline
### c) Modelo
- `RandomForestRegressor` con `labelCol="label"` y `featuresCol="features"`


## Hiperparámetros
 
**Defaults conservadores** (orientados a entornos con recursos limitados):
 
| Parámetro | Default | Descripción |
|---|---|---|
| `numTrees` | 100 | Nº de árboles del bosque |
| `maxDepth` | 8 | Profundidad máxima por árbol |
| `subsamplingRate` | 0.6 | Fracción de datos por árbol |
| `maxBins` | 16 | Bins para discretizar features continuas |
| `maxMemoryInMB` | 32 | Memoria por nodo/partición para el algoritmo |
| `minInstancesPerNode` | 2 | Mínimo de muestras por hoja |
| `featureSubsetStrategy` | `sqrt` | Nº de features candidatas por split |


## Búsqueda de hiperparámetros (tuning)
 
Cuando se activa `--tune`, se realiza un **grid search manual sobre validación temporal**:
 
- Grid por defecto: `numTrees ∈ {80,100,120}`, `maxDepth ∈ {7,8}`, `subsamplingRate ∈ {0.5,0.6}`, `maxBins ∈ {16}`, `maxMemoryInMB ∈ {32}`
- Se muestrea una fracción de train/val para acelerar (`tune_train_sample_frac`, `tune_val_sample_frac`)
- Métrica de selección configurable: `rmse` (default), `mae`, o `r2`
- El mejor trial se guarda en `random_forest_stress_spark_tuning_report.json` y todos los trials en `random_forest_stress_spark_tuning_trials.csv`


## Protocolo de entrenamiento y evaluación
 
```
[Train] → fit pipeline → predice en Train, Val, Test  (detectar overfitting)
           ↓
[Train + Val] → refit con mejores hiperparámetros → predice en Test  (modelo final)
```
 
- El **modelo base** (solo train) sirve para medir overfitting mirando los gaps entre splits.
- El **refit train+val** es el modelo final que se guarda y se usa para evaluar en test.
- Opcionalmente, `--fit-all-data` entrena un tercer modelo sobre train+val+test para despliegue (sus métricas son in-sample y no válidas para medir generalización).

**Métricas calculadas:** `MAE`, `RMSE`, `R²`
 
**Overfit gaps calculados automáticamente:**
- `val_rmse - train_rmse`
- `test_rmse - train_rmse`
- `train_r2 - val_r2`
- `train_r2 - test_r2`


## A) Análisis numérico


**1. Desempeño y Generalización**

A partir de las métricas obtenidas para los tres horizontes temporales (+1h, +3h y +24h), observamos un comportamiento sólido del modelo Random Forest:

- **Buena generalización en los tres targets:** el modelo mantiene brechas moderadas entre train, validation y test final, sin señales de sobreajuste severo.
- **Capacidad predictiva alta y estable:** en test final, el $R^2$ se mantiene en torno a **0.75 - 0.79** (`+1h: 0.7863`, `+3h: 0.7474`, `+24h: 0.7744`), con errores MAE entre `0.499` y `0.557`.
- **Patrón temporal coherente:** el horizonte `+3h` resulta el más exigente (mayor RMSE), mientras que `+1h` y `+24h` muestran mejor ajuste, consistente con continuidad de corto plazo y estacionalidad diaria.



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Target</th>
      <th>Train R²</th>
      <th>Val R²</th>
      <th>Test Final R²</th>
      <th>Train RMSE</th>
      <th>Val RMSE</th>
      <th>Test Final RMSE</th>
      <th>Train MAE</th>
      <th>Val MAE</th>
      <th>Test Final MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>+1h</td>
      <td>0.7983</td>
      <td>0.7861</td>
      <td>0.7863</td>
      <td>0.7343</td>
      <td>0.7705</td>
      <td>0.7935</td>
      <td>0.4676</td>
      <td>0.4842</td>
      <td>0.4990</td>
    </tr>
    <tr>
      <th>1</th>
      <td>+3h</td>
      <td>0.7639</td>
      <td>0.7530</td>
      <td>0.7474</td>
      <td>0.7946</td>
      <td>0.8277</td>
      <td>0.8629</td>
      <td>0.5219</td>
      <td>0.5353</td>
      <td>0.5566</td>
    </tr>
    <tr>
      <th>2</th>
      <td>+24h</td>
      <td>0.7872</td>
      <td>0.7771</td>
      <td>0.7744</td>
      <td>0.7543</td>
      <td>0.7865</td>
      <td>0.8154</td>
      <td>0.4887</td>
      <td>0.5031</td>
      <td>0.5221</td>
    </tr>
  </tbody>
</table>
</div>


**2. Overfit gaps**

Los gaps entre entrenamiento y evaluación futura son contenidos y bastante homogéneos:

- En el modelo final (`train+val -> test`), el incremento de RMSE está alrededor de **0.053 - 0.063** según el target.
- La caída de $R^2$ frente a `train_val` es pequeña, en torno a **0.010 - 0.015**.
- Esto confirma que el pipeline temporal y la regularización implícita del bosque (`maxDepth`, `subsamplingRate`) están controlando bien el riesgo de sobreajuste.



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Target</th>
      <th>Val - Train RMSE (Base)</th>
      <th>Train - Val R² (Base)</th>
      <th>Test - TrainVal RMSE (Final)</th>
      <th>TrainVal - Test R² (Final)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>+1h</td>
      <td>0.0362</td>
      <td>0.0123</td>
      <td>0.0532</td>
      <td>0.0102</td>
    </tr>
    <tr>
      <th>1</th>
      <td>+3h</td>
      <td>0.0332</td>
      <td>0.0109</td>
      <td>0.0632</td>
      <td>0.0151</td>
    </tr>
    <tr>
      <th>2</th>
      <td>+24h</td>
      <td>0.0322</td>
      <td>0.0101</td>
      <td>0.0558</td>
      <td>0.0114</td>
    </tr>
  </tbody>
</table>
</div>


**3. Análisis del tuning**

Revisando los Top 3 trials por horizonte temporal se observan patrones claros:

- **Zona óptima estable:** las mejores combinaciones convergen en `maxDepth=8`, con `numTrees` entre `80` y `100` y `subsamplingRate` entre `0.5` y `0.6`.
- **Diferencias pequeñas entre trials top:** la separación en `val_rmse` entre el mejor y el tercer trial es reducida en cada target, señal de robustez del modelo.
- **Sin dependencia de un único trial “mágico”:** el rendimiento se mantiene competitivo en varias configuraciones cercanas, lo que da confianza para reproducibilidad operativa.



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Target</th>
      <th>Trial</th>
      <th>Num Trees</th>
      <th>Max Depth</th>
      <th>Subsampling</th>
      <th>Val RMSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>+1h</td>
      <td>6</td>
      <td>100</td>
      <td>8</td>
      <td>0.5</td>
      <td>0.772641</td>
    </tr>
    <tr>
      <th>1</th>
      <td>+1h</td>
      <td>1</td>
      <td>100</td>
      <td>8</td>
      <td>0.6</td>
      <td>0.772784</td>
    </tr>
    <tr>
      <th>2</th>
      <td>+1h</td>
      <td>3</td>
      <td>80</td>
      <td>8</td>
      <td>0.5</td>
      <td>0.773411</td>
    </tr>
    <tr>
      <th>3</th>
      <td>+3h</td>
      <td>1</td>
      <td>100</td>
      <td>8</td>
      <td>0.6</td>
      <td>0.829860</td>
    </tr>
    <tr>
      <th>4</th>
      <td>+3h</td>
      <td>6</td>
      <td>100</td>
      <td>8</td>
      <td>0.5</td>
      <td>0.830290</td>
    </tr>
    <tr>
      <th>5</th>
      <td>+3h</td>
      <td>4</td>
      <td>80</td>
      <td>8</td>
      <td>0.6</td>
      <td>0.830549</td>
    </tr>
    <tr>
      <th>6</th>
      <td>+24h</td>
      <td>4</td>
      <td>80</td>
      <td>8</td>
      <td>0.6</td>
      <td>0.785464</td>
    </tr>
    <tr>
      <th>7</th>
      <td>+24h</td>
      <td>3</td>
      <td>80</td>
      <td>8</td>
      <td>0.5</td>
      <td>0.785678</td>
    </tr>
    <tr>
      <th>8</th>
      <td>+24h</td>
      <td>1</td>
      <td>100</td>
      <td>8</td>
      <td>0.6</td>
      <td>0.785761</td>
    </tr>
  </tbody>
</table>
</div>


**4. Feature Importance: Evolución Temporal**

El patrón de importancia revela estructura temporal consistente del problema:

- **Variables de volumen y estado agregado dominan** en los tres horizontes: `n_trips`, `z_log1p_num_trips`, `roll_24h_trips` y `stress_score` aparecen de forma recurrente.
- En `+3h`, gana peso la **inercia de ventana** (`roll_24h_trips`, `roll_24h_price_variability`), reflejando mayor dificultad al cruzar bloques horarios.
- En `+1h` y `+24h`, el modelo conserva mayor señal de estado inmediato (`n_trips`, `stress_score`) y rezagos largos (`lag_168h_trips`), coherente con persistencia de corto y ciclo diario.



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>+1h</th>
      <th>+3h</th>
      <th>+24h</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28</th>
      <td>stress_score</td>
      <td>0.1374</td>
      <td>0.0768</td>
      <td>0.1283</td>
    </tr>
    <tr>
      <th>31</th>
      <td>z_log1p_num_trips</td>
      <td>0.1221</td>
      <td>0.0823</td>
      <td>0.1213</td>
    </tr>
    <tr>
      <th>18</th>
      <td>n_trips</td>
      <td>0.1176</td>
      <td>0.0834</td>
      <td>0.1324</td>
    </tr>
    <tr>
      <th>25</th>
      <td>roll_24h_trips</td>
      <td>0.1125</td>
      <td>0.2110</td>
      <td>0.0934</td>
    </tr>
    <tr>
      <th>8</th>
      <td>lag_168h_trips</td>
      <td>0.0903</td>
      <td>0.0739</td>
      <td>0.0814</td>
    </tr>
    <tr>
      <th>24</th>
      <td>roll_24h_price_variability</td>
      <td>0.0531</td>
      <td>0.0917</td>
      <td>0.0422</td>
    </tr>
    <tr>
      <th>14</th>
      <td>lag_24h_trips</td>
      <td>0.0514</td>
      <td>0.0366</td>
      <td>0.0588</td>
    </tr>
    <tr>
      <th>32</th>
      <td>z_price_variability</td>
      <td>0.0483</td>
      <td>0.0273</td>
      <td>0.0538</td>
    </tr>
    <tr>
      <th>11</th>
      <td>lag_1h_trips</td>
      <td>0.0376</td>
      <td>0.0237</td>
      <td>0.0534</td>
    </tr>
    <tr>
      <th>20</th>
      <td>price_variability</td>
      <td>0.0350</td>
      <td>0.0082</td>
      <td>0.0379</td>
    </tr>
  </tbody>
</table>
</div>


## B) Análisis gráfico


**1. Barchart de feature importances**



    
![png](random_forest_regressor_files/random_forest_regressor_18_0.png)
    


**2. Heatmap de métricas**



    
![png](random_forest_regressor_files/random_forest_regressor_20_0.png)
    


**3. Scatter plot: predicciones vs. valores reales**

El modelo reproduce adecuadamente la tendencia general entre valores reales y predichos, con una concentración de densidad alineada con la diagonal ideal. No obstante, se observa una dispersión considerable, lo que indica errores no despreciables incluso en regiones densas.

Se aprecia un sesgo sistemático leve hacia la media: los valores altos de estrés tienden a subestimarse, mientras que los valores bajos se sobreestiman. Además, el modelo parece presentar una cierta saturación en los valores predichos, dificultando la captura de picos extremos.

El rendimiento es más robusto en el rango de valores bajo-medio, donde se concentra la mayoría de las observaciones.


    WARNING: Using incubator modules: jdk.incubator.vector
    Using Spark's default log4j profile: org/apache/spark/log4j2-defaults.properties
    26/04/29 10:35:23 WARN Utils: Your hostname, rosita, resolves to a loopback address: 127.0.1.1; using 10.255.255.254 instead (on interface lo)
    26/04/29 10:35:23 WARN Utils: Set SPARK_LOCAL_IP if you need to bind to another address
    Using Spark's default log4j profile: org/apache/spark/log4j2-defaults.properties
    Setting default log level to "WARN".
    To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
    26/04/29 10:35:25 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
    26/04/29 10:35:39 WARN DataSource: [COLUMN_ALREADY_EXISTS] The column `month` already exists. Choose another name or rename the existing column. SQLSTATE: 42711
    26/04/29 10:35:55 WARN SparkStringUtils: Truncated the string representation of a plan since it was too large. This behavior can be adjusted by setting 'spark.sql.debug.maxToStringFields'.
                                                                                    


    
![png](random_forest_regressor_files/random_forest_regressor_22_1.png)
    


**4. Distribución de residuos** 

La distribución de residuos en los tres horizontes temporales muestra un comportamiento globalmente adecuado, con una clara concentración en torno a cero, lo que indica ausencia de sesgo global significativo.

La forma leptocúrtica de las distribuciones revela que la mayoría de las predicciones presentan errores pequeños, especialmente en los rangos más frecuentes del fenómeno. Sin embargo, la presencia de colas relativamente largas indica que existen errores no despreciables en ciertos casos, particularmente en situaciones menos habituales.

Se observa una ligera asimetría hacia residuos positivos, lo que sugiere una tendencia del modelo a subestimar valores altos de estrés, en línea con el comportamiento conservador observado previamente.

Asimismo, se aprecia una leve degradación del rendimiento a medida que aumenta el horizonte temporal, reflejada en una mayor dispersión de los residuos. Este comportamiento es consistente con el incremento de incertidumbre inherente a predicciones a más largo plazo.

Finalmente, los resultados sugieren una posible heterocedasticidad, donde la varianza del error aumenta en escenarios más extremos.


    26/04/29 10:37:51 WARN DataSource: [COLUMN_ALREADY_EXISTS] The column `month` already exists. Choose another name or rename the existing column. SQLSTATE: 42711
                                                                                    


    
![png](random_forest_regressor_files/random_forest_regressor_24_1.png)
    


**5. Residuos vs. tiempo**

El modelo demuestra una robustez estructural notable al mantener la media de los residuos cercana a cero durante todo el periodo de test. Esta estabilidad confirma la ausencia de data drift, garantizando que las predicciones mantienen su validez técnica sin sufrir una degradación inmediata tras el entrenamiento inicial.

Se han detectado oscilaciones cíclicas en la tendencia de los errores, lo que revela que el modelo no captura con total exactitud la estacionalidad horaria de la ciudad. Estas fluctuaciones sugieren que el sistema tiende a ser ligeramente optimista o pesimista en franjas específicas, marcando una oportunidad de mejora futura.

La precisión es inversamente proporcional al horizonte temporal, mostrando un control casi total en las predicciones a corto plazo frente a una mayor volatilidad en las de largo plazo. El ensanchamiento de las ondas de error en horizontes lejanos es una consecuencia natural y esperada de la incertidumbre acumulada.

En conclusión, el sistema es plenamente apto para su despliegue operativo al presentar fallos predecibles y una varianza constante en el tiempo. La fiabilidad general es alta, siempre que se considere el aumento del margen de error en las planificaciones que superan el ciclo diario de veinticuatro horas.


    26/04/29 10:40:24 WARN DataSource: [COLUMN_ALREADY_EXISTS] The column `month` already exists. Choose another name or rename the existing column. SQLSTATE: 42711
                                                                                    


    
![png](random_forest_regressor_files/random_forest_regressor_26_1.png)
    


**6. Curva de tuning**

La curva de optimización permite visualizar la exploración de hiperparámetros y el mejor trial (mínimo RMSE de validación) en cada horizonte. En Random Forest, se aprecia una zona de rendimiento estable alrededor de profundidades altas moderadas (`maxDepth=8`) y bosques de tamaño medio (`80-100` árboles).



    
![png](random_forest_regressor_files/random_forest_regressor_28_0.png)
    


**7. Comparativa de RMSE por horizonte temporal**



    
![png](random_forest_regressor_files/random_forest_regressor_30_0.png)
    

