# Red neuronal tabular con embeddings — Predicción de Estrés


El objetivo es predecir el nivel de **estrés futuro** en una zona geográfica, agregado por hora y día. Se definen **tres targets** distintos que representan el estrés en diferentes horizontes temporales:
 
| Target | Horizonte |
|---|---|
| `target_stress_t1` | +1 hora |
| `target_stress_t3` | +3 horas |
| `target_stress_t24` | +24 horas |
 
Se entrena **un modelo independiente por target**, todos con la misma arquitectura: **red neuronal tabular en PyTorch con embeddings para categóricas**, usando Spark únicamente para lectura, filtrado y partición temporal del dataset.

En los resultados adjuntos para este notebook se dispone del experimento `target_stress_t24`, por lo que el análisis queda centrado en el horizonte **+24h**. El código está preparado para incorporar automáticamente `t1` y `t3` si existen sus carpetas de salida correspondientes.


Los datos provienen de un dataset de parquets particionados en `data/aggregated/ex_stress/df_stress_zone_hour_day`. Se filtran filas donde el target o el timestamp sean nulos.
 
La partición es **temporal estricta** (no aleatoria), lo cual es crítico para evitar data leakage en series temporales:
 
```text
Train: 70%  →  Val: 15%  →  Test: 15%
```
 
El split se hace por `timestamp_hour`, preservando el orden cronológico. En el experimento +24h los cortes fueron:

| Split | Fecha |
|---|---|
| Fin de train | `2025-02-07 13:00:00` |
| Inicio de validation | `2025-02-07 14:00:00` |
| Fin de validation | `2025-07-20 18:00:00` |
| Inicio de test | `2025-07-20 19:00:00` |


## Preprocesamiento y pipeline
 
El pipeline del modelo neuronal tiene las siguientes etapas en orden:
 
### a) Preparación con Spark
- Se lee el parquet completo con Spark.
- Se aplica el filtro de `timestamp_hour` y del target no nulo.
- Se realiza un split temporal estricto `train → validation → test`.
- Tras el split, los subconjuntos se materializan para evitar recálculos.

### b) Preprocesamiento para PyTorch
- Las columnas numéricas se imputan con la mediana de train y se estandarizan con media/desviación típica de train.
- Las columnas categóricas se transforman con mapas aprendidos en train, reservando índice `0` para valores desconocidos (`<UNK>`).
- En este experimento hay **32 variables numéricas** y **1 categórica** (`borough`), con **8 categorías**.

### c) Modelo
- `EmbeddingMLPRegressor` combina las numéricas escaladas con embeddings aprendidos para categóricas.
- La red usa capas densas `64 → 32`, activación `ReLU` y `Dropout`.
- Se entrena directamente sobre la escala del stress con `SmoothL1Loss`, lo que hace el entrenamiento más robusto frente a valores extremos que un MSE puro.


## Hiperparámetros
 
**Defaults conservadores** utilizados por el script `d_model_nn.py` y confirmados en el report adjunto:
 
| Parámetro | Valor | Descripción |
|---|---:|---|
| `nn_hidden_dims` | `64,32` | Arquitectura de capas ocultas del MLP |
| `nn_dropout` | `0.15` | Regularización por apagado aleatorio de neuronas |
| `nn_learning_rate` | `0.001` | Learning rate inicial de AdamW |
| `nn_weight_decay` | `1e-05` | Regularización L2 del optimizador |
| `nn_batch_size` | `2048` | Tamaño de batch |
| `nn_max_epochs` | `30` | Máximo de épocas |
| `nn_patience` | `5` | Early stopping si validation no mejora |
| `nn_embedding_dim_cap` | `12` | Dimensión máxima de los embeddings categóricos |
| `device` | `cuda` | Entrenamiento ejecutado en GPU |


## Búsqueda de hiperparámetros (tuning)
 
Cuando se activa `--tune`, el script realiza un **grid search manual sobre validación temporal**:
 
- Grid por defecto: arquitecturas `256,128,64` y `512,256,128`, `dropout ∈ {0.15,0.25}`, `lr ∈ {1e-3,5e-4}`, `weight_decay ∈ {1e-5,5e-5}`, `batch_size ∈ {1024,2048}` y `embedding_dim_cap ∈ {24,32}`.
- Se muestrea una fracción de train/val para acelerar (`tune_train_sample_frac`, `tune_val_sample_frac`).
- Métrica de selección configurable: `rmse` (default), `mae`, o `r2`.
- El mejor trial se guarda en `neural_network_stress_tuning_report.json` y todos los trials en `neural_network_stress_tuning_trials.csv`.

En el resultado adjunto, el tuning aparece como **desactivado** (`enabled=false`), por lo que se evalúa la configuración base `64,32`.


## Protocolo de entrenamiento y evaluación
 
```text
[Train] → fit red neuronal → predice en Train, Val, Test  (detectar overfitting)
           ↓
[Validation] → early stopping y reducción de learning rate
           ↓
[Test] → evaluación final sobre datos futuros no vistos
```
 
- El **modelo base** se entrena solo con train y usa validation para controlar el entrenamiento.
- En este experimento **no se activó** `--refit-train-val`; por tanto, el modelo final guardado corresponde al entrenamiento en train con early stopping sobre validation.
- Opcionalmente, `--fit-all-data` entrena un tercer modelo sobre train+val+test para despliegue, pero sus métricas son in-sample y no válidas para medir generalización.

**Métricas calculadas:** `MAE`, `RMSE`, `R²`
 
**Overfit gaps calculados automáticamente:**
- `val_rmse - train_rmse`
- `test_rmse - train_rmse`
- `train_r2 - val_r2`
- `train_r2 - test_r2`


## A) Análisis numérico


**1. Desempeño y Generalización**

A partir de las métricas obtenidas para el horizonte temporal +24h, podemos extraer tres conclusiones clave que validan la robustez del modelo neuronal:

- **Ausencia de overfitting relevante:** La diferencia entre train y datos no vistos es reducida. El $R^2$ pasa de **0.7677** en train a **0.7558** en validation y **0.7524** en test. La caída frente al test es de solo **0.0153 puntos de $R^2$**, lo que indica que la red no se limita a memorizar la serie histórica.

- **Alto poder predictivo general:** El modelo explica aproximadamente el **75.24% de la varianza** del estrés futuro a +24h en datos completamente posteriores al entrenamiento. Además, el error medio absoluto se mantiene en **0.4931** puntos en test, una desviación contenida para una variable sintética de estrés urbano.

- **Buen encaje para el horizonte diario:** El objetivo +24h se beneficia de la fuerte **estacionalidad diaria** de Nueva York. La red neuronal captura patrones repetitivos de demanda, precio, clima, eventos y zona, apoyándose en variables de lag/rolling y en el embedding de `borough` para representar diferencias espaciales.



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
      <td>+24h</td>
      <td>0.7677</td>
      <td>0.7558</td>
      <td>0.7524</td>
      <td>0.7713</td>
      <td>0.8051</td>
      <td>0.8262</td>
      <td>0.4661</td>
      <td>0.4776</td>
      <td>0.4931</td>
    </tr>
  </tbody>
</table>
</div>


**2. Overfit gaps**

Los resultados demuestran que el modelo de red neuronal generaliza de forma saludable sobre datos futuros. Las métricas de degradación entre datos vistos en entrenamiento y datos no vistos son contenidas:

- **Degradación controlada:** El RMSE sube de **0.7713** en train a **0.8051** en validation y **0.8262** en test. El gap test-train es de **0.0549**, razonable para un horizonte de 24 horas.

- **Consistencia entre validation y test:** La validación ya anticipa correctamente el rendimiento futuro: el RMSE de validation es **0.8051** y el de test **0.8262**, con una diferencia pequeña. Esto indica que la partición temporal está midiendo bien la generalización.

- **Validación de la arquitectura:** Mantener gaps estrechos confirma que las decisiones de diseño —capas `64,32`, `dropout=0.15`, `weight_decay=1e-5`, early stopping y embeddings categóricos— han funcionado correctamente, permitiendo que la red aprenda patrones reales sin memorizar ruido histórico.



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
      <th>Test - Train RMSE (Base)</th>
      <th>Train - Val R² (Base)</th>
      <th>Train - Test R² (Base)</th>
      <th>Test - TrainVal RMSE (Final)</th>
      <th>TrainVal - Test R² (Final)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>+24h</td>
      <td>0.0339</td>
      <td>0.0549</td>
      <td>0.0119</td>
      <td>0.0153</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


**3. Análisis del tuning**

En este experimento, el tuning aparece desactivado. Esto significa que el resultado +24h corresponde a la configuración base del script, no a una búsqueda de hiperparámetros:

- **Configuración estable:** La red `64,32` con `dropout=0.15`, `lr=0.001`, `weight_decay=1e-5` y `batch_size=2048` ya alcanza un $R^2$ de **0.7524** en test.

- **Early stopping efectivo:** Aunque `nn_max_epochs=30`, el historial adjunto termina en la época **13**, lo que indica parada temprana tras varias épocas sin mejora suficiente en validation.

- **Siguiente mejora natural:** Si se quisiera apurar rendimiento, el siguiente paso sería activar `--tune` y comparar arquitecturas más anchas (`256,128,64` o `512,256,128`) contra la base, vigilando que los gaps de overfitting no aumenten.



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
      <th>Tuning activo</th>
      <th>Hidden dims</th>
      <th>Dropout</th>
      <th>Learning rate</th>
      <th>Weight decay</th>
      <th>Batch size</th>
      <th>Embedding cap</th>
      <th>Max epochs</th>
      <th>Patience</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>+24h</td>
      <td>False</td>
      <td>64,32</td>
      <td>0.15</td>
      <td>0.001</td>
      <td>0.00001</td>
      <td>2048</td>
      <td>12</td>
      <td>30</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>


**4. Variables de entrada y feature importance**

La red neuronal utiliza una mezcla de señales temporales, operativas, económicas, contextuales y espaciales. En el report adjunto no existe `neural_network_stress_feature_importance.csv` porque `compute_importance=false`; por tanto, no se puede afirmar una importancia cuantitativa de cada feature sin ejecutar la importancia por permutación.

Aun así, el conjunto de entrada confirma una arquitectura informativa y coherente:

- **Variables de demanda y precio:** `n_trips`, `avg_price`, `price_variability`, `z_price_variability`, `z_log1p_num_trips` y `stress_score` resumen el estado actual del sistema.
- **Memoria temporal:** lags y rolling windows como `lag_24h_trips`, `lag_168h_trips`, `roll_24h_trips` o `roll_24h_price_variability` capturan inercia diaria/semanal.
- **Contexto externo:** clima, eventos, restauración y renta zonal ayudan a explicar variaciones de demanda no puramente temporales.
- **Componente espacial:** `borough` se codifica mediante un embedding aprendido, permitiendo que la red represente diferencias estructurales entre zonas sin expandir la dimensionalidad con one-hot.

Para obtener importancias reales, se puede reejecutar el script con `--compute-importance`; el notebook las cargará automáticamente si el CSV existe.



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
      <th>feature</th>
      <th>tipo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>+24h</td>
      <td>year</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>1</th>
      <td>+24h</td>
      <td>month</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>2</th>
      <td>+24h</td>
      <td>hour</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>3</th>
      <td>+24h</td>
      <td>hour_block_3h</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>4</th>
      <td>+24h</td>
      <td>day_of_week</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>5</th>
      <td>+24h</td>
      <td>is_weekend</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>6</th>
      <td>+24h</td>
      <td>pu_location_id</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>7</th>
      <td>+24h</td>
      <td>n_trips</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>8</th>
      <td>+24h</td>
      <td>price_variability</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>9</th>
      <td>+24h</td>
      <td>avg_price</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>10</th>
      <td>+24h</td>
      <td>lag_1h_trips</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>11</th>
      <td>+24h</td>
      <td>lag_24h_trips</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>12</th>
      <td>+24h</td>
      <td>lag_168h_trips</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>13</th>
      <td>+24h</td>
      <td>roll_3h_trips</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>14</th>
      <td>+24h</td>
      <td>roll_24h_trips</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>15</th>
      <td>+24h</td>
      <td>lag_1h_price_variability</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>16</th>
      <td>+24h</td>
      <td>lag_24h_price_variability</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>17</th>
      <td>+24h</td>
      <td>roll_3h_price_variability</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>18</th>
      <td>+24h</td>
      <td>roll_24h_price_variability</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>19</th>
      <td>+24h</td>
      <td>lag_1h_avg_price</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>20</th>
      <td>+24h</td>
      <td>lag_24h_avg_price</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>21</th>
      <td>+24h</td>
      <td>roll_24h_avg_price</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>22</th>
      <td>+24h</td>
      <td>temp_c</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>23</th>
      <td>+24h</td>
      <td>precip_mm</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>24</th>
      <td>+24h</td>
      <td>city_n_events</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>25</th>
      <td>+24h</td>
      <td>city_has_event</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>26</th>
      <td>+24h</td>
      <td>n_restaurants_zone</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>27</th>
      <td>+24h</td>
      <td>n_cuisines_zone</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>28</th>
      <td>+24h</td>
      <td>rent_price_zone</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>29</th>
      <td>+24h</td>
      <td>z_price_variability</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>30</th>
      <td>+24h</td>
      <td>z_log1p_num_trips</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>31</th>
      <td>+24h</td>
      <td>stress_score</td>
      <td>numérica</td>
    </tr>
    <tr>
      <th>32</th>
      <td>+24h</td>
      <td>borough</td>
      <td>categórica / embedding</td>
    </tr>
  </tbody>
</table>
</div>


    No se encontró CSV de importancias. Ejecuta d_model_nn.py con --compute-importance para generar neural_network_stress_feature_importance.csv.


## B) Análisis gráfico


**1. Barchart de variables / importancias**



    
![png](neural_network_files/neural_network_18_0.png)
    


**2. Heatmap de métricas**



    
![png](neural_network_files/neural_network_20_0.png)
    


**3. Scatter plot: predicciones vs. valores reales**

La red neuronal debe reproducir adecuadamente la tendencia general entre valores reales y predichos. En un buen ajuste, la densidad de puntos se alinea con la diagonal ideal. Para modelos de estrés urbano es habitual observar cierta regresión a la media: los valores extremos altos pueden quedar algo subestimados y los valores muy bajos algo sobreestimados.

La celda siguiente reconstruye las predicciones del modelo `.pt` guardado. Si se ejecuta fuera del repositorio o sin el dataset/modelo, la celda avisa y no rompe el notebook.


    WARNING: Using incubator modules: jdk.incubator.vector
    Using Spark's default log4j profile: org/apache/spark/log4j2-defaults.properties
    26/04/29 19:16:20 WARN Utils: Your hostname, vegapc, resolves to a loopback address: 127.0.1.1; using 10.255.255.254 instead (on interface lo)
    26/04/29 19:16:20 WARN Utils: Set SPARK_LOCAL_IP if you need to bind to another address
    Using Spark's default log4j profile: org/apache/spark/log4j2-defaults.properties
    Setting default log level to "WARN".
    To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
    26/04/29 19:16:22 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
    26/04/29 19:16:29 WARN DataSource: [COLUMN_ALREADY_EXISTS] The column `month` already exists. Choose another name or rename the existing column. SQLSTATE: 42711
    26/04/29 19:16:30 WARN SparkStringUtils: Truncated the string representation of a plan since it was too large. This behavior can be adjusted by setting 'spark.sql.debug.maxToStringFields'.
                                                                                    


    
![png](neural_network_files/neural_network_22_1.png)
    


**4. Distribución de residuos** 

La distribución de residuos permite comprobar si el modelo presenta sesgo global. En este caso, con las métricas del report se observa un error medio absoluto de **0.4931** y un RMSE de **0.8262** en test para +24h, valores consistentes con una concentración razonable de errores alrededor de cero.

Si se dispone del parquet y del `.pt`, la siguiente celda grafica los residuos reales (`real - predicho`) en escala normal y logarítmica. La lectura deseable es una distribución centrada cerca de cero, con colas controladas y sin desplazamiento sistemático fuerte.



    
![png](neural_network_files/neural_network_24_0.png)
    


**5. Residuos vs. tiempo**

El análisis temporal de residuos sirve para detectar degradación progresiva, cambios de régimen o sesgos por temporada. El resultado numérico ya sugiere estabilidad: validation y test tienen rendimientos próximos, con $R^2$ **0.7558** y **0.7524** respectivamente.

La siguiente gráfica, cuando se dispone de predicciones reconstruidas, muestra residuos individuales por zona/hora y una media móvil de 24h. Un modelo estable debería mantener la tendencia cerca de cero durante el periodo de test.



    
![png](neural_network_files/neural_network_26_0.png)
    


**6. Curva de entrenamiento y early stopping**

La curva de entrenamiento muestra cómo evoluciona la función de pérdida `SmoothL1Loss`. En el experimento +24h, la validation loss mejora de forma clara hasta la época **8**, donde alcanza su mejor valor aproximado (**0.2151**). A partir de ahí se estabiliza y empeora ligeramente, por lo que el early stopping detiene el proceso en la época **13**.

Esto refuerza la lectura de generalización: el entrenamiento no se fuerza hasta memorizar el train completo, sino que se conserva el estado que mejor funciona en validation.



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
      <th>epoch</th>
      <th>train_loss</th>
      <th>val_loss</th>
      <th>lr</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.247280</td>
      <td>0.225012</td>
      <td>0.0010</td>
      <td>+24h</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>0.222330</td>
      <td>0.225154</td>
      <td>0.0010</td>
      <td>+24h</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>0.217113</td>
      <td>0.222900</td>
      <td>0.0010</td>
      <td>+24h</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>0.214667</td>
      <td>0.220534</td>
      <td>0.0010</td>
      <td>+24h</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>0.213052</td>
      <td>0.220386</td>
      <td>0.0010</td>
      <td>+24h</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.0</td>
      <td>0.211956</td>
      <td>0.222591</td>
      <td>0.0010</td>
      <td>+24h</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.0</td>
      <td>0.211111</td>
      <td>0.216371</td>
      <td>0.0010</td>
      <td>+24h</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.0</td>
      <td>0.210610</td>
      <td>0.215122</td>
      <td>0.0010</td>
      <td>+24h</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9.0</td>
      <td>0.210156</td>
      <td>0.218711</td>
      <td>0.0010</td>
      <td>+24h</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10.0</td>
      <td>0.209789</td>
      <td>0.217325</td>
      <td>0.0010</td>
      <td>+24h</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11.0</td>
      <td>0.209239</td>
      <td>0.216126</td>
      <td>0.0005</td>
      <td>+24h</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12.0</td>
      <td>0.208568</td>
      <td>0.218395</td>
      <td>0.0005</td>
      <td>+24h</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13.0</td>
      <td>0.208345</td>
      <td>0.221250</td>
      <td>0.0005</td>
      <td>+24h</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](neural_network_files/neural_network_28_1.png)
    

