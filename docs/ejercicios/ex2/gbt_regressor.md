# GBT Regressor — Predicción de Estrés

El objetivo es predecir el nivel de **estrés futuro** en una zona geográfica, agregado por hora y día. Se definen **tres targets** distintos que representan el estrés en diferentes horizontes temporales:
 
| Target | Horizonte |
|---|---|
| `target_stress_t1` | +1 hora |
| `target_stress_t3` | +3 horas |
| `target_stress_t24` | +24 horas |
 
Se entrena **un modelo independiente por target**, todos con la misma arquitectura: **GBTRegressor sobre Spark ML**.

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
- `GBTRegressor` con `labelCol="label"` y `featuresCol="features"`

## Hiperparámetros
 
**Defaults conservadores** (orientados a entornos con recursos limitados):
 
| Parámetro | Default | Descripción |
|---|---|---|
| `maxIter` | 50 | Nº de árboles (iteraciones de boosting) |
| `maxDepth` | 4 | Profundidad máxima por árbol |
| `stepSize` | 0.1 | Learning rate |
| `subsamplingRate` | 0.8 | Fracción de datos por iteración (bagging) |
| `maxBins` | 16 | Bins para discretizar features continuas |
| `minInstancesPerNode` | 2 | Mínimo de muestras por hoja |
| `featureSubsetStrategy` | `sqrt` | Nº de features por split (estilo Random Forest) |

## Búsqueda de hiperparámetros (tuning)
 
Cuando se activa `--tune`, se realiza un **grid search manual sobre validación temporal**:
 
- Grid por defecto: `maxIter ∈ {30,50,70}`, `maxDepth ∈ {3,4,5}`, `stepSize ∈ {0.05,0.1}`, `subsamplingRate ∈ {0.7,0.8}`
- Se muestrea una fracción de train/val para acelerar (`tune_train_sample_frac`, `tune_val_sample_frac`)
- Métrica de selección configurable: `rmse` (default), `mae`, o `r2`
- El mejor trial se guarda en `gbt_stress_spark_tuning_report.json` y todos los trials en `gbt_stress_spark_tuning_trials.csv`

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

A partir de las métricas obtenidas para los tres horizontes temporales (+1h, +3h y +24h), podemos extraer tres conclusiones clave que validan la robustez de los modelos:

- **Ausencia de Overfitting (Excelente Generalización):** En los tres targets, la diferencia de rendimiento entre los datos de entrenamiento (Train) y los datos no vistos (Val y Test Final) es mínima. Por ejemplo, en el target a +1h, la caída del $R^2$ es de apenas un 0.0087 (de 0.7698 a 0.7611), y el RMSE crece menos de 0.05 puntos. Esto demuestra que la estrategia de partición temporal y la regularización del modelo (control de profundidad y _subsampling_) han funcionado perfectamente, evitando que el árbol memorice la serie.

- **Alto Poder Predictivo General:** Los tres modelos explican una proporción muy alta de la varianza del estrés. Mantener un $R^2$ en la franja del 0.73 - 0.76 en datos de test completamente futuros es un resultado excepcional para series temporales. Además, el error medio absoluto (MAE) se mantiene estable alrededor de 0.48 - 0.52 puntos, lo que significa que la desviación real de nuestras predicciones es muy contenida.

- **Comportamiento Temporal y Estacionalidad:** Lo habitual es que el rendimiento se degrade linealmente a medida que aumenta el horizonte de predicción. Sin embargo, observamos un patrón muy interesante: el modelo a +3h ($R^2$ 0.7380) rinde ligeramente peor que el modelo a +24h ($R^2$ 0.7551). Esto tiene todo el sentido del negocio: predecir el estrés exactamente a la misma hora del día siguiente (+24h) es más fácil debido a la fuerte **estacionalidad diaria** (patrones cíclicos de la ciudad), mientras que un horizonte de +3h cruza diferentes bloques horarios que pueden ser más volátiles e impredecibles (ej. pasar de un bloque valle a plena hora punta).


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
      <td>0.7698</td>
      <td>0.7597</td>
      <td>0.7611</td>
      <td>0.7678</td>
      <td>0.7984</td>
      <td>0.8116</td>
      <td>0.4596</td>
      <td>0.4780</td>
      <td>0.4860</td>
    </tr>
    <tr>
      <th>1</th>
      <td>+3h</td>
      <td>0.7537</td>
      <td>0.7431</td>
      <td>0.7380</td>
      <td>0.7942</td>
      <td>0.8255</td>
      <td>0.8501</td>
      <td>0.4921</td>
      <td>0.5080</td>
      <td>0.5246</td>
    </tr>
    <tr>
      <th>2</th>
      <td>+24h</td>
      <td>0.7653</td>
      <td>0.7573</td>
      <td>0.7551</td>
      <td>0.7752</td>
      <td>0.8026</td>
      <td>0.8217</td>
      <td>0.4693</td>
      <td>0.4833</td>
      <td>0.4976</td>
    </tr>
  </tbody>
</table>
</div>


**2. Overfit gaps**

Los resultados demuestran de forma contundente que ninguno de los tres modelos sufre de overfitting. Las métricas de degradación entre los datos vistos en entrenamiento y los datos estrictamente futuros (Test) son mínimas y excepcionalmente saludables:

- **Degradación casi nula:** Al evaluar el modelo final en el conjunto de Test, el error (RMSE) apenas sube unos 0.04 puntos de media, y la capacidad explicativa ($R^2$) cae menos de un 1% (aprox. 0.009) en todos los horizontes temporales.

- **Consistencia total:** Los gaps son prácticamente idénticos tanto en el modelo base (Train vs Val) como en el modelo final a producción (TrainVal vs Test). El comportamiento de los modelos es muy predecible, sin sorpresas al enfrentarse a datos nuevos.

- **Validación de la arquitectura:** Mantener unos gaps tan estrechos confirma que las decisiones de diseño y los hiperparámetros aplicados (como limitar el `maxDepth` y usar un `subsamplingRate` de 0.8) han funcionado a la perfección, logrando que el árbol aprenda los patrones reales de la ciudad y no memorice el ruido del pasado.


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
      <td>0.0306</td>
      <td>0.0101</td>
      <td>0.0411</td>
      <td>0.0087</td>
    </tr>
    <tr>
      <th>1</th>
      <td>+3h</td>
      <td>0.0313</td>
      <td>0.0106</td>
      <td>0.0441</td>
      <td>0.0101</td>
    </tr>
    <tr>
      <th>2</th>
      <td>+24h</td>
      <td>0.0274</td>
      <td>0.0080</td>
      <td>0.0422</td>
      <td>0.0093</td>
    </tr>
  </tbody>
</table>
</div>


**3. Análisis del tuning**

Revisando los 3 mejores trials (Top 3) resultantes del grid search para cada horizonte temporal, extraemos dos conclusiones muy claras que refuerzan la solidez del proyecto:

- **Unanimidad en los parámetros ganadores:** El espacio de búsqueda ha convergido exactamente hacia la misma configuración óptima para los tres horizontes temporales (+1h, +3h y +24h). El modelo ganador (Trial 8) siempre utiliza **70 iteraciones, profundidad máxima de 5 y un step size de 0.1**. Esto indica que la naturaleza del problema base (predecir estrés) mantiene una estructura matemática similar independientemente de la distancia temporal.

- **Alta estabilidad y robustez:** El _tuning_ es muy estable. En los tres targets, el orden de los mejores modelos es idéntico (Trials 8, 7 y 1) y la diferencia de error (RMSE) entre el primero y el tercero es mínima (apenas ~0.02 puntos). Esto demuestra que el modelo es robusto frente a ligeras variaciones de hiperparámetros; no hemos ganado "por casualidad" en un pico de suerte del algoritmo, sino que hemos encontrado una zona óptima y estable de aprendizaje.

- **Aprovechamiento de la complejidad:** Los modelos tienden a elegir los parámetros más complejos del grid propuesto (máxima profundidad y número de árboles evaluados). Como en el apartado anterior demostramos que no hay overfitting, esto es una gran noticia: significa que nuestros datos tienen suficiente señal y volumen para alimentar un modelo profundo sin que este empiece a memorizar ruido.


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
      <th>Max Iter</th>
      <th>Max Depth</th>
      <th>Step Size</th>
      <th>Val RMSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>+1h</td>
      <td>8</td>
      <td>70</td>
      <td>5</td>
      <td>0.1</td>
      <td>0.798501</td>
    </tr>
    <tr>
      <th>1</th>
      <td>+1h</td>
      <td>7</td>
      <td>70</td>
      <td>4</td>
      <td>0.1</td>
      <td>0.809365</td>
    </tr>
    <tr>
      <th>2</th>
      <td>+1h</td>
      <td>1</td>
      <td>50</td>
      <td>4</td>
      <td>0.1</td>
      <td>0.820875</td>
    </tr>
    <tr>
      <th>3</th>
      <td>+3h</td>
      <td>8</td>
      <td>70</td>
      <td>5</td>
      <td>0.1</td>
      <td>0.823337</td>
    </tr>
    <tr>
      <th>4</th>
      <td>+3h</td>
      <td>7</td>
      <td>70</td>
      <td>4</td>
      <td>0.1</td>
      <td>0.838629</td>
    </tr>
    <tr>
      <th>5</th>
      <td>+3h</td>
      <td>1</td>
      <td>50</td>
      <td>4</td>
      <td>0.1</td>
      <td>0.848859</td>
    </tr>
    <tr>
      <th>6</th>
      <td>+24h</td>
      <td>8</td>
      <td>70</td>
      <td>5</td>
      <td>0.1</td>
      <td>0.803563</td>
    </tr>
    <tr>
      <th>7</th>
      <td>+24h</td>
      <td>7</td>
      <td>70</td>
      <td>4</td>
      <td>0.1</td>
      <td>0.813509</td>
    </tr>
    <tr>
      <th>8</th>
      <td>+24h</td>
      <td>1</td>
      <td>50</td>
      <td>4</td>
      <td>0.1</td>
      <td>0.821250</td>
    </tr>
  </tbody>
</table>
</div>


**4. Feature Importance: Evolución Temporal**

Al comparar las 10 variables más predictivas en los tres horizontes (+1h, +3h y +24h), observamos cómo el modelo adapta inteligentemente su estrategia según la distancia de la predicción:

- **La volatilidad de precios:** La variable `z_price_variability` domina absolutamente en los tres modelos (pesos entre 0.17 y 0.23). Esto confirma una hipótesis de negocio clave: las fluctuaciones en el precio (dinamismo de tarifas) son el mejor indicador temprano del nivel de estrés y saturación de una zona.

- **El efecto "Espejo Diario" (+1h y +24h):** Los modelos a +1h y +24h se comportan de forma sorprendentemente similar. Le dan muchísimo peso al estado inmediato de la zona (`n_trips` y `avg_price`). Esto ocurre porque predecir a +1h es una continuación directa del presente, y predecir a +24h se apoya en la fortísima estacionalidad diaria (lo que pasa hoy a las 10:00 es muy similar a lo que pasará mañana a las 10:00).

- **El "Cambio de Régimen" (+3h):** El horizonte a +3h es el más complejo porque implica saltar a un bloque horario distinto (ej. pasar de un valle de tarde a la hora punta nocturna). Aquí, el modelo penaliza la información actual (`n_trips` cae de 0.15 a 0.04) y delega todo el peso a las variables temporales (`hour` sube de 0.09 a 0.17, y `hour_block_3h` dobla su peso) y a las tendencias suavizadas (`roll_24h_price_variability` salta de 0.03 a 0.10). El modelo "sabe" que la foto actual ya no sirve dentro de 3 horas, por lo que se guía por la hora del día y la inercia diaria.

- **El stress_score como ancla:** La métrica base de estrés se mantiene estable aportando un valor constante (0.04 - 0.06) en todos los horizontes, demostrando que su formulación inicial era robusta.


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
      <th>32</th>
      <td>z_price_variability</td>
      <td>0.2105</td>
      <td>0.1719</td>
      <td>0.2254</td>
    </tr>
    <tr>
      <th>18</th>
      <td>n_trips</td>
      <td>0.1499</td>
      <td>0.0468</td>
      <td>0.1479</td>
    </tr>
    <tr>
      <th>5</th>
      <td>hour</td>
      <td>0.0973</td>
      <td>0.1714</td>
      <td>0.0523</td>
    </tr>
    <tr>
      <th>25</th>
      <td>roll_24h_trips</td>
      <td>0.0699</td>
      <td>0.1098</td>
      <td>0.0631</td>
    </tr>
    <tr>
      <th>28</th>
      <td>stress_score</td>
      <td>0.0591</td>
      <td>0.0447</td>
      <td>0.0676</td>
    </tr>
    <tr>
      <th>0</th>
      <td>avg_price</td>
      <td>0.0528</td>
      <td>0.0032</td>
      <td>0.0728</td>
    </tr>
    <tr>
      <th>6</th>
      <td>hour_block_3h</td>
      <td>0.0455</td>
      <td>0.0869</td>
      <td>0.0315</td>
    </tr>
    <tr>
      <th>12</th>
      <td>lag_24h_avg_price</td>
      <td>0.0353</td>
      <td>0.0160</td>
      <td>0.0258</td>
    </tr>
    <tr>
      <th>13</th>
      <td>lag_24h_price_variability</td>
      <td>0.0338</td>
      <td>0.0087</td>
      <td>0.0391</td>
    </tr>
    <tr>
      <th>24</th>
      <td>roll_24h_price_variability</td>
      <td>0.0316</td>
      <td>0.1044</td>
      <td>0.0291</td>
    </tr>
  </tbody>
</table>
</div>


## B) Análisis gráfico

**1. Barchart de feature importances**


    
![png](gbt_regressor_files/gbt_regressor_18_0.png)
    


**2. Heatmap de métricas**


    
![png](gbt_regressor_files/gbt_regressor_20_0.png)
    


**3. Scatter plot: predicciones vs. valores reales**

El modelo reproduce adecuadamente la tendencia general entre valores reales y predichos, con una concentración de densidad alineada con la diagonal ideal. No obstante, se observa una dispersión considerable, lo que indica errores no despreciables incluso en regiones densas.

Se aprecia un sesgo sistemático leve hacia la media: los valores altos de estrés tienden a subestimarse, mientras que los valores bajos se sobreestiman. Además, el modelo parece presentar una cierta saturación en los valores predichos, dificultando la captura de picos extremos.

El rendimiento es más robusto en el rango de valores bajo-medio, donde se concentra la mayoría de las observaciones.
                                                                                    


    
![png](gbt_regressor_files/gbt_regressor_22_1.png)
    


**4. Distribución de residuos** 

La distribución de residuos en los tres horizontes temporales muestra un comportamiento globalmente adecuado, con una clara concentración en torno a cero, lo que indica ausencia de sesgo global significativo.

La forma leptocúrtica de las distribuciones revela que la mayoría de las predicciones presentan errores pequeños, especialmente en los rangos más frecuentes del fenómeno. Sin embargo, la presencia de colas relativamente largas indica que existen errores no despreciables en ciertos casos, particularmente en situaciones menos habituales.

Se observa una ligera asimetría hacia residuos positivos, lo que sugiere una tendencia del modelo a subestimar valores altos de estrés, en línea con el comportamiento conservador observado previamente.

Asimismo, se aprecia una leve degradación del rendimiento a medida que aumenta el horizonte temporal, reflejada en una mayor dispersión de los residuos. Este comportamiento es consistente con el incremento de incertidumbre inherente a predicciones a más largo plazo.

Finalmente, los resultados sugieren una posible heterocedasticidad, donde la varianza del error aumenta en escenarios más extremos.
                                                                                    


    
![png](gbt_regressor_files/gbt_regressor_24_1.png)
    


**5. Residuos vs. tiempo**

El modelo demuestra una robustez estructural notable al mantener la media de los residuos cercana a cero durante todo el periodo de test. Esta estabilidad confirma la ausencia de data drift, garantizando que las predicciones mantienen su validez técnica sin sufrir una degradación inmediata tras el entrenamiento inicial.

Se han detectado oscilaciones cíclicas en la tendencia de los errores, lo que revela que el modelo no captura con total exactitud la estacionalidad horaria de la ciudad. Estas fluctuaciones sugieren que el sistema tiende a ser ligeramente optimista o pesimista en franjas específicas, marcando una oportunidad de mejora futura.

La precisión es inversamente proporcional al horizonte temporal, mostrando un control casi total en las predicciones a corto plazo frente a una mayor volatilidad en las de largo plazo. El ensanchamiento de las ondas de error en horizontes lejanos es una consecuencia natural y esperada de la incertidumbre acumulada.

En conclusión, el sistema es plenamente apto para su despliegue operativo al presentar fallos predecibles y una varianza constante en el tiempo. La fiabilidad general es alta, siempre que se considere el aumento del margen de error en las planificaciones que superan el ciclo diario de veinticuatro horas.
                                                                                    


    
![png](gbt_regressor_files/gbt_regressor_26_1.png)
    


**6. Curva de tuning**

La curva de optimización muestra una exploración efectiva del espacio de hiperparámetros, logrando identificar el "Mejor Trial" (marcado en rojo) con el menor RMSE de validación. Se observa una tendencia de mejora respecto a los intentos iniciales, lo que justifica el proceso de tuning para maximizar la capacidad de generalización del modelo GBT. Este ajuste fino permite encontrar el equilibrio óptimo entre la profundidad de los árboles y la tasa de aprendizaje, evitando el sobreajuste.


    
![png](gbt_regressor_files/gbt_regressor_28_0.png)
    


**7. Comparativa de RMSE por horizonte temporal**


    
![png](gbt_regressor_files/gbt_regressor_30_0.png)
    

