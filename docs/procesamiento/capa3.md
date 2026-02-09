# Capa 3: agregados de demanda y variabilidad de precio

La capa 3 toma como entrada la capa 2 estandarizada de viajes de taxi y genera varios DataFrames agregados que resumen:
- Cómo evoluciona la demanda y el precio por día y tipo de servicio.
- Cuáles son las combinaciones zona–hora–día con más actividad y precios más altos.
- En qué zonas y horas el precio es más impredecible (alta variabilidad).

Incluye además un script de inspección para revisar rápidamente la calidad de estos agregados.

---

## 1. Script `capa3.py`: construcción de la capa 3

### Objetivo

A partir de la capa 2 (`data/standarized`), crear y guardar cuatro tablas de agregados en `data/aggregated`:

1. `df_daily_service`: evolución diaria por tipo de servicio.
2. `df_zone_hour_day_global`: “hotspots” por zona y hora (global, mezclando servicios).
3. `df_zone_hour_day_service`: “hotspots” por zona, hora y tipo de servicio.
4. `df_variability`: variabilidad de precio por zona, hora y servicio (IQR + score de negocio).

### Sesión de Spark

La función `get_spark`:

- Crea una sesión Spark con nombre `"PD2-Capa3"`.
- Usa modo local (`local[*]`) con 6 GB de memoria para driver y executor.
- Ajusta `spark.sql.shuffle.partitions` a 200.
- Fija la zona horaria a `"America/New_York"`.
- Reduce los logs a nivel `WARN`.

---

### 1. Lectura de la capa 2 con higiene mínima

La función `read_layer2`:

- Lee la capa 2 desde `data/standarized` (ruta configurable con `layer2_path`).
- Aplica filtros “defensivos” para garantizar datos limpios antes de agregar:
  - Elimina filas con nulos en:
    - `date`
    - `hour`
    - `service_type`
    - `pu_location_id`
    - `total_amount_std`.
  - Limita el rango temporal:
    - Solo mantiene filas con `date` entre `min_date` y `max_date` (por defecto de 2019-01-01 a 2024-12-31, en `main` se acota a 2024-03-01).
  - Limpia el precio:
    - Quita viajes con `total_amount_std` menor o igual a 0 (refunds o errores).
    - Quita viajes con `total_amount_std` mayor o igual que `cap_max_price` (ej. 500), para evitar outliers extremos.

Resultado: un DataFrame de capa 2 listo para agregaciones estables.

---

### 2. Construcción de los agregados de capa 3

La función `build_layer3` recibe el DataFrame de capa 2 ya filtrado y dos umbrales:

- `min_trips_df2`: mínimo de viajes para considerar una combinación zona–hora–día fiable (por defecto 30).
- `min_trips_df3`: mínimo de viajes para estimar bien la variabilidad por zona–hora–servicio (por defecto 100).

A partir de esto genera cuatro DataFrames.

#### 2.1 `df_daily_service`: evolución diaria por servicio

Agrupa por `date` y `service_type` y calcula:

- `num_trips`: número de viajes ese día y servicio.
- `avg_price`: precio medio (`total_amount_std`).
- `std_price`: desviación estándar del precio.
- `unique_zones`: número de zonas de origen distintas (`pu_location_id`).

Sirve para ver:

- Volumen por día y tipo de servicio.
- Cómo cambian los precios medios a lo largo del tiempo.
- Qué tan disperso es el precio cada día.
- Cuánta variedad de zonas se activa cada día para cada servicio.

#### 2.2 `df_zone_hour_day_global`: hotspots zona–hora–día (global)

Agrupa por `pu_location_id`, `hour` y `date` (sin separar por servicio) y calcula:

- `num_trips`: viajes en esa zona–hora–día.
- `avg_price`: precio medio en esa combinación.
- `std_price`: desviación estándar del precio.

Después aplica un filtro:

- Solo conserva filas con `num_trips >= min_trips_df2` (por defecto, al menos 30 viajes).

Uso típico:

- Encontrar las combinaciones zona–hora–día con más actividad de taxi en general.
- Ver en qué momentos el precio medio es más alto.
- Detectar zonas/horas con comportamiento raro en precio.

#### 2.3 `df_zone_hour_day_service`: hotspots zona–hora–día por servicio

Similar a la anterior, pero separando por tipo de servicio:

- Agrupa por `pu_location_id`, `hour`, `date`, `service_type`.
- Calcula `num_trips`, `avg_price`, `std_price`.
- Aplica el mismo filtro de mínimo de viajes (`num_trips >= min_trips_df2`).

Esto permite:

- Comparar taxis tradicionales vs VTC en la misma zona–hora–día.
- Ver si los precios medios y la variabilidad difieren mucho por servicio.
- Detectar posiciones competitivas (ej. dónde un servicio domina en volumen).

#### 2.4 `df_variability`: variabilidad de precio (IQR) por zona–hora–servicio

Aquí se busca medir **qué tan impredecible** es el precio en una zona y hora para cada tipo de servicio.

Pasos:

1. Agrupa por `pu_location_id`, `hour`, `service_type`.
2. Calcula:
   - `num_trips`: número de viajes en esa zona–hora–servicio.
   - `avg_price`: precio medio.
   - `p75`: percentil 75 del precio.
   - `p25`: percentil 25 del precio (ambos con `percentile_approx`).
3. Define `price_variability` como `p75 - p25` (rango intercuartílico, IQR). Cuanto más alto, más dispersos los precios.
4. Filtra:
   - Solo se quedan combinaciones con `num_trips >= min_trips_df3` (por defecto, al menos 100 viajes), para asegurar que el IQR es fiable.
5. Elimina las columnas temporales `p75` y `p25`, dejando `price_variability` como métrica estable.
6. Calcula un **score de negocio** `biz_score`:
   - `biz_score = price_variability * log1p(num_trips)`.
   - La idea es combinar variabilidad alta y volumen alto.
   - `log1p` (logaritmo de `1 + num_trips`) hace que el volumen grande cuente, pero sin que eclipsa totalmente a la variabilidad.

Este DataFrame permite priorizar:

- Zonas/horas/servicios donde el precio es muy variable y hay mucho tráfico.
- Potenciales oportunidades de negocio (por ejemplo, segmentos donde un buen modelo de predicción de precio aporta más valor).

---

### 3. Guardado de la capa 3

La función `save_layer3` guarda cada DataFrame en formato Parquet bajo `data/aggregated` (ruta configurable con `out_base`):

- `df_daily_service`:
  - Ruta: `data/aggregated/df_daily_service`.
  - Particionado por `service_type`.
- `df_zone_hour_day_global`:
  - Ruta: `data/aggregated/df_zone_hour_day_global`.
  - Particionado por `date`.
- `df_zone_hour_day_service`:
  - Ruta: `data/aggregated/df_zone_hour_day_service`.
  - Particionado por `date` y `service_type`.
- `df_variability`:
  - Ruta: `data/aggregated/df_variability`.
  - Particionado por `service_type`.

Usa modo `overwrite` en todos los casos, reemplazando el contenido existente. Al final imprime un resumen con las rutas de salida.

---

### 4. Flujo completo (`main` de `capa3.py`)

La función `main` encadena todo el proceso:

1. Crea la sesión Spark con `get_spark`.
2. Lee y limpia la capa 2 con `read_layer2`, acotando fechas entre 2019-01-01 y 2024-03-01, y precio máximo 500.
3. Llama a `build_layer3` para obtener:
   - `df1` = `df_daily_service`.
   - `df2a` = `df_zone_hour_day_global`.
   - `df2b` = `df_zone_hour_day_service`.
   - `df3` = `df_variability`.
4. Si `DEBUG` está a `True`, muestra pequeñas muestras ordenadas para revisar a mano.
5. Guarda todos los DataFrames con `save_layer3`.
6. Cierra la sesión Spark.

Al ejecutar `python capa3.py`, se pone en marcha este flujo completo.

---

## 2. Script `inspect_capa3.py`: inspección de la capa 3

### Objetivo

Permitir una **revisión rápida y guiada** de los cuatro DataFrames agregados de capa 3:

- Ver esquema, conteos y muestras.
- Revisar rangos temporales y nulos.
- Detectar combinaciones con más viajes, precios medios más altos y mayor variabilidad.
- Validar que las horas están bien (0–23) y que no hay valores raros.

### Sesión de Spark y utilidades

- `get_spark`: crea una sesión similar a la de capa 3, con misma configuración de memoria, particiones y zona horaria.

Funciones auxiliares:

- `header(title)`: imprime un separador visual con el título, para organizar bien la salida en consola.
- `safe_exists(spark, path)`: intenta leer un Parquet en `path` y devuelve `True` si se puede leer al menos una fila, `False` si hay error (ruta no existe, esquema corrupto, etc.).
- `basic_profile(df, name)`:
  - Muestra:
    - Esquema (`printSchema`).
    - Conteo de filas.
    - Muestra de 10 filas completas.
- `null_profile(df, cols, name, sample_fraction)`:
  - Toma una muestra aleatoria (por defecto 0.1 % de las filas).
  - Calcula cuántos nulos hay por cada columna de `cols` que exista.
  - Muestra una única fila con esos contadores.
- `temporal_range(df, name)`:
  - Si existe `date`, muestra mínimo y máximo de fecha; si no, avisa.
- `by_service_counts(df, name)`:
  - Si existe `service_type`, muestra conteo de filas por servicio, ordenado de mayor a menor.

---

### 1. Inspector de `df_daily_service`

Función: `inspect_df_daily_service(df)`.

Flujo:

- Aplica `basic_profile` para ver esquema, número de filas y 10 ejemplos.
- Muestra el rango de fechas (si hay columna `date`) con `temporal_range`.
- Muestra conteos por servicio con `by_service_counts`.
- Muestra:
  - Top días por `num_trips` (días con más viajes).
  - Top días por `avg_price` (días más caros de media).
  - Top días por `std_price` (días con precio más disperso), si existe esa columna.
- Ejecuta `null_profile` sobre columnas clave:
  - `date`, `service_type`, `num_trips`, `avg_price`, `std_price`, `unique_zones`.

---

### 2. Inspector de `df_zone_hour_day_global`

Función: `inspect_df_zone_hour_day_global(df)`.

Flujo:

- Aplica `basic_profile` y `temporal_range`.
- Muestra:
  - Top combinaciones zona–hora–día por `num_trips` (más actividad).
  - Top combinaciones por `avg_price` (más caras).
  - Top combinaciones por `std_price` si la columna existe.
- Analiza la distribución de filas por `hour`:
  - `groupBy("hour").count()` ordenado por hora, para ver qué horas tienen más registros.
- Hace un sanity check de horas:
  - Muestra filas con `hour` fuera del rango [0, 23]. Si aparece algo aquí, es una señal de datos mal construidos.
- Ejecuta `null_profile` para:
  - `pu_location_id`, `hour`, `date`, `num_trips`, `avg_price`, `std_price` (las que existan).

---

### 3. Inspector de `df_zone_hour_day_service`

Función: `inspect_df_zone_hour_day_service(df)`.

Muy similar al anterior, pero teniendo en cuenta `service_type`:

- Aplica `basic_profile`, `temporal_range` y `by_service_counts`.
- Muestra:
  - Top combinaciones zona–hora–día–servicio por `num_trips`.
  - Top combinaciones por `avg_price`.
  - Top por `std_price` si existe.
- Sanity check de `hour` fuera de [0, 23].
- `null_profile` para:
  - `pu_location_id`, `hour`, `date`, `service_type`, `num_trips`, `avg_price`, `std_price`.

Esto ayuda a ver diferencias por servicio en las mismas franjas zona–hora–día.

---

### 4. Inspector de `df_variability`

Función: `inspect_df_variability(df)`.

Enfocado en la tabla de variabilidad (IQR):

- Aplica `basic_profile` y `by_service_counts` (si existe `service_type`).
- Muestra:
  - Top combinaciones por `price_variability` (zonas/horas donde el precio es más impredecible).
  - Top combinaciones por `avg_price` (más caras).
  - Top combinaciones por `num_trips` (más volumen).
  - Top combinaciones por `biz_score` (score de negocio), si la columna existe; si no, avisa de que puede que estés leyendo una versión antigua del DF.
- Sanity check:
  - Muestra filas con `price_variability < 0` (no deberían existir; si aparecen, algo está mal en el cálculo o en los datos).
- `null_profile` sobre:
  - `pu_location_id`, `hour`, `service_type`, `price_variability`, `avg_price`, `num_trips`, `biz_score`.

---

### 5. Flujo completo (`main` de `inspect_capa3.py`)

En `main`:

1. Crea la sesión Spark.
2. Define `base = "data/aggregated"` y cuatro rutas esperadas:
   - `df_daily_service`
   - `df_zone_hour_day_global`
   - `df_zone_hour_day_service`
   - `df_variability`.
3. Comprueba cada ruta con `safe_exists` y muestra si está OK o da error/no existe.
4. Para cada DF que exista:
   - Lo lee desde Parquet.
   - Llama a su función `inspect_df_*` correspondiente.
5. Cierra la sesión Spark.

Al ejecutar `python inspect_capa3.py` tendrás un informe por consola que te permite validar si la capa 3 tiene buena calidad antes de usarla en análisis o modelos.