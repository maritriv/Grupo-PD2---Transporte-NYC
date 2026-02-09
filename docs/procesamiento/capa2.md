# Capa 2 de taxis NYC

Este pequeño módulo define un flujo en dos pasos:

1. Construir una capa de datos estandarizada (`capa2.py`) a partir de los datos RAW de taxis (yellow, green y fhvhv).
2. Inspeccionar de forma rápida la calidad y el rango de esa capa (`inspect_capa2.py`).

Los scripts están pensados para ejecutarse con Spark en local (`master("local[*]")`) y volúmenes grandes de datos (decenas de millones de filas).

---

## 1. Script `capa2.py`: construcción de la capa estandarizada

### Objetivo

Crear una tabla de viajes de taxi con un **esquema común** para distintos servicios (yellow, green, fhvhv), donde:

- Las fechas y horas de recogida/fin estén normalizadas.
- El precio total sea comparable entre servicios aunque las columnas tengan nombres distintos.
- Existan columnas derivadas útiles para análisis (año, mes, día de la semana, duración del viaje, etc.).
- Se incorporen las zonas TLC (borough y zone) tanto de origen como de destino.
- La salida se guarde en formato Parquet, particionada por año, mes y tipo de servicio.

### Componentes principales

#### 1. Sesión de Spark

La función `get_spark`:

- Crea una sesión de Spark con nombre de aplicación `"PD2-Capa2"`.
- Usa el modo local con todos los núcleos disponibles (`local[*]`).
- Ajusta memoria del driver y executor a 6 GB.
- Configura el número de particiones de shuffle en 200.
- Fija la zona horaria a `"America/New_York"`.
- Reduce el nivel de logs a `WARN` para evitar ruido.

#### 2. Lectura de la capa RAW

La función `read_raw_services`:

- Busca ficheros Parquet en `data/raw/{yellow,green,fhvhv}` (configurable con `base_path` y `services`).
- Para cada servicio:
  - Lista los `.parquet`; si no hay, muestra un aviso y lo salta.
  - Lee todos los Parquet de ese servicio en un DataFrame.
  - Añade una columna `service_type` con el nombre del servicio (yellow, green, fhvhv).
  - Si está activado `DEBUG`, muestra el esquema y una pequeña muestra.
- Une todos los DataFrames de servicios con `unionByName`, permitiendo columnas ausentes según el servicio.
- Si no encuentra ningún fichero, lanza un error.

Resultado: un único DataFrame RAW con todos los viajes y una columna `service_type` que distingue el origen del dato.

#### 3. Construcción de la Capa 2

La función `build_layer2` aplica varias transformaciones sobre el DataFrame RAW:

##### a) Normalización de timestamps

- Construye `pickup_dt` como la primera columna no nula entre `tpep_pickup_datetime`, `lpep_pickup_datetime` y `pickup_datetime`.
- Construye `dropoff_dt` de forma equivalente para la hora de llegada.
- Crea columnas:
  - `pickup_datetime` y `dropoff_datetime` en tipo `timestamp`.
  - `pu_location_id` y `do_location_id` como enteros a partir de `PULocationID` y `DOLocationID`.

##### b) Precio total estandarizado

Los distintos servicios usan columnas de precios distintas. Para que el total sea comparable:

- `airport_fee_any`: mezcla `airport_fee`, `Airport_fee` o 0 si no existen.
- `tips_any`: mezcla `tip_amount` o `tips`, o 0.
- `tolls_any`: mezcla `tolls_amount` o `tolls`, o 0.
- `total_amount_std`:
  - Si existe `total_amount`, la usa.
  - Si no, suma:
    - `base_passenger_fare` (si existe, si no 0),
    - propinas,
    - peajes,
    - tasa de aeropuerto,
    - `congestion_surcharge` (si existe, si no 0).
- El resultado se guarda en una columna numérica `total_amount_std`.

##### c) Variables de fecha y duración del viaje

Sobre `pickup_datetime`:

- Crea columnas:
  - `date` (solo fecha),
  - `year`, `month`, `hour`,
  - `day_of_week` (día de la semana),
  - `is_weekend` (1 si es fin de semana, 0 en otro caso),
  - `week_of_year`.
- Calcula `trip_duration_min`:
  - Diferencia en minutos entre `dropoff_datetime` y `pickup_datetime` cuando ambos existen.
  - Si la duración es negativa o mayor de 360 minutos (6 horas), la marca como nula.

Si `DEBUG` está activo, muestra una previsualización con las columnas clave de Capa 2.

#### 4. Enriquecimiento con zonas TLC

La función `add_zone_lookup`:

- Lee el fichero CSV `data/external/taxi_zone_lookup.csv` con las zonas TLC.
- Selecciona y renombra:
  - `LocationID` → `location_id` (int),
  - `Borough` → `borough`,
  - `Zone` → `zone`.
- Hace dos joins usando broadcast:
  - Con `pu_location_id` para añadir `pu_borough` y `pu_zone`.
  - Con `do_location_id` para añadir `do_borough` y `do_zone`.
- Si algo falla (por ejemplo, falta el CSV), informa del motivo y devuelve el DataFrame original sin zonas.
- Si `DEBUG` está activo, enseña una muestra con las zonas de origen/destino.

#### 5. Selección de columnas finales

La función `select_layer2_columns` define un **esquema canónico**:

- Columnas obligatorias:
  - `service_type`, `pickup_datetime`, `dropoff_datetime`,
  - `date`, `year`, `month`, `hour`, `day_of_week`, `is_weekend`, `week_of_year`,
  - `trip_duration_min`,
  - `pu_location_id`, `do_location_id`,
  - `pu_borough`, `pu_zone`, `do_borough`, `do_zone`,
  - `total_amount_std`.
- Columnas opcionales (si existen en los datos):
  - Distintos componentes de precio y otros campos originales: `total_amount`, `fare_amount`, `extra`, `mta_tax`, `tip_amount`, `tips`, `tolls_amount`, `tolls`, `improvement_surcharge`, `congestion_surcharge`, `Airport_fee`, `airport_fee`, `sales_tax`, `bcf`, `driver_pay`, `base_passenger_fare`.
  - Información de distancia y tiempo: `trip_distance`, `trip_miles`, `trip_time`.
  - Metadatos: `VendorID`, `passenger_count`, `RatecodeID`, `payment_type`, `store_and_fwd_flag`, `trip_type`, `ehail_fee`, `hvfhs_license_num`, `dispatching_base_num`, `originating_base_num`, `request_datetime`, `on_scene_datetime`, `shared_request_flag`, `shared_match_flag`, `access_a_ride_flag`, `wav_request_flag`, `wav_match_flag`.
- Se queda solo con las columnas que realmente existen en el DataFrame.
- Si `DEBUG` está activo, informa de qué columnas faltan y cuántas conserva.

#### 6. Guardado de la capa

La función `save_layer2`:

- Escribe el DataFrame final como Parquet en `data/standarized` (ruta configurable).
- Usa modo `overwrite` para reemplazar el contenido anterior.
- Particiona por `year`, `month` y `service_type` para facilitar consultas y filtrado.
- Muestra un mensaje de éxito con la ruta.

#### 7. Flujo completo (`main`)

La función `main` orquesta todo:

1. Crea la sesión de Spark.
2. Lee los datos RAW con `read_raw_services`.
3. Construye la capa 2 con `build_layer2`.
4. Añade las zonas TLC con `add_zone_lookup`.
5. Selecciona las columnas finales con `select_layer2_columns`.
6. Guarda el resultado con `save_layer2`.
7. Cierra la sesión de Spark.

---

## 2. Script `inspect_capa2.py`: inspección rápida de la Capa 2

### Objetivo

Dar una **vista rápida de calidad y rango** de la capa estandarizada ya construida:

- Ver el esquema y las columnas.
- Revisar distribución temporal (mínimo/máximo de fechas).
- Hacer una muestra pequeña para inspección manual.
- Calcular nulos sobre columnas clave.
- Hacer sanity checks sencillos sobre precio, duración y hora.
- Obtener un mini perfil por tipo de servicio.

### Parámetros y constantes

En la cabecera se definen:

- `LAYER2_PATH = "data/standarized"`: ruta donde se lee la capa 2.
- `SAMPLE_FRACTION = 0.0005`: fracción de filas para la muestra (0.05 %). Para 45 millones de filas, unas 22 000.
- `SEED = 42`: semilla para muestreo reproducible.
- `MIN_DATE_EXPECTED = "2019-01-01"` y `MAX_DATE_EXPECTED = "2024-03-01"`:
  - Rango de fechas que se considera razonable; todo lo que está fuera se reporta.
- `CAP_MAX_PRICE_INSPECT = 500.0`: precio máximo considerado normal a efectos de inspección (outliers por encima se cuentan, no se filtran).

### 1. Sesión de Spark

La función `get_spark`:

- Crea una sesión Spark con nombre `"Inspect-Capa2"` en modo local.
- Fija la zona horaria a `"America/New_York"`.
- Ajusta logs a nivel `WARN`.

### 2. Lectura de la capa y esquema

En `main`:

- Se crea la sesión Spark.
- Se lee el Parquet de `LAYER2_PATH` en un DataFrame `df`.
- Se muestra un mensaje indicando la ruta usada.

A continuación:

- Se imprime el **schema** completo (`df.printSchema()`).
- Se listan las columnas ordenadas alfabéticamente.

### 3. Conteos básicos

El script calcula:

- **Conteo por tipo de servicio** (`service_type`), si existe:
  - Agrupa por `service_type`, cuenta filas y ordena de mayor a menor.
- **Rango temporal de `pickup_datetime`**, si la columna existe:
  - Mínimo y máximo de `pickup_datetime`.
- **Rango temporal de `date`**, si la columna existe:
  - Mínimo y máximo de `date`.
  - Número de filas fuera del rango esperado `[MIN_DATE_EXPECTED, MAX_DATE_EXPECTED]` y porcentaje sobre el total.

Si alguna columna falta, imprime un aviso y pasa al siguiente bloque.

### 4. Muestra para inspección manual

- Toma una muestra del DataFrame con `sample(fraction=SAMPLE_FRACTION, seed=SEED)`.
- Define un conjunto de columnas clave (si existen): servicio, fechas, hora, indicadores temporales, zona de recogida/destino, precio estándar, duración y año/mes.
- Muestra las primeras 10 filas de la muestra, con esas columnas si están disponibles.

Esto permite hacer una revisión rápida visual de que los datos “tienen buena pinta”.

### 5. Nulos y chequeos básicos (sobre la muestra)

#### Nulos

- Define una lista de columnas importantes: `pickup_datetime`, `date`, `hour`, `pu_location_id`, `do_location_id`, `total_amount_std`, `trip_duration_min`, `service_type`, `pu_borough`, `pu_zone`.
- Para cada columna que exista:
  - Calcula el número de valores nulos en la muestra.
- Muestra una única fila con todos esos contadores.

Si ninguna de esas columnas está presente, lo indica por pantalla.

#### Sanity checks

Construye expresiones de control según las columnas presentes:

- Sobre `total_amount_std`:
  - Número de filas con precio menor o igual que 0.
  - Número de filas con precio mayor que `CAP_MAX_PRICE_INSPECT` (posibles outliers).
  - Mediana, percentil 95 y percentil 99 de `total_amount_std`.
- Sobre `trip_duration_min`:
  - Número de duraciones negativas.
  - Número de duraciones mayores de 360 minutos.
  - Mediana y percentil 95 de `trip_duration_min`.
- Sobre `hour`:
  - Número de registros con `hour` fuera del rango 0–23.

Muestra una única fila con todos estos indicadores.  
Si no puede calcular ninguno porque faltan columnas, lo avisa.

### 6. Mini perfil por servicio (sobre la muestra)

Si existen `service_type` y `total_amount_std`:

- Agrupa por `service_type` y calcula:
  - `n`: número de filas de muestra.
  - `avg_total_amount_std`: precio medio.
  - `median_total_amount_std`: mediana.
  - `p95_total_amount_std`: percentil 95.
  - `n_le_0`: número de filas con precio menor o igual que 0.
  - `n_gt_cap`: número de filas con precio mayor que `CAP_MAX_PRICE_INSPECT`.
- Ordena los servicios de mayor a menor número de filas.

Esto da una idea rápida de cómo se comporta el precio por tipo de servicio.

### 7. Conteo total

Finalmente:

- Imprime el número total de filas en la capa 2 (`df.count()`), avisando de que puede tardar.
- Cierra la sesión de Spark.

Al ejecutar el script directamente (`python inspect_capa2.py`), se llama a `main()`.