# Capa 1 — Validación inicial (Green, FHV, Yellow & FHVHV)

## Objetivo de la Capa 1
La **Capa 1** es una capa previa a la Capa 2 cuyo objetivo es realizar un **control de calidad estructural** sobre los datos en bruto (raw), usando como referencia los **Data Dictionaries oficiales de TLC**.

Esta capa se centra en:
- Validar **tipos** (fechas, numéricos, enteros) y convertirlos a un esquema consistente.
- Validar **dominios/códigos** cuando el diccionario define valores permitidos (p. ej. `VendorID`, `RatecodeID`, etc.).
- Detectar errores críticos de coherencia temporal (`dropoff < pickup`).
- Separar datos en:
  - **clean**: registros que pasan las validaciones.
  - **bad_rows** (opcional): registros que fallan validaciones críticas.
- Generar **reports** en JSON con contadores de errores/avisos por regla.

> Nota: La Capa 1 **no** aplica (por defecto) limpieza de outliers de negocio (distancias extremadamente altas, importes anómalos, etc.). Eso se puede añadir más adelante si hace falta (Capa 2 o reglas extra en Capa 1).

---

## Ubicación de scripts
- `src/procesamiento/capa1/capa1_green.py`
- `src/procesamiento/capa1/capa1_fhv.py`
- `src/procesamiento/capa1/capa1_yellow.py`
- `src/procesamiento/capa1/capa1_fhvhv.py`

---

## Estructura de salida
Por defecto se escribe en:

- **Green**
  - `data/validated/green/clean/`
  - `data/validated/green/bad_rows/` (si se activa)
  - `data/validated/green/reports/`

- **FHV**
  - `data/validated/fhv/clean/`
  - `data/validated/fhv/bad_rows/` (si se activa)
  - `data/validated/fhv/geo_ready/` (si se activa)
  - `data/validated/fhv/reports/`

- **Yellow**
  - `data/validated/yellow/clean/`
  - `data/validated/yellow/bad_rows/` (si se activa)
  - `data/validated/yellow/reports/`

- **FHVHV (HVFHS)**
  - `data/validated/fhvhv/clean/`
  - `data/validated/fhvhv/bad_rows/` (si se activa)
  - `data/validated/fhvhv/reports/`

Además, se genera un resumen global:
- `outputs/procesamiento/capa1_green/...json`
- `outputs/procesamiento/capa1_fhv/capa1_fhv_validation_summary.json`
- `outputs/procesamiento/capa1_yellow/...json`
- `outputs/procesamiento/capa1_fhvhv/capa1_fhvhv_validation_summary.json`

---

# 1) Capa 1 — GREEN (LPEP)

## Entrada esperada
Parquets en:
- `data/raw/green/green_tripdata_YYYY-MM.parquet`

## Qué valida (Green)
Validación columna a columna basada en el Data Dictionary de **Green Trip Records (LPEP)**:

### Tipos / casts
- `lpep_pickup_datetime`, `lpep_dropoff_datetime` → `datetime`
- IDs y contadores → enteros nullable (`Int64`)
  - `VendorID`, `RatecodeID`, `PULocationID`, `DOLocationID`, `passenger_count`, `payment_type`, `trip_type`
- Importes / medidas → `float`
  - `trip_distance`, `fare_amount`, `extra`, `mta_tax`, `tip_amount`, `tolls_amount`,
    `improvement_surcharge`, `total_amount`, `congestion_surcharge`, `cbd_congestion_fee`

### Dominios (según diccionario TLC)
- `VendorID` permitido: `{1, 2, 6}`
- `RatecodeID` permitido: `{1, 2, 3, 4, 5, 6, 99}`
- `payment_type` permitido: `{0, 1, 2, 3, 4, 5, 6}`
- `trip_type` permitido: `{1, 2}`
- `store_and_fwd_flag` permitido: `{"Y", "N"}` (normalizado a mayúsculas)

### Coherencia temporal
- Se marca como inválido si:  
  `lpep_dropoff_datetime < lpep_pickup_datetime`

### Consideración sobre `cbd_congestion_fee`
- El campo puede existir pero venir vacío en años anteriores a 2025.
- Se trata como columna **opcional** (si falta, se crea como NA para mantener esquema).

## Cómo ejecutar
Procesar todo lo que haya en `data/raw/green/`:

```bash
uv run python -m src.procesamiento.capa1.capa1_green --write-bad
```

---

# 2) Capa 1 — FHV

## Entrada esperada
Parquets en:
- `data/raw/fhv/fhv_tripdata_YYYY-MM.parquet`

> Nota: El dataset FHV **no contiene campos de tarifa/precio** como taxis, por lo que la validación se centra en tiempo, localización y metadatos del viaje.

---

## Qué valida (FHV)

### Estandarización de nombres de columnas
En FHV pueden variar mayúsculas/minúsculas (`dropOff_datetime` vs `dropoff_datetime`, etc.).  
La Capa 1 renombra columnas automáticamente a los nombres canónicos:

- `dispatching_base_num`
- `pickup_datetime`
- `dropOff_datetime`
- `PUlocationID`
- `DOlocationID`
- `SR_Flag`
- `Affiliated_base_number`

Esto evita errores por diferencias de capitalización entre ficheros/meses.

---

### Validaciones críticas (invalid → van a bad_rows)
Estas validaciones determinan si una fila entra en `clean` o se descarta como `bad_rows`:

#### 1) Timestamps
- `pickup_datetime` y `dropOff_datetime`:
  - deben existir (no nulos)
  - deben parsear correctamente a `datetime`

#### 2) Coherencia temporal
- Se marca como inválido si:
  - `dropOff_datetime < pickup_datetime`

#### 3) Location IDs (tipo)
- `PUlocationID` y `DOlocationID`:
  - se castean a entero nullable (`Int64`)
  - se consideran **inválidos** si el valor:
    - no es numérico, o
    - trae decimales (ej. `123.5`)

> Importante: **por defecto no se considera inválido** que falten `PUlocationID` o `DOlocationID`, porque en FHV muchos registros vienen sin LocationID. En ese caso se registran como *warning*.

---

### Warnings (no eliminan filas)
Se reportan en `warning_counts`, pero **no expulsan filas** del dataset `clean`:

- `PUlocationID` missing (por defecto)
- `DOlocationID` missing (por defecto)
- formato no estándar de:
  - `dispatching_base_num`
  - `Affiliated_base_number`  
  (patrón típico TLC: `B00013`)
- `SR_Flag` fuera de dominio (se espera `NULL` o `1`)

---

## `geo_ready` (MUY IMPORTANTE)

### ¿Qué es?
`geo_ready` es un subconjunto de `clean` que contiene únicamente filas con:

- `PUlocationID` **no nulo**
- `DOlocationID` **no nulo**

### ¿Por qué es importante?
Solo con `PUlocationID`/`DOlocationID` se puede:
- unir con el **Taxi Zone Lookup**
- asignar **zona / borough**
- generar agregaciones espaciales (por borough/zona)

En la práctica, muchos registros FHV no traen LocationID, por eso `geo_ready` suele ser bastante más pequeño que `clean`.

### Cuándo usar `clean` vs `geo_ready`
- Si el análisis es **temporal** (viajes por hora/día, estacionalidad) → usar `clean`
- Si el análisis requiere **geografía** (por borough/zona) → usar `geo_ready`

---

## Cómo ejecutar (FHV)

### Modo recomendado (no sangra y permite geo_ready)
Genera `clean` casi completo y opcionalmente `geo_ready`:

```bash
uv run python -m src.procesamiento.capa1.capa1_fhv --months 2023-01 2025-10 --write-geo-ready
```

---

# 3) Capa 1 — YELLOW (TPEP)

## Entrada esperada
Parquets en:
- `data/raw/yellow/yellow_tripdata_YYYY-MM.parquet`

## Qué valida (Yellow)
Validación columna a columna basada en el Data Dictionary de **Yellow Trip Records (TPEP)**.  
La lógica es equivalente a Green, pero usando los nombres de columnas propios de Yellow.

### Tipos / casts
- `tpep_pickup_datetime`, `tpep_dropoff_datetime` → `datetime`
- IDs y contadores → enteros nullable (`Int64`)
  - `VendorID`, `RatecodeID`, `PULocationID`, `DOLocationID`, `passenger_count`, `payment_type`
- Importes / medidas → `float`
  - `trip_distance`, `fare_amount`, `extra`, `mta_tax`, `tip_amount`, `tolls_amount`,
    `improvement_surcharge`, `total_amount`, `congestion_surcharge`, `airport_fee`, `cbd_congestion_fee`

### Dominios (según diccionario TLC)
- `VendorID` permitido: `{1, 2, 6}`
- `RatecodeID` permitido: `{1, 2, 3, 4, 5, 6, 99}`
- `payment_type` permitido: `{0, 1, 2, 3, 4, 5, 6}`
- `store_and_fwd_flag` permitido: `{"Y", "N"}` (normalizado a mayúsculas)

### Coherencia temporal
- Se marca como inválido si:  
  `tpep_dropoff_datetime < tpep_pickup_datetime`

### Consideración sobre `cbd_congestion_fee`
- El campo puede existir pero venir vacío en años anteriores a 2025.
- Se trata como columna **opcional** (si falta, se crea como NA para mantener esquema).

## Cómo ejecutar (Yellow)
Procesar todo lo que haya en `data/raw/yellow/`:

```bash
uv run python -m src.procesamiento.capa1.capa1_yellow --write-bad
```

Procesar meses concretos:

```bash
uv run python -m src.procesamiento.capa1.capa1_yellow --months 2025-04 --write-bad
```

---

# 4) Capa 1 — FHVHV (HVFHS)

## Entrada esperada
Parquets en:
- `data/raw/fhvhv/fhvhv_tripdata_YYYY-MM.parquet`

## Qué valida (FHVHV)
Validación estructural basada en el Data Dictionary de **High Volume FHV (HVFHS)**.  
A diferencia de FHV, este dataset incluye variables de tarifa (p. ej. `base_passenger_fare`, `tips`, `driver_pay`) y flags de servicio.

### Tipos / casts
- Datetimes → `datetime`
  - `request_datetime`, `on_scene_datetime`, `pickup_datetime`, `dropoff_datetime`
- Location IDs y tiempos → enteros nullable (`Int64`)
  - `PULocationID`, `DOLocationID`, `trip_time`
- Distancias e importes → `float`
  - `trip_miles`, `base_passenger_fare`, `tolls`, `bcf`, `sales_tax`, `congestion_surcharge`,
    `airport_fee`, `tips`, `driver_pay`, `cbd_congestion_fee`

### Dominios / formatos
- `hvfhs_license_num`: formato `HVdddd` (ej. `HV0003`)
  - formato inválido → invalid (bad_rows)
  - código fuera del set conocido `{HV0002, HV0003, HV0004, HV0005}` → warning (pueden aparecer nuevos)
- `dispatching_base_num` y `originating_base_num`:
  - se validan con patrón típico TLC (ej. `B00013`)
  - missing o formato no estándar → warning
- Flags (`Y/N`):
  - `shared_request_flag`, `shared_match_flag`, `access_a_ride_flag`, `wav_request_flag`, `wav_match_flag`
  - missing → warning
  - valor fuera de `{Y, N}` → invalid (bad_rows)

### Coherencia temporal
- Se marca como inválido si:  
  `dropoff_datetime < pickup_datetime`

Además, se registran warnings (no eliminan filas) si:
- `request_datetime > pickup_datetime`
- `on_scene_datetime > pickup_datetime`
- `on_scene_datetime < request_datetime`

### Reglas adicionales
- `trip_time < 0` → invalid (bad_rows)
- `trip_miles < 0` → invalid (bad_rows)
- importes negativos (ajustes/refunds) → warning
- `cbd_congestion_fee` puede estar vacío antes de 2025 → warning (no invalid)

## Ejecución optimizada (batches)
Los ficheros FHVHV suelen ser muy grandes. Para evitar cargar un mes completo en memoria:
- Se procesa en streaming por batches (`--batch-size`).
- Se leen solo las columnas necesarias del parquet (según `EXPECTED_COLUMNS`).
- Se escribe con `ParquetWriter` (evita concatenaciones grandes en memoria).

## Cómo ejecutar (FHVHV)
Procesar un mes con batch controlado:

```bash
uv run python -m src.procesamiento.capa1.capa1_fhvhv --months 2024-02 --batch-size 200000
```

Si se quieren guardar `bad_rows`:

```bash
uv run python -m src.procesamiento.capa1.capa1_fhvhv --months 2024-02 --batch-size 200000 --write-bad
```
