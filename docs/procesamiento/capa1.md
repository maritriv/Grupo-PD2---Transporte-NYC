# Capa 1 — Validación inicial (Green, FHV, Yellow & FHVHV)

## Objetivo de la Capa 1

La **Capa 1** es una capa previa a la **Capa 2** cuyo objetivo es realizar un **control de calidad estructural y de plausibilidad fuerte** sobre los datos en bruto (*raw*), usando como referencia los **Data Dictionaries oficiales de TLC** y reglas mínimas de coherencia **temporal, espacial y numérica**.

Esta capa se centra en:

* Validar **tipos** (fechas, numéricos, enteros) y convertirlos a un esquema consistente.
* Validar **dominios/códigos** cuando el diccionario define valores permitidos.
* Detectar **errores críticos de coherencia temporal** (`dropoff < pickup`).
* Detectar **registros implausibles pero defendibles como erróneos**, por ejemplo:

  * fechas futuras
  * pickup fuera del mes esperado del fichero
  * duración imposible
  * velocidad implícita físicamente inverosímil
  * IDs de zona inexistentes (si se aporta catálogo TLC)

Separar datos en:

* **clean**: registros que pasan las validaciones críticas.
* **bad_rows** *(opcional)*: registros que fallan validaciones críticas.

Generar **reports en JSON** con contadores de errores y avisos por regla.

Añadir **trazabilidad**:

* `warning_reasons` en **clean**
* `rejection_reasons` en **bad_rows**

> **Nota:** La Capa 1 ya no se limita solo a validación estructural pura.
> En **Yellow, Green y FHVHV** incorpora también **reglas de plausibilidad fuerte**.
> Aun así, **no elimina automáticamente todos los outliers**: los casos raros pero no demostrablemente falsos se conservan como *warnings*.

---

# Ubicación de scripts

```
src/procesamiento/capa1/capa1_green.py
src/procesamiento/capa1/capa1_fhv.py
src/procesamiento/capa1/capa1_yellow.py
src/procesamiento/capa1/capa1_fhvhv.py
```

---

# Estructura de salida

Por defecto se escribe en:

## Green

```
data/validated/green/clean/
data/validated/green/bad_rows/ (si se activa)
data/validated/green/reports/
```

## FHV

```
data/validated/fhv/clean/
data/validated/fhv/bad_rows/ (si se activa)
data/validated/fhv/geo_ready/ (si se activa)
data/validated/fhv/reports/
```

## Yellow

```
data/validated/yellow/clean/
data/validated/yellow/bad_rows/ (si se activa)
data/validated/yellow/reports/
```

## FHVHV (HVFHS)

```
data/validated/fhvhv/clean/
data/validated/fhvhv/bad_rows/ (si se activa)
data/validated/fhvhv/reports/
```

Además, se genera un resumen global:

```
outputs/procesamiento/capa1_green/capa1_green_validation_summary.json
outputs/procesamiento/capa1_fhv/capa1_fhv_validation_summary.json
outputs/procesamiento/capa1_yellow/capa1_yellow_validation_summary.json
outputs/procesamiento/capa1_fhvhv/capa1_fhvhv_validation_summary.json
```

---

# 1) Capa 1 — GREEN (LPEP)

## Entrada esperada

Parquets en:

```
data/raw/green/green_tripdata_YYYY-MM.parquet
```

## Qué valida (Green)

Validación columna a columna basada en el **Data Dictionary de Green Trip Records (LPEP)**, ampliada con reglas de plausibilidad fuerte.

### Estandarización de columnas

* Renombrado a nombres canónicos **case-insensitive**
* Se aceptan **aliases históricos**
* Si falta una columna esperada → se crea como **NA**
* `cbd_congestion_fee` se trata como **opcional**

### Tipos / casts

**Datetimes**

```
lpep_pickup_datetime
lpep_dropoff_datetime
```

**Enteros nullable (Int64)**

```
VendorID
RatecodeID
PULocationID
DOLocationID
passenger_count
payment_type
trip_type
```

**Float64**

```
trip_distance
fare_amount
extra
mta_tax
tip_amount
tolls_amount
improvement_surcharge
total_amount
congestion_surcharge
cbd_congestion_fee
```

### Dominios (según TLC)

```
VendorID → {1,2,6}
RatecodeID → {1,2,3,4,5,6,99}
payment_type → {0,1,2,3,4,5,6}
trip_type → {1,2}
store_and_fwd_flag → {Y,N}
```

### Reglas temporales críticas

Invalid si:

* datetime inválido
* `dropoff < pickup`
* fechas futuras
* pickup fuera del mes esperado
* dropoff demasiado más allá del mes esperado

### Duración y velocidad implícita

Se calculan:

```
trip_duration_min
implied_speed_mph
```

Invalid si:

* duración negativa
* duración > 360 min
* velocidad > 100 mph

### Reglas numéricas

Invalid si:

```
trip_distance < 0
mta_tax < 0
improvement_surcharge < 0
congestion_surcharge < 0
cbd_congestion_fee < 0
```

### Location IDs

* validación como **Int64**
* opcional contra catálogo TLC (`--taxi-zones-csv`)

### Warnings

* `passenger_count = 0`
* `trip_distance = 0`
* `trip_duration = 0`
* `store_and_fwd_flag` missing
* `fare_amount` extremo
* `total_amount` extremo
* discrepancia componentes vs total

### Duplicados

Duplicados exactos → **invalid**

### Trazabilidad

* `warning_reasons` en **clean**
* `rejection_reasons` en **bad_rows**

### Cómo ejecutar

Procesar todo:

```
uv run python -m src.procesamiento.capa1.capa1_green --write-bad
```

Meses concretos:

```
uv run python -m src.procesamiento.capa1.capa1_green \
--months 2024-01 2024-02 \
--write-bad \
--taxi-zones-csv data/raw/taxi_zone_lookup.csv
```

---

# 2) Capa 1 — FHV

## Entrada

```
data/raw/fhv/fhv_tripdata_YYYY-MM.parquet
```

Este dataset **no contiene variables de tarifa**.

### Estandarización de columnas

Se normalizan nombres:

```
dispatching_base_num
pickup_datetime
dropOff_datetime
PUlocationID
DOlocationID
SR_Flag
Affiliated_base_number
```

### Validaciones críticas

**Timestamps**

* no nulos
* parseables a datetime

**Coherencia temporal**

```
dropOff_datetime < pickup_datetime → invalid
```

**Location IDs**

* casteo a **Int64**
* no numérico → invalid

### Warnings

* `PUlocationID` missing
* `DOlocationID` missing
* formato extraño en bases
* `SR_Flag` fuera de dominio

---

## geo_ready (MUY IMPORTANTE)

Subconjunto de **clean** con:

```
PUlocationID != NULL
DOlocationID != NULL
```

Permite:

* unir con **Taxi Zone Lookup**
* agregar por **zona / borough**

### Cuándo usar

**Temporal →** `clean`
**Espacial →** `geo_ready`

### Ejecución

```
uv run python -m src.procesamiento.capa1.capa1_fhv \
--months 2023-01 2025-10 \
--write-geo-ready
```

---

# 3) Capa 1 — YELLOW (TPEP)

## Entrada

```
data/raw/yellow/yellow_tripdata_YYYY-MM.parquet
```

### Tipos

Datetimes:

```
tpep_pickup_datetime
tpep_dropoff_datetime
```

Enteros:

```
VendorID
RatecodeID
PULocationID
DOLocationID
passenger_count
payment_type
```

Float64:

```
trip_distance
fare_amount
extra
mta_tax
tip_amount
tolls_amount
improvement_surcharge
total_amount
congestion_surcharge
airport_fee
cbd_congestion_fee
```

### Dominios

```
VendorID → {1,2,6,7}
RatecodeID → {1,2,3,4,5,6,99}
payment_type → {0..6}
store_and_fwd_flag → {Y,N}
```

### Reglas críticas

Invalid si:

* datetime inválido
* `dropoff < pickup`
* fechas futuras
* pickup fuera del mes
* dropoff demasiado lejos

### Velocidad / duración

Invalid si:

```
duración > 360 min
velocidad > 100 mph
```

### Numéricas

Invalid si:

```
trip_distance < 0
mta_tax < 0
improvement_surcharge < 0
congestion_surcharge < 0
airport_fee < 0
cbd_congestion_fee < 0
```

### Warnings

* `passenger_count = 0`
* `trip_distance = 0`
* `trip_duration = 0`
* flag missing
* fares extremos

### Ejecución

```
uv run python -m src.procesamiento.capa1.capa1_yellow --write-bad
```

Mes concreto:

```
uv run python -m src.procesamiento.capa1.capa1_yellow \
--months 2025-04 \
--write-bad
```

Con zonas:

```
uv run python -m src.procesamiento.capa1.capa1_yellow \
--months 2025-04 \
--write-bad \
--taxi-zones-csv data/raw/taxi_zone_lookup.csv
```

---

# 4) Capa 1 — FHVHV (HVFHS)

## Entrada

```
data/raw/fhvhv/fhvhv_tripdata_YYYY-MM.parquet
```

### Datetimes

```
request_datetime
on_scene_datetime
pickup_datetime
dropoff_datetime
```

### Enteros

```
PULocationID
DOLocationID
trip_time
```

### Float64

```
trip_miles
base_passenger_fare
tolls
bcf
sales_tax
congestion_surcharge
airport_fee
tips
driver_pay
cbd_congestion_fee
```

### Dominios

`hvfhs_license_num`

```
HVdddd
```

inválido → **invalid**

```
HV0002 HV0003 HV0004 HV0005
```

otros → **warning**

### Flags

```
shared_request_flag
shared_match_flag
access_a_ride_flag
wav_request_flag
wav_match_flag
```

missing → warning
valor ≠ {Y,N} → invalid

### Reglas temporales

Invalid:

```
dropoff < pickup
pickup futuro
dropoff futuro
pickup fuera del mes
```

Warnings:

```
request_datetime inválido
on_scene_datetime inválido
request > pickup
on_scene > pickup
```

### Duración y velocidad

Invalid:

```
duración negativa
duración > 360 min
velocidad > 100 mph
```

### Numéricas

Invalid:

```
trip_time < 0
trip_miles < 0
```

Warnings:

```
importes negativos
driver_pay extremo
fare extremo
```

### Duplicados

Duplicados exactos en batch → **invalid**

### Procesamiento por batches

Para evitar problemas de memoria:

* streaming por batches (`--batch-size`)
* lectura selectiva de columnas
* escritura incremental
* `gc.collect()`

### Ejecución

```
uv run python -m src.procesamiento.capa1.capa1_fhvhv \
--months 2024-02 \
--batch-size 200000
```

Con bad_rows:

```
uv run python -m src.procesamiento.capa1.capa1_fhvhv \
--months 2024-02 \
--batch-size 200000 \
--write-bad
```

Con validación de zonas:

```
uv run python -m src.procesamiento.capa1.capa1_fhvhv \
--months 2024-02 \
--batch-size 200000 \
--write-bad \
--taxi-zones-csv data/raw/taxi_zone_lookup.csv
```

---

# Resumen metodológico actual

En el estado actual del proyecto:

**Yellow, Green y FHVHV** usan una **Capa 1 ampliada** que combina:

* validación estructural
* validación de dominios
* plausibilidad temporal
* plausibilidad numérica fuerte
* warnings para anomalías no concluyentes
* trazabilidad por fila

**FHV** mantiene de momento una validación más básica centrada en:

* timestamps
* LocationIDs
* base numbers
* SR_Flag
* subconjunto `geo_ready`

La idea general es que **Capa 1 elimine lo objetivamente inválido o implausible**, mientras que los **casos raros pero no demostrablemente falsos se mantengan como warnings** para evitar **sobrelimpiar el dataset**.
