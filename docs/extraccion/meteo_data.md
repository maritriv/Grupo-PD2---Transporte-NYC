# Extracción de datos meteorológicos con Open-Meteo (Archive API)

Este módulo descarga datos horarios históricos de la API de **Open-Meteo Archive** (`/v1/archive`) y los guarda en formato Parquet, organizados por mes y rango de años.

---

## 1. ¿Qué datos meteorológicos se descargan?

La función principal de negocio es `download_meteo_aggregated`, que construye un dataset horario con las siguientes columnas:

- **date**: fecha en formato `YYYY-MM-DD` (extraída del timestamp de Open-Meteo).  
- **hour**: hora del día en formato entero `0–23`.  
- **temp_c**: temperatura del aire a 2 m en grados Celsius (variable `temperature_2m` de Open-Meteo).  
- **precip_mm**: precipitación total en milímetros (variable `precipitation`).  
- **rain_mm**: lluvia en milímetros (variable `rain`).  
- **snowfall_mm**: nieve caída en milímetros (variable `snowfall`).  
- **wind_kmh**: velocidad del viento a 10 m en km/h (variable `wind_speed_10m`).  
- **weather_code**: código meteorológico (tiempo dominante, por ejemplo, despejado, lluvia, nieve, etc.).

Los datos se obtienen de la API **Historical Weather / Archive** de Open-Meteo, que permite consultar variables horarias para un rango de fechas dado utilizando parámetros como `latitude`, `longitude`, `start_date`, `end_date`, `hourly` y `timezone`.

---

## 2. Estructura general del script

El archivo `download_meteo_data.py` está organizado en los siguientes bloques:

- **Configuración**
  - `BASE_URL = https://archive-api.open-meteo.com/v1/archive`: URL base de la API de Open-Meteo.  
  - `DEFAULT_OUT_DIR = obtener_ruta("data/external/meteo/raw")`: ruta por defecto donde se guardan los Parquet.  

- **Helpers Parquet**
  - `dataframe_to_parquet(df, parquet_path)`: convierte el `DataFrame` al esquema esperado y lo escribe en disco como Parquet usando PyArrow.  

- **Helpers Open-Meteo**
  - `_date_chunks(date_from, date_to, chunk_days=31)`: trocea el rango de fechas en ventanas de hasta 31 días.  
  - `_fetch_open_meteo_hourly(..)`: realiza una petición HTTP GET a la Archive API de Open-Meteo y devuelve el JSON con las series horarias.  

- **Lógica de negocio**
  - `download_meteo_aggregated(..)`: descarga y agrega datos horarios para un rango de fechas arbitrario y genera un Parquet temporal.  
  - `download_meteo_month(..)`: descarga un mes completo (unidad básica) y genera un Parquet por mes.  
  - `download_meteo_range(..)`: orquesta la descarga de varios años/meses llamando repetidamente a `download_meteo_month`.  

- **CLI (Click)**
  - Comando `main(..)` con opciones de línea de comandos para lanzar la descarga desde terminal.  

---

## 3. Flujo interno y comportamiento

### 3.1. Troceo de fechas y llamadas a la API

Para evitar rangos demasiado largos en una sola petición, el script divide el intervalo `[date_from, date_to]` en trozos de hasta **31 días** mediante `_date_chunks`.  
Cada subrango se consulta llamando a `_fetch_open_meteo_hourly`, que construye los parámetros de la URL con:

- `latitude`, `longitude`: coordenadas WGS84 del punto de interés.  
- `start_date`, `end_date`: fechas de inicio y fin en formato `YYYY-MM-DD`.  
- `timezone`: zona horaria en la que se devolverán los timestamps (por ejemplo `America/New_York`).  
- `hourly`: lista de variables horarias, unida como string separado por comas (por ejemplo `temperature_2m,precipitation,wind_speed_10m,..`).  

La API devuelve un objeto JSON con una sección `hourly` que contiene:

- `time`: lista de timestamps en formato ISO8601, normalmente `YYYY-MM-DDTHH:MM`.  
- Una lista por cada variable solicitada (misma longitud que `time`).  

El script recorre `time` posición a posición y, para cada índice `i`:

1. Extrae la fecha (`date = time[:10]`) y la hora (`hour = int(time[11:13])`).  
2. Utiliza `_safe_get` para obtener el valor `i` de cada variable, evitando errores si falta alguna posición.  
3. Construye un diccionario con todas las columnas y lo añade a la lista `rows`.  

Durante la descarga, se muestra una barra de progreso en consola (via `rich.Progress`) indicando el número de filas recopiladas.

### 3.2. Conversión a Parquet

Una vez recopiladas todas las filas, `download_meteo_aggregated` crea un `DataFrame` de pandas y llama a `dataframe_to_parquet`.  
En este helper se realiza:

- Conversión de `date` a tipo datetime (`pd.to_datetime`).  
- Conversión de `hour` a entero de 32 bits (`int32`).  
- Conversión segura a numérico de las columnas meteorológicas (`to_numeric(errors="coerce")`).  
- Escritura del `DataFrame` a disco con `df.to_parquet(.., engine="pyarrow", index=False)`.  

El resultado es un archivo Parquet listo para análisis posterior (por ejemplo, con pandas, Polars o motores de datos).

### 3.3. Descarga por mes y rango de años

- `download_meteo_month(year, month, ..)`:  
  - Calcula el primer y último día del mes (`calendar.monthrange`).  
  - Construye el nombre final: `meteo_{year}_{month:02d}.parquet` por defecto.  
  - Si el Parquet ya existe, lo marca como `skipped` y no vuelve a descargar.  
  - Si no existe, llama a `download_meteo_aggregated` para ese mes y renombra el archivo temporal a definitivo.  

- `download_meteo_range(start_year, end_year, start_month, end_month, ..)`:  
  - Rellena valores por defecto desde `meteo_config` si no se pasan `out_dir`, `latitude`, `longitude` o `timezone`.  
  - Recorre todas las combinaciones de años y meses con `itertools.product`.  
  - Lleva un contador de `ok`, `skipped`, `failed` y `total` para tener estadísticas de la extracción.  
  - Muestra en consola el resumen al final (descargados, omitidos, fallidos, total).  

---

## 4. Interfaz de línea de comandos (CLI)

El script expone una interfaz de línea de comandos usando `click` con el comando `main`.

### 4.1. Opciones disponibles

- `--start-year` (int, requerido): año inicial del rango.  
- `--end-year` (int, requerido): año final del rango.  
- `--start-month` (int, 1–12, default=1): mes inicial.  
- `--end-month` (int, 1–12, default=12): mes final.  
- `--latitude` (float, opcional): latitud; si se omite, se usa la configurada en `meteo_config`.  
- `--longitude` (float, opcional): longitud; si se omite, se usa la configurada en `meteo_config`.  
- `--timezone` (str, opcional): zona horaria; si se omite, se usa la configurada.  

### 4.2. Ejemplos de uso

Descargar todos los datos de 2024 (usando coordenadas y timezone por defecto de configuración):

```bash
uv run -m src.extraccion.download_meteo_data \
  --start-year 2024 \
  --end-year 2024
```

Descargar datos solo de junio a agosto de 2023:

```bash
uv run -m src.extraccion.download_meteo_data \
  --start-year 2023 \
  --end-year 2023 \
  --start-month 6 \
  --end-month 8
```

Descargar datos para unas coordenadas personalizadas (por ejemplo, Times Square):

```bash
uv run -m src.extraccion.download_meteo_data \
  --start-year 2024 \
  --end-year 2024 \
  --latitude 40.7580 \
  --longitude -73.9855
```