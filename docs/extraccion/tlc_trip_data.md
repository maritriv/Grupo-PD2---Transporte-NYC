# Documentación de datos NYC TLC y script de descarga
## Descripción general del dataset
La NYC Taxi & Limousine Commission (TLC) publica registros de viajes de taxis y vehículos de alquiler (FHV) con detalle a nivel de viaje individual en formato PARQUET. [github](https://github.com/awslabs/open-data-registry/blob/main/datasets/nyc-tlc-trip-records-pds.yaml)

Incluye principalmente cuatro tipos de servicios:

- **Yellow**: taxis amarillos (medallion taxis), que pueden recoger pasajeros en la calle (street‑hail).

- **Green**: taxis verdes (Street Hail Livery), autorizados a recoger en ciertas zonas y horarios fuera de Manhattan central.

​- **FHV**: For‑Hire Vehicles tradicionales (livery, black car, luxury limo), reportados como “FHV Trip Record Data”.

​- **HVFHV / fhvhv**: High Volume For‑Hire Vehicles (por ejemplo servicios tipo app), reportados desde 2019 en un dataset separado y más detallado. [nyc](https://www.nyc.gov/site/tlc/about/fhv-trip-record-data.page)

Los registros de yellow y green incluyen, entre otros, los siguientes campos típicos:

- Fechas y horas de recogida y destino.
- Identificadores de localización de recogida y destino (Taxi Zone Location ID).
- Distancia del viaje.
- Importe detallado de la tarifa (fare, extras, impuestos, propinas, peajes, recargos).
- Tipo de tarifa, método de pago.
- Número de pasajeros reportado por el conductor.

Desde 2025 se añade una columna `cbd_congestion_fee` en los datasets de Yellow, Green y High Volume FHV para reflejar los nuevos cargos de congestión en el centro de Manhattan.

Los datos se publican mensualmente, normalmente con un retraso de aproximadamente dos meses, y se almacenan en formato PARQUET debido a su tamaño.

La propia TLC advierte que los datos proceden de proveedores tecnológicos y que no puede garantizar su precisión o completitud, aunque realiza revisiones y acciones de supervisión.

---

## Cobertura temporal y disponibilidad
En la página oficial de TLC se listan datasets mensuales desde 2009 hasta el presente para taxis y FHV, con tablas por año y enlaces mensuales.

- Periodo aproximado: 2009–actualidad (según tipo de servicio y mes). [dss.princeton](https://dss.princeton.edu/catalog/resource3766)

- Los datos de 2020 pueden estar más incompletos en FHV por extensiones concedidas a pequeños operadores durante COVID‑19.

- La publicación se realiza mes a mes, con cierto retraso, por lo que meses muy recientes pueden no estar aún disponibles.

La cobertura exacta por servicio es:

- Yellow: datos desde 2009.

- Green: datos desde 2013, cuando arranca el programa de Street Hail Livery.

- FHV: datos históricamente separados en la página específica de FHV Trip Record Data.

- HVFHV: datos disponibles desde 2019 en un dataset separado.
---

## Ubicación real de los archivos PARQUET
Aunque los enlaces oficiales están en la web de NYC TLC, los archivos PARQUET se sirven a través de un bucket público en CloudFront (AWS), en la ruta:
```text
https://d37ci6vzurychx.cloudfront.net/trip-data/
```

Los ficheros siguen un patrón de nombre bien definido, por ejemplo: [hamedbh](https://www.hamedbh.com/posts/downloading-taxi-data/)

- Yellow: `yellow_tripdata_YYYY-MM.parquet`

- Green: `green_tripdata_YYYY-MM.parquet`

- FHV: `fhv_tripdata_YYYY-MM.parquet`

- HVFHV: `fhvhv_tripdata_YYYY-MM.parquet`

Ejemplos verificados de URLs válidas:

- `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet`

- `https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2013-08.parquet`

---

## Estructura del script de descarga
Archivo: `src/data/download_tlc_data.py` (Python, CLI con `click`).

### Configuración base
- `BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"`: apunta al bucket público de TLC donde están los PARQUET.

- `DEFAULT_DATA_DIR = ... / "data" / "raw"`: directorio por defecto para guardar los archivos.

---

### Construcción de URLs
```python
def build_url(service: str, year: int, month: int) -> str:
    """Construye la URL del archivo parquet según el formato TLC."""
    return f"{BASE_URL}/{service}_tripdata_{year}-{month:02d}.parquet"
```
- `service`: uno de `yellow`, `green`, `fhv`, `fhvhv`.

- `year`: año numérico (por ejemplo 2023).

- `month`: mes numérico (1–12) formateado a dos dígitos (`01`, `02`, …).

---

### Descarga de un archivo

Firma:
```python
def download_file(url: str, dest_path: Path) -> bool:
```
Comportamiento:
- Si el archivo de destino ya existe, lo omite y devuelve `True` (se cuenta como “descargado” para no fallar el proceso).

- Hace una petición `GET` con `stream=True` y `timeout=30` usando requests.

- Si `status_code == 200`:

    - Crea directorios si no existen.

    - Si el servidor envía cabecera content-length, usa rich.progress.Progress para mostrar barra de progreso, volumen descargado y velocidad.

    - Si no hay content-length, descarga en chunks sin barra de progreso.

- Si el código HTTP no es 200:

    - Muestra mensaje `ERROR <status>` con descripción legible `(get_http_error_description)`.

    - Devuelve `False`.

- Maneja de forma explícita:

    - `requests.exceptions.Timeout`: tiempo de espera agotado (>30s).

    - `requests.exceptions.ConnectionError`: problemas de red/conexión.

    - `requests.exceptions.RequestException`: otros errores de requests.

    - `IOError`: problemas al guardar el archivo en disco.

    - `Exception`: cualquier error inesperado.

---

### Descarga por servicio y rango

Firma:
```python
def download_service_data(
    service: str,
    years: range,
    months: range,
    data_dir: Optional[Path] = None
):
```
Funcionamiento:

1. Si `data_dir` es `None`, usa `DEFAULT_DATA_DIR`.

2. Crea un subdirectorio específico para el servicio (`data/raw/yellow`, `data/raw/green`, etc.).

3. Genera todas las combinaciones `(year, month)` con `itertools.product`.

4. Recorre cada combinación:

    - Construye la URL con `build_url`.

    - Define el nombre de archivo `<service>_tripdata_<YYYY>-<MM>.parquet`.

    - Si el archivo ya existe, lo marca como skipped.

    - Si no, llama a download_file y cuenta successful o failed.

5. Imprime un resumen final con:

    - Total de archivos a procesar.

    - Descargas correctas.

    - Fallos.

    - Archivos omitidos (ya existentes).

6. Devuelve un diccionario con estadísticas de ejecución.

---

### Ejemplos de ejecución y salida esperada

#### Ejemplo 1: descargar yellow 2023–2025

Comando:
```bash
uv run -m src.data.download_tlc_data --service yellow
```

Equivalente a:
```bash
uv run -m src.data.download_tlc_data -s yellow --start-year 2023 --end-year 2025 --start-month 1 --end-month 12
```

Al final imprime un resumen, por ejemplo:
```text
Completado: 34/36 archivos descargados correctamente
ADVERTENCIA: 2 archivo(s) no se pudieron descargar
```

#### Ejemplo 2: descargar sólo green 2024
```bash
uv run -m src.data.download_tlc_data -s green --start-year 2024 --end-year 2024
```

#### Ejemplo 3: HVFHV verano 2023 a carpeta custom
```bash
uv run -m src.data.download_tlc_data \
  -s fhvhv \
  --start-year 2023 --end-year 2023 \
  --start-month 6 --end-month 8 \
  --data-dir /tmp/nyc_tlc
```