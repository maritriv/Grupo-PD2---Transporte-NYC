# Documentación de datos: `download_events_data.py`

## Origen de los datos

- Fuente: plataforma NYC Open Data, expuesta mediante Socrata.  
- Acceso: se usa la API de Socrata sobre un dataset configurable mediante su `dataset_id` (por defecto el configurado en `eventos_config["dataset_id"]`).  
- URL base: se toma de `eventos_config["url_base"]`, `https://data.cityofnewyork.us`.  

Para entender el dataset concreto (campos originales, definiciones, etc.) se obtienen primero sus metadatos con una petición HTTP a `{BASE_URL}/api/views/{dataset_id}`.  

## Qué datos se extraen

El script no descarga todos los campos originales, sino un agregado por fecha, hora, borough y tipo de evento.  
A partir del campo de fecha/hora de inicio del evento en el dataset (detectado automáticamente) se construye una tabla diaria y horaria.  

Las columnas del resultado final son:  

- `date`: fecha del evento truncada a día (se aplica `date_trunc_ymd` sobre el campo de inicio original).  
- `hour`: hora del día (0–23) extraída del campo de inicio (`date_extract_hh`).  
- `borough`: borough asociado al evento; el script intenta detectar la columna correspondiente buscando nombres tipo `"borough"` en los metadatos.  
- `event_type`: tipo o nombre del evento; se busca un campo candidato entre `"event_type"`, `"event"`, `"name"`, y en su defecto se usa `"event_name"`.  
- `n_events`: número de eventos (conteo de filas) que cumplen la combinación `date, hour, borough, event_type`.  

La consulta a Socrata se construye con cláusulas `$select`, `$where`, `$group` y `$order` sobre la API tipo `/resource/{dataset_id}.json`:  

- `$select`: agrega y renombra los campos a `date, hour, borough, event_type, n_events`.  
- `$where`: filtra por rango de fechas `date_from`–`date_to` en el campo de inicio detectado.  
- `$group`: agrupa por fecha truncada, hora, borough y tipo de evento.  
- `$order`: ordena por `date asc, hour asc`.  

## Estructura y formato de salida

Para cada mes solicitado se genera un fichero Parquet con datos agregados a nivel diario-horario.  

- Ruta de salida por defecto: `data/external/events`, obtenida con `obtener_ruta("data/external/events")`.  
- Un fichero por mes: `events_{year}_{month:02d}.parquet` (por ejemplo, `events_2024_06.parquet`).  
- Esquema Parquet (tras conversión y tipado en Pandas):  
  - `date`: tipo datetime.  
  - `hour`: entero (`int32`).  
  - `borough`: texto.  
  - `event_type`: texto.  
  - `n_events`: entero (`int32`).  

El proceso intermedio escribe primero un CSV temporal con cabecera `date,hour,borough,event_type,n_events`, lo convierte a Parquet mediante `pandas.to_parquet(engine="pyarrow")` y luego borra el CSV.  

## Cómo obtener más información sobre el dataset

Si se necesita conocer todos los campos originales, definiciones oficiales o documentación de la ciudad:  

1. Consultar la página del dataset en NYC Open Data usando el `dataset_id` que se pasa al script (por defecto `DEFAULT_DATASET`).  
2. Revisar los metadatos devueltos por `BASE_URL/api/views/{dataset_id}` (títulos de columnas, tipos, descripciones, etc.).  
3. Ajustar la lógica de extracción (por ejemplo, los candidatos de `_pick_field`) si se usan otros datasets o si cambia el esquema.  

Desde el punto de vista del código, el punto de entrada de uso es la función `download_events_range`, que muestra en consola el período, el dataset y la ruta de destino y finalmente imprime estadísticas de descargas correctas, omitidas (ficheros ya existentes) y fallidas.  

## Funcionamiento del script

### Flujo interno

1. **Configuración**  
   - Se leen parámetros de `eventos_config`: URL base, `dataset_id` por defecto, límite de Socrata (`SOCRATA_LIMIT`), timeout de peticiones, y directorio de salida.  
   - Se inicializa un objeto `Console` de `rich` para tener una salida formateada en terminal.  

2. **Descarga paginada desde Socrata** (`_paged_socrata_json`)  
   - Recibe la URL del recurso `/resource/{dataset_id}.json` y un diccionario `params` con `$select`, `$where`, etc.  
   - Hace peticiones GET sucesivas con `$limit` y `$offset` hasta que no queda ningún lote (lista vacía).  
   - Acumula todas las filas devueltas en una lista `rows`, mostrando una barra de progreso (spinner, barra y contador de filas) mediante `rich.progress.Progress`.  

3. **Agregación por rango de fechas** (`download_events_aggregated`)  
   - Crea los ficheros temporales CSV/Parquet en el directorio de salida.  
   - Llama a `_fetch_view_metadata` y `_pick_field` para detectar dinámicamente el campo de inicio (`start_field`) y los campos de borough y tipo de evento.  
   - Construye las cadenas de `$select`, `$where`, `$group` y `$order` y llama a `_paged_socrata_json` para traer todos los registros agregados.  
   - Escribe el CSV temporal y lo convierte a Parquet con `csv_to_parquet`, borrando después el CSV.  

4. **Descarga mensual** (`download_events_month`)  
   - Calcula el primer y el último día del mes con `calendar.monthrange`.  
   - Comprueba si el Parquet final ya existe; si existe, marca el mes como `"skipped"` y no vuelve a descargar.  
   - Si no existe, llama a `download_events_aggregated` para el mes completo y renombra el Parquet temporal a `events_{year}_{month}.parquet`.  

5. **Descarga por rango de años/meses** (`download_events_range`)  
   - Genera el producto cartesiano de años y meses usando `itertools.product`.  
   - Para cada combinación año–mes invoca `download_events_month` y mantiene un diccionario `stats` con contadores de `"ok"`, `"skipped"` y `"failed"`.  
   - Controla errores con `try/except` por mes, de forma que un fallo en un mes no interrumpe el resto de la descarga.  

### Interfaz de línea de comandos (CLI)

El script expone un comando CLI con `click`:  

```bash
uv run -m src.extraccion.download_events_data \
  --start-year 2024 \
  --end-year 2024
```
Opciones principales:

- `--dataset`: ID del dataset de NYC Open Data (por defecto DEFAULT_DATASET).

- `--start-year` / `--end-year` (obligatorios): años inicial y final del período.

- `--start-month` (1–12, por defecto 1) y `--end-month` (1–12, por defecto 12): meses inicial y final dentro de cada año.

El comando llama internamente a download_events_range con estos parámetros y muestra un resumen de la descarga al terminar.