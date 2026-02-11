# MacBrides

## Descripción

Proyecto de **Proyecto de Datos II (UCM, 2025/26)**.  
El objetivo es **analizar el sistema de transporte de pago en Nueva York** (taxis/VTC) e incorporar fuentes externas (meteorología y eventos) para:

- Entender patrones de demanda (zonas/horas)
- Detectar tensiones del sistema (picos, desigualdad, variabilidad…)
- Proponer una **aplicación basada en datos** con impacto medible
- Respaldar la propuesta con visualizaciones y estudio de mercado 

---
### Funcionalidades

- **Extracción** de fuentes reales:
  - TLC (viajes taxi / VTC)
  - **Meteorología** (Open-Meteo)
  - **Eventos** (NYC Open Data)
- **Pipeline por capas**:
  - `RAW` (datos descargados)
  - `Capa 2` (limpieza + estandarización)
  - `Capa 3` (agregación lista para análisis y cruces)
- **Visualizaciones** para justificar la propuesta (viajes vs meteo vs eventos)

---

## Estructura general del repositorio
```
Grupo-PD2---Transporte-NYC/
├── src/
│ ├── extraccion/ ← Descarga de datasets (RAW)
│ ├── procesamiento/
│ │ ├── capa2/ ← Limpieza + estandarización (standarized)
│ │ └── capa3/ ← Agregaciones listas para análisis (aggregated)
│ └── ... ← notebooks / visualización (según repo)
├── config/
│ └── settings.py ← rutas + configuración (si aplica)
├── data/ ← (NO se sube a GitHub, va por Drive)
│ ├── external/ ← RAW
│ ├── standarized/ ← Capa 2
│ ├── aggregated/ ← Capa 3
│ └── ...
├── pyproject.toml ← dependencias (uv)
├── uv.lock ← lockfile (uv)
├── .gitignore
└── README.md
```

---

## Índice
Detalles completos de cada subdirectorio y archivo:


---

## Índice del pipeline (scripts principales)

### 1) Extracción (RAW) — `data/external/`

- **Meteorología (Open-Meteo)**
  - `src/extraccion/download_meteo_data.py`
  - Descarga meteo **horaria** para NYC y la guarda en Parquet (por meses).
  - Salida: `data/external/meteo/raw/*.parquet`

- **Eventos (NYC Open Data)**
  - `src/extraccion/download_events_data.py`
  - Descarga eventos agregados por `date/hour/borough/type` (según vuestra implementación).
  - Salida: `data/external/events/...`

- **(Transporte TLC)**
  - `src/extraccion/...` (según vuestros scripts)
  - Descarga/gestión de Yellow Taxi + VTC.
  - Salida: `data/external/...`

---

### 2) Capa 2 (Standarized) — `data/standarized/`

- **Capa 2 Meteo**
  - `src/procesamiento/capa2/capa2_meteo.py`
  - Limpia y tipa columnas (`date`, `hour`, numéricos), añade variables temporales
    (`year`, `month`, `day_of_week`, `is_weekend`, `week_of_year`)
  - Particiona estilo Spark: `year=YYYY/month=MM/part-00000.parquet`
  - Salida: `data/standarized/meteo/`

- **Capa 2 Eventos**
  - `src/procesamiento/capa2/capa2_eventos.py`
  - Estandariza y particiona los eventos.
  - Salida: `data/standarized/events/`

- **Capa 2 Transporte (Taxi/VTC)**
  - `src/procesamiento/capa2/...` (según repo)
  - Estandariza los viajes y variables temporales / zonas.
  - Salida: `data/standarized/...`

---

### 3) Capa 3 (Aggregated) — `data/aggregated/`

- **Capa 3 Meteo**
  - `src/procesamiento/capa3/capa3_meteo.py`
  - Genera datasets agregados listos para cruzar con viajes:
    - `df_hour_day` (date+hour)
    - `df_daily` (date)
    - `df_hourly_pattern` (hour)
    - `df_weathercode_daily` (opcional)
  - Salida: `data/aggregated/meteo/...`

- **Capa 3 Eventos**
  - `src/procesamiento/capa3/capa3_eventos.py`
  - Agregados por borough/hora/día (según vuestro diseño).
  - Salida: `data/aggregated/events/...`

- **Capa 3 Transporte**
  - `src/procesamiento/capa3/...` (según repo)
  - Agregados para visualizaciones y comparativas.
  - Salida: `data/aggregated/...`

---

## 🛠️ Instalación del entorno

Pasos necesarios para instalar el proyecto. Descripción paso a paso de cómo poner en funcionamiento el entorno de desarrollo.

----

**1.** Clonar el repositorio:  

```
git clone <URL_DEL_REPO>
cd Grupo-PD2---Transporte-NYC
```

----

**2.**  Descargar los datos  
Descargar desde el Drive del proyecto. Mover la carpeta datos al repositorio raíz.
[Enlace a los datos (Drive)](https://drive.google.com/drive/u/2/folders/1gWM-5GU0OTZgczfwt1Mxz7wQFQUuLo5Z)   

----

**3.** Descarga las librerías necesarias creando automáticamente un entorno virtual con `uv sync` (desde la ubicación del `pyproject.toml`):
Instala uv (si no lo tienes instalado):
   ```
   pip install uv
```

```
uv sync
```
----

## Equipo de desarrollo

Este proyecto fue desarrollado por los siguientes estudiantes del Grado en Ingeniería de Datos e Inteligencia Artificial (UCM): 
- Vega García Camacho
- Marina Triviño de las Heras
- Rosa Gómez-Gil Jónsdóttir
- Ignacio Ramírez Suárez
- Daniel Higueras Llorente

##  Recursos adicionales
# Memorias y presentaciones
- [Presentación Entrega 1]()
- [Memoria Entrega 1](https://docs.google.com/document/d/1znwca7mk1cS6DRcjjuXsSMnBJvdzIXBFVLbBbsAFyls/edit?usp=sharing)

# Enlace a Google Drive:
Este enlace contiene los datos necarios para el proyecto. La carpeta llamada `data` simula aproximadamente la ejecución de todos los scripts del proyecto en orden.
- [Enlace a los datos (Drive)](https://drive.google.com/drive/u/2/folders/1gWM-5GU0OTZgczfwt1Mxz7wQFQUuLo5Z).  
