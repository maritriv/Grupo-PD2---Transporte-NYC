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
│ ├── extraccion/          ← Extracción de datos
│ ├── procesamiento/       ← Limpieza, procesamiento y unificación de datos
│ └── visualizaciones/     ← Análisis exploratorio y visualización
├── config/
│ └── settings.py          ← Centraliza toda la configuración estructural del proyecto
├── pyproject.toml         ← Configuración del proyecto y gestión de paquetes en uv
├── uv.lock                ← lockfile (uv)
├── .gitignore             ← Exclusión de archivos innecesarios
└── README.md              ← Este documento, Documentación principal del proyecto
```

---

## Índice
Detalles completos de cada subdirectorio y archivo:

- Directorio __`src/`__: Contiene el código fuente del proyecto.

    - Directorio __`extraccion/`__: Código responsable de la extracción de datos desde diversas fuentes.
      
            -   Archivo `download_event_data.py`: Descarga eventos desde NYC Open Data (Socrata), los agrega por date+hour+borough+event_type, y guarda un Parquet por mes.
            -   Archivo `download_meteo_data.py`: Descarga datos meteorológicos horarios de NYC desde Open-Meteo por meses/años, los guarda en Parquet y permite repetir descargas sin duplicar archivos.
            -   Archivo `download_tlc_data.py`: Descarga los ficheros Parquet mensuales de la NYC TLC (yellow/green/fhv/fhvhv) para un rango de fechas, gestionando skips, errores HTTP y barras de progreso.
            -   Archivo `main.py`:  Ejecuta los .py del modulo extraccion en orden (uso opciónal).

    - Directorio __`procesamiento/`__: Código encargado de procesar y estructurar los datos para su uso óptimo.
        -   Directorio __`capa 1/`__: Contiene código responsable de limpiar los datos extraídos.
          
                -   Directorio `capa1_resultados`: procesa y unifica datos de películas, profesionales del cine y sus puntuaciones, agregando información sobre premios y popularidad por año. Genera un archivo CSV unificado que contiene los detalles de las películas junto con los premios, nominaciones y popularidad de los profesionales asociados llamado peliculas_definitivo.csv
                -   Archivo `capa1.py`: Hace una exploración rápida de meses aleatorios, saca stats básicas (volumen, precio medio, hora pico, nº variables) y guarda un resumen en CSV/MD.
                
               
        -   Directorio __`capa 2/`__: Contiene scripts que integran y fusionan los diferentes conjuntos de datos procesados.
          
                -   Archivo `capa2_tlc.py`: Une los parquets RAW de taxis/VTC y los deja en un schema estándar (timestamps, variables temporales, precio unificado) y añade lookup de zonas; guarda en data/standarized/.
                -   Archivo `capa2_eventos.py`: Limpia y tipa los eventos (date/hour/borough/type), crea variables temporales y lo deja listo en formato estandarizado particionado en data/standarized/events/.
                -   Archivo `capa2_meteo.py`: Limpia y tipa meteorología (date/hour + numéricas), añade variables temporales y la guarda como parquets por año-mes en data/external/meteo/standarized/.
                -   Archivo `inspect_capa2.py`: Limpia y tipa meteorología (date/hour + numéricas), añade variables temporales y la guarda como parquets por año-mes en data/external/meteo/standarized/.

        -   Directorio __`capa 3/`__: Contiene scripts que integran y fusionan los diferentes conjuntos de datos procesados.
          
                -   Archivo `capa3_tlc.py`: Genera agregados de negocio de viajes (tendencia diaria, hotspots por zona/hora, y variabilidad de precio tipo “IQR”) y los guarda en data/aggregated/.
                -   Archivo `capa3_eventos.py`: Genera agregados de negocio de viajes (tendencia diaria, hotspots por zona/hora, y variabilidad de precio tipo “IQR”) y los guarda en data/aggregated/.
                -   Archivo `capa3meteo.py`: Construye agregados meteo por hora+día, resumen diario y patrón horario medio, y lo deja en data/external/meteo/aggregated/.
                -   Archivo `inspect_capa3.py`: Construye agregados meteo por hora+día, resumen diario y patrón horario medio, y lo deja en data/external/meteo/aggregated/.

    - Directorio __`visualizaciones/`__: Código encargado de procesar y estructurar los datos para su uso óptimo.
        -   Directorio __`viz_meteo/`__: Contiene código responsable de limpiar los datos extraídos.
     
                -   Archivo `clima_tipico.py.py`: Grafica el “día promedio” de NYC usando el patrón horario (temp media + desviación).
                -   Archivo `estacionalidad.py`: Visualiza estacionalidad: boxplots de temperatura por mes y comparación de precipitación entre laborables vs finde.
                -   Archivo `horaria_calor_viento.py`: Genera heatmaps por día de semana y hora para temperatura y viento (ej: enero vs diciembre).
                -   Archivo `overview.py`: Muestra la evolución histórica diaria (temperatura media/min/max + precipitación total) para detectar días extremos.
                -   Archivo `tendecias_clima.py`: Saca tendencias generales (temp + precip) y un gráfico con la distribución de códigos WMO (weather_code).

        -   Directorio __`viz_tlc/`__: Contiene código responsable de limpiar los datos extraídos.
     
                -   Archivo `visualizaciones_compartidas.py.py`: Genera scatterplots comparativos (distancia vs precio) entre servicios (yellow/green/fhvhv) para varios meses, usando muestreo eficiente con PyArrow.
                -   Archivo `visualizaciones_individuales.py`: Procesa cada Parquet por servicio y crea visualizaciones por archivo (heatmap de demanda por día/hora y dispersión precio vs distancia), aplicando muestreo para evitar problemas de memoria.
                -   Archivo `viz_01_overview.py`: Crea un overview temporal desde Capa 3: evolución del número de viajes y del precio medio diario por servicio.
                -   Archivo `viz_02_hotspots.py`: Construye heatmaps de “hotspots” (demanda media y precio medio) por zona y hora, quedándose con las zonas top por volumen para que sea legible.
                -   Archivo `viz_03_taxi_vs_vtc.py`: Compara Taxi vs VTC en zonas clave, graficando patrones horarios de demanda y precio medio por servicio.
                -   Archivo `viz_0t_tensions.py`: Analiza “tensiones” del mercado con la Capa 3: scatter volumen vs variabilidad (IQR) y ranking de oportunidades con biz_score.
                -   Archivo `viz_common.py`: Funciones comunes para las visualizaciones (Spark session, lectura de Capa 3, normalización de fechas y guardado de figuras).


---

## 🛠️ Instalación del entorno

Pasos necesarios para instalar el proyecto. Descripción paso a paso de cómo poner en funcionamiento el entorno de desarrollo.

----

**1.** Clonar el repositorio:  

```
git clone https://github.com/maritriv/Grupo-PD2---Transporte-NYC.git
cd Grupo-PD2---Transporte-NYC
```

----

**2.** Descarga las librerías necesarias creando automáticamente un entorno virtual con `uv sync` (desde la ubicación del `pyproject.toml`):
Instala uv (si no lo tienes instalado):
   ```
   pip install uv
```

```
uv sync
```
----

**3. Descargar los datos**

Los datos del proyecto se almacenan en **MinIO** (object storage).

Para descargarlos y mantener la estructura de directorios original, ejecuta:

```bash
uv run -m src.extraccion.download_from_minio
```

Por defecto:

- Descarga todo el contenido bajo data/

- Mantiene la misma estructura de carpetas

- Omite archivos que ya existen localmente

Opciones útiles:

```bash
# Descargar solo una subcarpeta
uv run -m src.extraccion.download_from_minio --prefix data/raw/

# Descargar en un directorio específico
uv run -m src.extraccion.download_from_minio --dest-dir /ruta/destino

# Forzar descarga (sobrescribir existentes)
uv run -m src.extraccion.download_from_minio --no-skip
```

> Es necesario que el archivo `credentials.json` esté configurado en la raíz del proyecto antes de ejecutar la descarga.

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
