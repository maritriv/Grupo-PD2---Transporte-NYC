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
  - `Capa 0` (exploración y muestreo inicial de los datos raw de TLC para estimar volumen, precio medio, hora pico y estructura por servicio y mes, generando un resumen exportable)
  - `Capa 1` (control de calidad estructural sobre los datos en bruto (raw), usando como referencia los Data Dictionaries oficiales de TLC)
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
            -   Archivo `download_from_minio.py`: Descarga todos los datos subidos a MinIO
            -   Archivo `download_meteo_data.py`: Descarga datos meteorológicos horarios de NYC desde Open-Meteo por meses/años, los guarda en Parquet y permite repetir descargas sin duplicar archivos.
            -   Archivo `download_tlc_data.py`: Descarga los ficheros Parquet mensuales de la NYC TLC (yellow/green/fhv/fhvhv) para un rango de fechas, gestionando skips, errores HTTP y barras de progreso.
            -   Archivo `main.py`:  Ejecuta los .py del modulo extraccion en orden salvo `download_from_minio.py` (uso opciónal).

    - Directorio __`procesamiento/`__: Código encargado de procesar y estructurar los datos para su uso óptimo.
        -   Archivo __`capa 0/main.py`__: Realiza una exploración diagnóstica sobre muestras representativas de los datos RAW.
Extrae estadísticas básicas (volumen estimado, precio medio, hora pico, número de variables) y genera un resumen en CSV y Markdown.

        -   Directorio __`capa 1/`__: Contiene código responsable de limpiar los datos extraídos.

                -   Archivo `fhv.py`: Valida estructura y coherencia del servicio FHV. Comprueba tipos, timestamps, rangos válidos y separa registros en clean y badrows.
                -   Archivo `fhvhv.py`: Validación específica para el servicio FHVHV (High Volume). Aplica controles estructurales y temporales adaptados a este esquema.
                -   Archivo `capa1_green.py`: Valida los datos del servicio Green Taxi, asegurando consistencia en campos monetarios, distancias y timestamps.
                -   Archivo `capa1_yellow.py`: Valida los datos del servicio Green Taxi, asegurando consistencia en campos monetarios, distancias y timestamps.
                -   Archivo `main.py`: Ejecuta secuencialmente los validadores de cada servicio y genera los datasets limpios.
               
        -   Directorio __`capa 2/`__: Contiene scripts que integran y fusionan los diferentes conjuntos de datos procesados.
          
                -   Archivo `capa2_tlc.py`: Une los parquets RAW de taxis/VTC y los deja en un schema estándar (timestamps, variables temporales, precio unificado) y añade lookup de zonas; guarda en data/standarized/.
                -   Archivo `capa2_eventos.py`: Limpia y tipa los eventos (date/hour/borough/type), crea variables temporales y lo deja listo en formato estandarizado particionado en data/standarized/events/.
                -   Archivo `capa2_meteo.py`: Limpia y tipa meteorología (date/hour + numéricas), añade variables temporales y la guarda como parquets por año-mes en data/external/meteo/standarized/.
                -   Archivo `main.py`: Orquesta la ejecución de todos los scripts de capa 2.

        -   Directorio __`capa 3/`__: Contiene scripts que integran y fusionan los diferentes conjuntos de datos procesados.
          
                -   Archivo `capa3_tlc.py`: Genera agregados de negocio de viajes (tendencia diaria, hotspots por zona/hora, y variabilidad de precio tipo “IQR”) y los guarda en data/aggregated/.
                -   Archivo `capa3_eventos.py`: Genera agregados de negocio de viajes (tendencia diaria, hotspots por zona/hora, y variabilidad de precio tipo “IQR”) y los guarda en data/aggregated/.
                -   Archivo `capa3_meteo.py`: Construye agregados meteo por hora+día, resumen diario y patrón horario medio, y lo deja en data/external/meteo/aggregated/.
                -   Archivo `main.py`: Ejecuta los procesos de agregación de capa 3 de forma coordinada.
            
        -   Archivo __`main.py`__: Orquesta el pipeline completo de procesamiento: capa0 → capa1 → capa2 → capa3. Permite ejecutar todo el flujo estructurado desde datos RAW hasta agregados finales.

- Directorio __`visualizaciones/`__: Código encargado de procesar y estructurar los datos para su uso óptimo.

  -   Directorio __`viz_conjuntas/`__: Cruza datos TLC + Meteorología + Eventos.

                -   Archivo `viz_2024.py`: Genera visualizaciones conjuntas para un año concreto. Utiliza agregados de capa 3 y Spark para el cruce eficiente de datasets.

  -   Directorio __`viz_meteo/`__: Contiene código responsable de limpiar los datos extraídos.
     
                -   Archivo `clima_tipico.py`: Grafica el “día promedio” de NYC usando el patrón horario (temp media + desviación).
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
                -   Archivo `viz_04_tensions.py`: Analiza “tensiones” del mercado con la Capa 3: scatter volumen vs variabilidad (IQR) y ranking de oportunidades con biz_score.
                -   Archivo `viz_common.py`: Funciones comunes para las visualizaciones (Spark session, lectura de Capa 3, normalización de fechas y guardado de figuras).

- Archivo __`main.py/`__: Script orquestador principal del proyecto. Ejecuta de forma secuencial el pipeline completo, lanzando primero los procesos de extracción de datos (src/extraccion/) y posteriormente el flujo de procesamiento por capas (capa0 → capa1 → capa2 → capa3). Permite ejecutar todo el proyecto desde datos externos hasta agregados finales listos para análisis y visualización mediante un único comando.

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

**⚠️ Configuración necesaria para Spark**
Este proyecto utiliza **PySpark** para las visualizaciones y agregaciones. Es necesario tener Java 17 (JDK) correctamente configurado.

**2.1. Instalar Java 17**

Si no tienes Java 17 instalado, accede a este enlace y descarga Temurin 17 (JDK).

[Enlace a Temurin 17](https://adoptium.net/es)

Verifica la instalación:
```
java -version
```

Debe mostrar algo similar a:
```
openjdk version "17.x.x"
```
**2.2. Configurar las variables de entorno**

🪟 En Windows (PowerShell):
```
$env:JAVA_HOME="C:\Program Files\Eclipse Adoptium\jdk-17.x.x"
$env:Path="$env:JAVA_HOME\bin;$env:Path"
$env:PYSPARK_PYTHON="python"
$env:PYSPARK_DRIVER_PYTHON="python"
```

🐧 En macOS / Linux:
Añadir al .zshrc o .bashrc:
```
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
export PATH=$JAVA_HOME/bin:$PATH
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYSPARK_PYTHON=python3
```
Aplicar cambios:
```
source ~/.zshrc
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

Para ejecutar las visualizaciones es necesario descargar los datos de zonas de NYC.
Ejecuta el siguiente comando en la raíz del proyecto según tu sistema:

**Linux / WSL / macOS (con wget):**

```bash
wget -P data/external/ https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv
```

Si wget no está instalado:
- Ubuntu/WSL: sudo apt install wget
- macOS (Homebrew): brew install wget

Alternativa en macOS (con curl):
```bash
curl -o data/external/taxi_zone_lookup.csv https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv
```

**Windows – PowerShell**

```bash
New-Item -ItemType Directory -Path "data\external" -Force
Invoke-WebRequest -Uri "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv" -OutFile "data\external\taxi_zone_lookup.csv"
```

**Windows – Command Prompt (cmd)**

```bash
mkdir data\external
curl -o data\external\taxi_zone_lookup.csv https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv
```
----

## Ejecución de la página web

**1. Abre una terminal en la raíz del proyecto y ejecuta:
```bash
uv run -m uvicorn backend.app.main:app --reload
```

Si arranca bien, aparecerá una linea parecida a:
```bash
Uvicorn running on http://127.0.0.1:8000
```
Busca en el navegador:
```bash 
http://127.0.0.1:8000/api/health
```

Debería aparecer algo como:
`{"status":"ok"}`


**2. Abre otra terminal y ejecuta:**
```bash
cd frontend
npm install
npm run dev
```

Si va bien, Vite te dará una URL, normalmente:
```bash
http://localhost:5173
```
Abre esa dirección en el navegador.

## Equipo de desarrollo

Este proyecto fue desarrollado por los siguientes estudiantes del Grado en Ingeniería de Datos e Inteligencia Artificial (UCM): 
- Vega García Camacho
- Rosa Gómez-Gil Jónsdóttir 
- Daniel Higueras Llorente
- Ignacio Ramírez Suárez
- Marina Triviño de las Heras

##  Recursos adicionales
# Memorias y presentaciones
**Entrega 1**
- [Presentación Entrega 1](https://docs.google.com/presentation/d/1tKNixIGUhMHiNGJyOn6zXNW0MvKN4t1Iz4JFZ0_KaAE/edit?slide=id.g3c872e10c63_0_294#slide=id.g3c872e10c63_0_294)
- [Memoria Entrega 1](https://docs.google.com/document/d/1znwca7mk1cS6DRcjjuXsSMnBJvdzIXBFVLbBbsAFyls/edit?usp=sharing)
- [Entrega 1: Distribución del Trabajo](https://docs.google.com/document/d/1K5g5cqhqr7BZ0P4KehW0uqv_OZGyN_cjHm6tFBikYTY/edit?usp=sharing)

**Entrega 2**
- [Memoria Entrega 2](https://docs.google.com/document/d/12uWMO4wAyDforRW51FLFtRzltHT_LC2Gat92gY-dCvg/edit?tab=t.0)

# Visualizaciones
- [Google drive](https://drive.google.com/drive/folders/1gWM-5GU0OTZgczfwt1Mxz7wQFQUuLo5Z?usp=drive_link)
