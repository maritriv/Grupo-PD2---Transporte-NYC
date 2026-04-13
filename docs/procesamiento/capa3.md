# Capa 3: Agregaciones y Datasets para Analítica y Modelos ML

## 1. Descripcion general

La Capa 3 es la capa de agregación final que transforma los datos estandarizados de la Capa 2 en tablas resumidas y datasets model-ready para análisis y machine learning.

Su propósito es:
- Generar agregados base reutilizables (demanda diaria, hotspots, variabilidad de precio)
- Construir datasets específicos para cada ejercicio/modelo de negocio
- Enriquecer datos de TLC con datos externos (meteorología, eventos, alquiler, restaurantes)
- Mantener separación clara entre análisis exploratorio y datasets para modelado

La arquitectura de Capa 3 se organiza en dos pasos secuenciales:
1. Paso 1: Generación de agregados base (reutilizables, independientes entre sí)
2. Paso 2: Ensamblaje de datasets para ejercicios (dependen de Paso 1, contienen lógica específica)


## 2. Arquitectura y estructura

### 2.1 Organización de directorios

```
src/procesamiento/capa3/
├── main.py                          # Orquestador principal (2 pasos)
├── aggregates/                      # Paso 1: Agregados base
│   ├── __init__.py
│   ├── tlc.py                       # Agregados TLC (demanda, hotspots, variabilidad)
│   ├── eventos.py                   # Agregados de eventos urbanos
│   ├── meteo.py                     # Agregados de meteorología
│   ├── rent.py                      # Datos estáticos de alquiler por zona
│   └── restaurants.py               # Datos estáticos de restaurantes por zona
├── ejercicios/                      # Paso 2: Datasets para ejercicios
│   ├── __init__.py
│   ├── ex1a_demand.py               # Ej.1a: Predicción de zona máxima demanda
│   ├── ex1b_tips.py                 # Ej.1b: Predicción de propina
│   ├── ex1c_patterns.py             # Ej.1c: Patrones de demanda (bajo/medio/alto)
│   ├── ex1d_socioeconomic.py        # Ej.1d: Poder adquisitivo por zona
│   └── ex2_stress.py                # Ej.2: Dashboard de estrés urbano
├── builders/                        # Lógica compleja para ensamblaje de features
│   ├── __init__.py
│   ├── demand_zone.py               # Constructor de dataset demanda (ex1a)
│   ├── propinas.py                  # Constructor de features de propina (ex1b)
│   └── stress_zone.py               # Constructor de dataset estrés (ex2)
├── pipelines/                       # Orquestadores de builders
│   ├── __init__.py
│   ├── run_demand_zone.py           # Ejecutor de builder demand_zone
│   └── run_stress_zone.py           # Ejecutor de builder stress_zone
└── common/                          # Utilidades compartidas
    ├── __init__.py
    ├── constants.py                 # Configuración global (fechas, console)
    ├── io.py                        # Lectura/escritura particionada, utilidades FS
    └── externals.py                 # Carga de datos externos (meteo, eventos, etc.)
```

### 2.2 Flujo de ejecución

Cuando se ejecuta main.py:

Paso 1: Agregados base (independientes)
- Cada script carga datos de su fuente (TLC, eventos, etc.)
- Realiza agregaciones específicas
- Guarda salidas particionadas
- Sin dependencias entre ellos (ejecución parallelizable)

Paso 2: Datasets ejercicios (dependen de Paso 1)
- Leen agregados base o construyen nuevas features
- Combinan múltiples fuentes (ex: ex1a combina TLC + meteo + eventos + rent)
- Generan datasets model-ready
- Cada uno orientado a una tarea concreta


## 3. Detalle de componentes

### 3.1 Paso 1: Agregados Base

#### aggregates/tlc.py

Lee datos estandarizados de TLC y genera agregados de demanda y variabilidad:

Entrada: data/standarized/ (Capa 2 TLC particionada por year/month/service_type)

Salidas en data/aggregated/:

1. df_daily_service/
   - Nivel: date + service_type
   - Campos: num_trips, avg_price, std_price, unique_zones
   - Uso: Evolución temporal de demanda y precios

2. df_zone_hour_day_global/
   - Nivel: pu_location_id + hour + date (sin separar servicios)
   - Campos: num_trips, avg_price, std_price
   - Filtro: num_trips >= 30 (confiabilidad mínima)
   - Uso: Identificar hotspots por zona-hora-día

3. df_zone_hour_day_service/
   - Nivel: pu_location_id + hour + date + service_type
   - Campos: num_trips, avg_price, std_price
   - Filtro: num_trips >= 30
   - Uso: Comparar servicios (yellow vs green vs fhvhv) en misma zona-hora-día

4. df_variability/
   - Nivel: pu_location_id + hour + service_type (agregado temporal)
   - Campos: num_trips, avg_price, price_variability (IQR), biz_score
   - Filtro: num_trips >= 100 (mínimo para estimar percentiles)
   - Uso: Identificar zonas-horas con precio impredecible

Parámetros configurables:
- min_date, max_date (rango temporal)
- cap_max_price (corte de outliers, default: 500)
- min_trips_df2 (mínimo para hotspots, default: 30)
- min_trips_df3 (mínimo para variabilidad, default: 100)

#### aggregates/eventos.py

Lee eventos urbanos estandarizados y genera agregados por zona-hora-día.

Salidas en data/external/events/aggregated/:
- df_borough_hour_day: Eventos por borough, hora y día
- df_daily_borough: Eventos por borough y día
- df_type_daily_borough: Ranking de tipos de evento
- df_hourly_pattern: Patrón horario medio por borough

#### aggregates/meteo.py

Lee datos meteorológicos estandarizados.

Salidas en data/external/meteo/aggregated/:
- Datos meteorológicos agregados disponibles para enriquecer modelos

#### aggregates/rent.py y aggregates/restaurants.py

Leen datos estáticos de rent y restaurantes agregados por zona TLC.

Salidas en data/external/{rent,restaurants}/aggregated/:
- Datos estáticos por pu_location_id (zona)


### 3.2 Paso 2: Datasets Ejercicios

#### ejercicios/ex1a_demand.py

Ej.1a: Predecir la zona de máxima demanda para un día/hora

Wrapper que ejecuta pipelines/run_demand_zone.py

Entrada: TLC (capa 2) + meteorología + eventos + rent + restaurantes

Salida: data/aggregated/ex1a/df_demand_zone_hour_day/

Campos:
- Temporales: date, year, month, hour, hour_block (franjas de 3h), day_of_week, is_weekend, week_of_year
- Geográficos: pu_location_id
- Features de demanda histórica: lag_1h, lag_24h, lag_168h (rezagos de demanda)
- Features de tendencia: rolling_mean_3h, rolling_mean_24h
- Variables externas: temp_c, precip_mm, city_n_events, city_has_event, n_restaurants_zone, rent_price_zone
- Target: target_n_trips (número de viajes en esa zona-hora)

Lógica en builders/demand_zone.py


#### ejercicios/ex1b_tips.py

Ej.1b: Predecir propina de un viaje individual

Entrada: TLC capa 2 a nivel viaje

Salida: data/aggregated/ex1b/df_trip_level_tips/ (particionado por year/month)

Campos:
- Identificadores: service_type, pickup_datetime, dropoff_datetime, pu_location_id, do_location_id
- Temporales: date, year, month, hour, day_of_week, is_weekend, week_of_year
- Viaje: trip_distance, trip_duration_min, passenger_count, payment_type, RatecodeID
- Económicos: total_amount_std, fare_amount
- Targets: target_tip_amount, target_tip_pct, has_tip (binario)

Filtros de calidad:
- is_valid_for_tip == 1
- fare_amount > 0
- No nulos en campos críticos

Arquitectura: Pandas (lectura eficiente de particiones mensuales)


#### ejercicios/ex1c_patterns.py

Ej.1c: Identificar patrones de demanda por zona-hora

Clasifica demanda en niveles (Baja/Media/Alta) según zona y franja horaria.
Agrega análisis de estabilidad (predecibilidad de demanda).

Entrada: data/aggregated/df_zone_hour_day_global (agregado TLC)

Salida: data/aggregated/ex1c/df_demand_patterns/ (particionado por pu_location_id)

Lógica:
1. Agrega zona-hora promediando sobre fechas (obtiene demanda media por zona-hora)
2. Clasifica demanda en terciles POR ZONA (respeta naturaleza local):
   - Baja: < percentil 33 de trips en esa zona
   - Media: percentil 33-66
   - Alta: > percentil 66
3. Calcula coeficiente de variación (CV) como medida de estabilidad
4. Clasifica estabilidad en terciles globales:
   - Predecible: CV bajo (demanda consistente)
   - Variable: CV medio
   - Volátil: CV alto (demanda impredecible)
5. Calcula puntuación de prioridad operacional (demanda × estabilidad)

Campos de salida:
- pu_location_id, hour
- num_trips_avg, num_trips_std, num_trips_min, num_trips_max, num_trips_count
- demand_level, stability, cv, operational_priority, operational_priority_label

Uso: Identificar zonas-horas confiables vs riesgosas para operación


#### ejercicios/ex1d_socioeconomic.py

Ej.1d: Caracterizar poder adquisitivo por zona

Mapeo de nivel socioeconómico usando datos de taxis (propinas, pasajeros, volumen).

Entrada: Agregados TLC (df_daily_service) + propinas agregadas

Salida: data/aggregated/ex1d/df_zone_socioeconomic/

Uso: Generar mapa coroplético de poder adquisitivo con colores por zona

Nota: Requiere implementación específica (stub actualmente)


#### ejercicios/ex2_stress.py

Ej.2: Dashboard y Modelo Predictivo de Estrés Urbano
Visualización de estabilidad del sistema de transporte combinando demanda + variabilidad, y preparación de datos para predicción a corto plazo.

Wrapper que ejecuta pipelines/run_stress_zone.py

Entradas: TLC + meteorología + eventos + alquileres + restaurantes

Salidas principales (ruta base: data/aggregated/ex_stress/):
  - df_stress_zone_hour_day (Dataset Model-Ready)
  - df_stress_zone_slot (Dataset de Panel Agregado)

Indicador sintético que combina:

- Variabilidad de precio (volatilidad estandarizada mediante Z-score)
- Volumen de demanda (magnitud logarítmica estandarizada)
- Contexto externo (eventos, clima, renta, restaurantes)

Horizonte de predicción y Targets:

Se predice a corto plazo (horizonte t+1, es decir, a 1 hora vista).
- target_stress_t1 (Regresión): Valor continuo del indicador sintético en la hora siguiente.
- target_is_stress_t1 (Clasificación): Variable binaria (1/0) que indica si en la siguiente hora el indicador superará el umbral crítico de estrés (por defecto, el percentil 90 histórico).

Útil para: Predicción de picos a 1 hora vista, alertas operacionales preventivas, identificación de contextos críticos, análisis histórico.


## 4. Datos de entrada y salida

### 4.1 Datos de entrada esperados

Capa 2 (TLC):
- Ubicación: data/standarized/
- Estructura: service_type=XX/year=YYYY/month=MM/*.parquet
- Columnas mínimas: date, hour, service_type, pu_location_id, total_amount_std, tip_amount, tip_pct, etc.

Datos externos:
- Meteorología: data/external/meteo/standarized/
- Eventos: data/external/events/standarized/
- Alquiler: data/external/rent/aggregated/
- Restaurantes: data/external/restaurants/aggregated/

### 4.2 Datos de salida generados

Agregados base:
- data/aggregated/df_daily_service/ (particionado por service_type)
- data/aggregated/df_zone_hour_day_global/ (particionado por date)
- data/aggregated/df_zone_hour_day_service/ (particionado por date, service_type)
- data/aggregated/df_variability/ (particionado por service_type)

Datasets ejercicios:
- data/aggregated/ex1a/df_demand_zone_hour_day/ (particionado por hour, pu_location_id)
- data/aggregated/ex1b/df_trip_level_tips/ (particionado por year, month)
- data/aggregated/ex1c/df_demand_patterns/ (particionado por pu_location_id)
- data/aggregated/ex1d/df_zone_socioeconomic/ (estructura por decidir)
- data/aggregated/ex_stress/df_stress_zone_hour_day/ (particionado por year, month) -> Dataset Model-Ready
- data/aggregated/ex_stress/df_stress_zone_slot/ (particionado por day_of_week) -> Dataset Panel para Dashboard


## 5. Ejecucion

### 5.1 Ejecutar Capa 3 completa

```bash
uv run -m src.procesamiento.capa3.main
```

Ejecuta secuencialmente:
- Paso 1: Todos los agregados base (tlc, eventos, meteo, rent, restaurants)
- Paso 2: Todos los datasets ejercicios (ex1a, ex1b, ex1c, ex1d, ex2)


### 5.2 Ejecutar componente individual

```bash
# Solo agregados TLC
uv run -m src.procesamiento.capa3.aggregates.tlc --mode overwrite --cap-max-price 500

# Solo Ejercicio 1b (propinas)
uv run -m src.procesamiento.capa3.ejercicios.ex1b_tips --mode overwrite

# Solo Ejercicio 1c (patrones)
uv run -m src.procesamiento.capa3.ejercicios.ex1c_patterns --mode overwrite
```

Parámetros comunes:
- --in-dir: Directorio de entrada
- --out-dir: Directorio de salida
- --from, --to: Rango de fechas (formato YYYY-MM-DD)
- --mode: 'overwrite' (borra previas) o 'append' (conserva)


## 6. Consideraciones de diseño

### 6.1 Terciles por zona (vs percentiles fijos)

Clasificación de demanda en Ej.1c usa terciles POR ZONA (no globales).

Razón: Respeta naturaleza local de cada zona. Midtown siempre tendrá demanda mayor que Queens, pero ambas pueden estar en su "tercil alto". Permite identificar franjas horarias de pico locales en cada zona sin sesgos geográficos.

### 6.2 Separación entre agregados y ejercicios

Paso 1 (agregados) genera tablas reutilizables que pueden consumirse multipropósito (análisis exploratorio, dashboards, etc).
Paso 2 (ejercicios) construye datasets específicos orientados a ML.

Esta separación permite que cambios en un ejercicio no afecten a otros.

### 6.3 Estabilidad + Demanda en Ej.1c

Se incluye análisis de estabilidad (variabilidad de demanda) además de nivel de demanda.

Razón: Operacionalmente es valioso saber no solo dónde hay demanda, sino QUE TAN PREDECIBLE es. Alta demanda predecible diferencia de alta demanda volátil.

### 6.4 Builders segregados de ejercicios

Lógica compleja (builders/) está separada de puntos de entrada (ejercicios/).

Razón: Builders contienen lógica reutilizable que puede compartirse entre ejercicios (ej: demand_zone.py es usado por ex1a pero podría usarse en otras análisis).


## 7. Notas técnicas

### 7.1 Gestión de memoria

Capa 3 está optimizada para procesar millones de registros:
- Lectura particionada (mes a mes para TLC)
- Garbage collection explícito en aggregates/tlc.py
- Escritura particionada (por date, service_type, etc.) para consultas eficientes

### 7.2 Particionamiento

Todas las salidas se guardan particionadas (estilo Spark), aunque se implementa en pandas. Esto permite:
- Lectura selectiva (sin leer todo el dataset)
- Paralelización posible en futuro
- Compatibilidad con herramientas de análisis (Spark, DuckDB, SQL)

### 7.3 Validación de datos

Cada módulo aplica filtros defensivos:
- Checks de nulos en columnas críticas
- Validaciones de rango (price > 0, hour 0-23, etc.)
- Marcadores originales de validez (is_valid_for_tip, etc.)
