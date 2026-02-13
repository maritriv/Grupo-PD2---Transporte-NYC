# src/procesamiento/capa3/capa3_eventos.py
"""
VISUALIZACIONES TLC + METEO + EVENTOS (POR AÑO)

FINALIDAD DEL SCRIPT
--------------------
Este script genera un conjunto de visualizaciones exploratorias para analizar
cómo factores externos —condiciones meteorológicas y eventos urbanos—
afectan a la demanda y al precio del transporte urbano en NYC (TLC),
diferenciando por:
- hora del día,
- tipo de servicio (yellow, green, fhvhv),
- y contexto (lluvia / eventos).

El objetivo NO es hacer predicción ni construir un baseline estadístico complejo,
sino entender y explicar patrones de comportamiento de forma clara y eficiente.


CONTROL DEL AÑO ANALIZADO
-------------------------
El año a analizar se elige mediante un parámetro de entrada (--year).
Por defecto se analiza 2024, pero el script puede ejecutarse para cualquier
otro año disponible en los datos sin modificar el código.

Esto garantiza que:
- el análisis no depende de qué años existan en disco,
- no se mezclan dinámicas de distintos años,
- y las conclusiones son comparables entre años si se desea.


DATOS DE PARTIDA
----------------
El script trabaja exclusivamente sobre datos ya agregados (CAPA 3):

- TLC (taxis y VTC): agregados por zona, hora, día y servicio.
- Meteo: agregados horarios (lluvia, temperatura, viento).
- Eventos: agregados por borough, hora y día.

NO se trabaja a nivel de viaje individual (CAPA 2), lo que permite:
- reducir drásticamente el coste computacional,
- evitar problemas de memoria (WSL-friendly),
- y centrar el análisis en patrones estructurales.


NIVEL DE AGREGACIÓN
-------------------
Todas las visualizaciones se construyen a nivel:
    zona (o borough) × hora × servicio

Este nivel es un compromiso óptimo entre:
- granularidad suficiente para capturar patrones reales,
- y estabilidad/interpretabilidad de los resultados.


VISUALIZACIONES GENERADAS
-------------------------

1) DEMANDA VS LLUVIA (PERFIL HORARIO)
   - Demanda media por hora, separando horas con lluvia y sin lluvia.
   - Responde:
       * ¿La lluvia modifica el patrón horario de la demanda?
       * ¿Afecta igual a todos los servicios?

2) PRECIO VS LLUVIA EN ZONAS CLAVE (HOTSPOTS)
   - Precio medio por hora en zonas de alta demanda,
     comparando no_rain vs heavy rain.
   - Responde:
       * ¿Existe un “weather premium”?
       * ¿En qué zonas y servicios es más evidente?

3) DEMANDA VS INTENSIDAD DE EVENTOS
   - Demanda media por hora según el nivel de actividad de eventos
     en el borough (none / low / mid / high).
   - Responde:
       * ¿Los eventos urbanos generan demanda adicional?
       * ¿En qué horarios y para qué servicios?

4) INTERACCIÓN LLUVIA × EVENTOS
   - Demanda media por hora combinando lluvia (sí/no) y eventos (sí/no).
   - Responde:
       * ¿El efecto de los eventos cambia cuando además llueve?
       * ¿Los shocks externos se refuerzan o se compensan?


QUÉ BUSCA RESPONDER EL ANÁLISIS
-------------------------------
Este script permite responder de forma visual y explicativa a:
- cómo factores externos alteran la movilidad urbana,
- cuándo se producen los mayores impactos,
- y qué servicios son más sensibles o más resilientes.

Está pensado como herramienta de análisis exploratorio e interpretativo,
no como pipeline de producción ni como modelo predictivo.
"""
# notebooks/faseB/viz_09_meteo_eventos.py
from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
from pyspark.sql import functions as F

from src.visualizaciones.viz_tlc.viz_common import get_spark, read_capa3, save_fig

# Rutas según tu pipeline
METEO_C3_PATH = "data/external/meteo/aggregated/df_hour_day/data.parquet"
EVENTS_C3_PATH = "data/aggregated/events/df_borough_hour_day"

FOCUS_ZONES = [79, 132, 138]

RAIN_LIGHT_MAX = 2.0
RAIN_MODERATE_MAX = 8.0


def read_meteo_c3(spark, year: int):
    m = (
        spark.read.parquet(METEO_C3_PATH)
        .withColumn("date", F.to_date("date"))
        .withColumn("hour", F.col("hour").cast("int"))
        .where(F.year("date") == year)
        .select(
            "date",
            "hour",
            F.col("rain_mm_sum").cast("double").alias("rain_mm"),
            F.col("precip_mm_sum").cast("double").alias("precip_mm"),
            F.col("temp_c_mean").cast("double").alias("temp_c"),
            F.col("wind_kmh_mean").cast("double").alias("wind_kmh"),
        )
        .withColumn("rain_flag", (F.col("rain_mm") > 0).cast("int"))
        .withColumn(
            "rain_bin",
            F.when(F.col("rain_mm") <= 0, F.lit("no_rain"))
             .when(F.col("rain_mm") <= F.lit(RAIN_LIGHT_MAX), F.lit("light"))
             .when(F.col("rain_mm") <= F.lit(RAIN_MODERATE_MAX), F.lit("moderate"))
             .otherwise(F.lit("heavy"))
        )
    )
    return m


def read_events_c3(spark, year: int):
    e = (
        spark.read.parquet(EVENTS_C3_PATH)
        .withColumn("date", F.to_date("date"))
        .withColumn("hour", F.col("hour").cast("int"))
        .where(F.year("date") == year)
        .select(
            F.col("borough").alias("pu_borough"),
            "date",
            "hour",
            F.col("n_events").cast("int").alias("n_events"),
            F.col("n_event_types").cast("int").alias("n_event_types"),
        )
        .withColumn("event_flag", (F.col("n_events") > 0).cast("int"))
        .withColumn(
            "event_level",
            F.when(F.col("n_events") <= 0, F.lit("none"))
             .when(F.col("n_events") <= 3, F.lit("low"))
             .when(F.col("n_events") <= 10, F.lit("mid"))
             .otherwise(F.lit("high"))
        )
    )
    return e


def plot_demand_by_hour_rainflag(dfj, year: int):
    agg = (
        dfj.groupBy("service_type", "hour", "rain_flag")
        .agg(F.avg("num_trips").alias("avg_num_trips"))
        .orderBy("service_type", "hour", "rain_flag")
    )
    pdf = agg.toPandas()

    fig = plt.figure()
    for svc in sorted(pdf["service_type"].unique()):
        sub = pdf[pdf["service_type"] == svc]
        for rf, label in [(0, "no_rain"), (1, "rain")]:
            s2 = sub[sub["rain_flag"] == rf]
            if len(s2) == 0:
                continue
            plt.plot(s2["hour"], s2["avg_num_trips"], label=f"{svc} | {label}")

    plt.title(f"{year} | Demanda media por hora condicionada por lluvia")
    plt.xlabel("Hora")
    plt.ylabel("Avg num_trips")
    plt.xticks(range(0, 24, 2))
    plt.legend()
    save_fig(fig, f"outputs/viz_tlc/09_{year}_demand_by_hour_rainflag.png")


def plot_price_focus_zones_rain(dfj, year: int, zones):
    dfz = dfj.where(F.col("pu_location_id").isin(zones))
    agg = (
        dfz.groupBy("pu_location_id", "service_type", "hour", "rain_bin")
        .agg(F.avg("avg_price").alias("avg_price2"))
        .orderBy("pu_location_id", "service_type", "hour")
    )
    pdf = agg.toPandas()

    for z in zones:
        fig = plt.figure()
        subz = pdf[pdf["pu_location_id"] == z]
        for svc in sorted(subz["service_type"].unique()):
            subsvc = subz[subz["service_type"] == svc]
            for rb in ["no_rain", "heavy"]:
                s2 = subsvc[subsvc["rain_bin"] == rb]
                if len(s2) == 0:
                    continue
                plt.plot(s2["hour"], s2["avg_price2"], label=f"{svc} | {rb}")

        plt.title(f"{year} | Zona {z} | Precio medio por hora | no_rain vs heavy")
        plt.xlabel("Hora")
        plt.ylabel("Avg price")
        plt.xticks(range(0, 24, 2))
        plt.legend()
        save_fig(fig, f"outputs/viz_tlc/10_{year}_price_by_hour_rainbin_zone_{z}.png")


def plot_demand_by_hour_eventlevel(dfj, year: int):
    agg = (
        dfj.groupBy("service_type", "hour", "event_level")
        .agg(F.avg("num_trips").alias("avg_num_trips"))
        .orderBy("service_type", "hour")
    )
    pdf = agg.toPandas()

    fig = plt.figure()
    levels = ["none", "low", "mid", "high"]
    for svc in sorted(pdf["service_type"].unique()):
        sub = pdf[pdf["service_type"] == svc]
        for lvl in levels:
            s2 = sub[sub["event_level"] == lvl]
            if len(s2) == 0:
                continue
            plt.plot(s2["hour"], s2["avg_num_trips"], label=f"{svc} | {lvl}")

    plt.title(f"{year} | Demanda media por hora condicionada por intensidad de eventos (borough)")
    plt.xlabel("Hora")
    plt.ylabel("Avg num_trips")
    plt.xticks(range(0, 24, 2))
    plt.legend()
    save_fig(fig, f"outputs/viz_tlc/11_{year}_demand_by_hour_eventlevel.png")


def plot_demand_by_hour_rain_x_events(dfj, year: int):
    df2 = dfj.withColumn("event_flag", F.coalesce(F.col("event_flag"), F.lit(0)).cast("int"))

    agg = (
        df2.groupBy("service_type", "hour", "rain_flag", "event_flag")
        .agg(F.avg("num_trips").alias("avg_num_trips"))
        .orderBy("service_type", "hour", "rain_flag", "event_flag")
    )
    pdf = agg.toPandas()

    fig = plt.figure()
    for svc in sorted(pdf["service_type"].unique()):
        sub = pdf[pdf["service_type"] == svc]
        for rf, rlab in [(0, "no_rain"), (1, "rain")]:
            for ef, elab in [(0, "no_events"), (1, "events")]:
                s2 = sub[(sub["rain_flag"] == rf) & (sub["event_flag"] == ef)]
                if len(s2) == 0:
                    continue
                plt.plot(s2["hour"], s2["avg_num_trips"], label=f"{svc} | {rlab} | {elab}")

    plt.title(f"{year} | Demanda media por hora: lluvia × eventos (borough)")
    plt.xlabel("Hora")
    plt.ylabel("Avg num_trips")
    plt.xticks(range(0, 24, 2))
    plt.legend()
    save_fig(fig, f"outputs/viz_tlc/12_{year}_demand_by_hour_rain_x_events.png")


def main(year: int):
    spark = get_spark(f"FaseB-Viz-09-Meteo-Eventos-{year}")

    # TLC capa3: usamos df2b (zona+hora+día+servicio) porque es perfecto para joins
    _, _, df2b, _ = read_capa3(spark)

    base = (
        df2b.select(
            F.to_date("date").alias("date"),
            F.col("hour").cast("int").alias("hour"),
            F.col("pu_location_id").cast("int").alias("pu_location_id"),
            F.col("service_type"),
            F.col("num_trips").cast("double").alias("num_trips"),
            F.col("avg_price").cast("double").alias("avg_price"),
        )
        .where(F.year("date") == year)
    )

    # Necesitamos borough para cruzar eventos
    zones = (
        spark.read.option("header", True).csv("data/external/taxi_zone_lookup.csv")
        .select(
            F.col("LocationID").cast("int").alias("pu_location_id"),
            F.col("Borough").alias("pu_borough"),
        )
    )
    base = base.join(F.broadcast(zones), on="pu_location_id", how="left")

    meteo = read_meteo_c3(spark, year)
    events = read_events_c3(spark, year)

    dfj = (
        base.join(F.broadcast(meteo), on=["date", "hour"], how="left")
            .join(F.broadcast(events), on=["pu_borough", "date", "hour"], how="left")
            .fillna({"event_flag": 0, "event_level": "none", "n_events": 0, "n_event_types": 0})
            .cache()
    )
    dfj.count()

    plot_demand_by_hour_rainflag(dfj, year)
    plot_price_focus_zones_rain(dfj, year, FOCUS_ZONES)
    plot_demand_by_hour_eventlevel(dfj, year)
    plot_demand_by_hour_rain_x_events(dfj, year)

    dfj.unpersist()
    spark.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Viz TLC + Meteo + Eventos (por año).")
    p.add_argument("--year", type=int, default=2024, help="Año a analizar (default: 2024)")
    args = p.parse_args()
    main(args.year)
