"""
VISUALIZACIONES TLC + METEO + EVENTOS (POR AÑO)

FINALIDAD DEL SCRIPT
--------------------
Este script genera visualizaciones exploratorias para entender cómo factores externos
(lluvia y eventos urbanos) se asocian con cambios en:
- la demanda (num_trips)
- y el precio medio (avg_price)
del transporte urbano en NYC, diferenciando por hora y por tipo de servicio
(yellow, green, fhvhv).

NO es un script de predicción ni un pipeline de producción: está diseñado para
explicar patrones de forma clara y eficiente.

CONTROL DEL AÑO ANALIZADO
-------------------------
El año se elige con --year (por defecto 2024). El script filtra TLC, meteo y eventos
a ese año, así que el análisis depende del año que tú decidas, no de “lo que haya”
en disco.

VISUALIZACIONES QUE CREA (y qué pregunta responde cada una)
-----------------------------------------------------------
(1) Demanda vs lluvia por hora (por servicio)
    - ¿Llueve => cambia la demanda? ¿Afecta distinto a cada servicio?

(2) % cambio de demanda por lluvia (por servicio)
    - Cuantifica el efecto relativo: “cuando llueve, la demanda sube/baja X%”
      (más defendible que solo niveles absolutos).

(3) Precio por hora en zonas foco: no_rain vs heavy
    - ¿Existe “weather premium” (subida de precio con lluvia fuerte) en hotspots concretos?
      Se muestran 2 zonas representativas (una sensible y otra menos sensible).

(4) Demanda vs eventos por hora (binario: events vs no_events)
    - ¿La presencia de eventos en el borough está asociada a más demanda?
      Versión binaria para máxima claridad.

REQUISITOS / RUTAS
------------------
- TLC capa3 disponible (read_capa3)
- Meteo capa3: data/external/meteo/aggregated/df_hour_day/data.parquet
- Eventos capa3: data/external/events/aggregated/df_borough_hour_day
- Lookup zonas: data/external/taxi_zone_lookup.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path  # <-- AÑADIDO: para crear carpeta de salida

import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import functions as F

from src.visualizaciones.viz_tlc.viz_common import get_spark, read_capa3, save_fig


# =========================
# CONFIG
# =========================
METEO_C3_PATH = "data/external/meteo/aggregated/df_hour_day/data.parquet"
EVENTS_C3_PATH = "data/external/events/aggregated/df_borough_hour_day"
ZONES_LOOKUP_CSV = "data/external/taxi_zone_lookup.csv"

# Zonas foco (elige 2: una “sensible” y otra “menos sensible”)
DEFAULT_FOCUS_ZONES = [79, 138]

# Bins lluvia
RAIN_LIGHT_MAX = 2.0
RAIN_MODERATE_MAX = 8.0


# =========================
# READERS
# =========================
def read_meteo_c3(spark, year: int):
    """
    Meteo capa3 (date+hour) con agregados:
      - rain_mm_sum, precip_mm_sum, temp_c_mean, wind_kmh_mean (según tu capa3_meteo)
    """
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
            .otherwise(F.lit("heavy")),
        )
    )
    return m


def read_events_c3(spark, year: int):
    """
    Eventos capa3 (borough+date+hour): df_borough_hour_day
    """
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
    )
    return e


def read_zones_lookup(spark):
    return (
        spark.read.option("header", True).csv(ZONES_LOOKUP_CSV)
        .select(
            F.col("LocationID").cast("int").alias("pu_location_id"),
            F.col("Borough").alias("pu_borough"),
        )
    )


# =========================
# PLOTS (pandas-level)
# =========================
def plot_demand_by_hour_rainflag(pdf, year: int, outpath: str):
    fig = plt.figure()

    # pdf columns: service_type, hour, rain_flag, avg_num_trips
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
    save_fig(fig, outpath)


def plot_pct_change_demand_rain(pdf_raw, year: int, outpath: str):
    """
    % cambio = (rain - no_rain) / no_rain * 100
    pdf_raw columns: service_type, hour, rain_flag, avg_num_trips
    """
    # Pivot: columns 0/1 -> no_rain / rain
    piv = (
        pdf_raw.pivot_table(
            index=["service_type", "hour"],
            columns="rain_flag",
            values="avg_num_trips",
            aggfunc="mean",
        )
        .dropna()
        .reset_index()
    )

    # columns 0 and 1 expected
    if 0 not in piv.columns or 1 not in piv.columns:
        # if one class missing, just create empty plot
        fig = plt.figure()
        plt.title(f"{year} | % cambio de demanda por lluvia (insuficientes datos)")
        save_fig(fig, outpath)
        return

    piv["pct_change"] = (piv[1] - piv[0]) / np.where(piv[0] == 0, np.nan, piv[0]) * 100

    fig = plt.figure()
    for svc in sorted(piv["service_type"].unique()):
        sub = piv[piv["service_type"] == svc]
        plt.plot(sub["hour"], sub["pct_change"], label=svc)

    plt.axhline(0, linestyle="--", alpha=0.5)
    plt.title(f"{year} | % cambio de demanda por lluvia")
    plt.xlabel("Hora")
    plt.ylabel("% cambio (rain vs no_rain)")
    plt.xticks(range(0, 24, 2))
    plt.legend()
    save_fig(fig, outpath)


def plot_price_focus_zones_no_rain_vs_heavy(pdf, year: int, zone_id: int, outpath: str):
    """
    pdf columns: pu_location_id, service_type, hour, rain_bin, avg_price2
    """
    fig = plt.figure()
    subz = pdf[pdf["pu_location_id"] == zone_id]

    for svc in sorted(subz["service_type"].unique()):
        subsvc = subz[subz["service_type"] == svc]
        for rb in ["no_rain", "heavy"]:
            s2 = subsvc[subsvc["rain_bin"] == rb]
            if len(s2) == 0:
                continue
            plt.plot(s2["hour"], s2["avg_price2"], label=f"{svc} | {rb}")

    plt.title(f"{year} | Zona {zone_id} | Precio medio por hora | no_rain vs heavy")
    plt.xlabel("Hora")
    plt.ylabel("Avg price")
    plt.xticks(range(0, 24, 2))
    plt.legend()
    save_fig(fig, outpath)


def plot_demand_by_hour_events_binary(pdf, year: int, outpath: str):
    """
    pdf columns: service_type, hour, event_flag, avg_num_trips
    """
    fig = plt.figure()

    for svc in sorted(pdf["service_type"].unique()):
        sub = pdf[pdf["service_type"] == svc]
        for ev, label in [(0, "no_events"), (1, "events")]:
            s2 = sub[sub["event_flag"] == ev]
            if len(s2) == 0:
                continue
            plt.plot(s2["hour"], s2["avg_num_trips"], label=f"{svc} | {label}")

    plt.title(f"{year} | Demanda media por hora: eventos vs no eventos (borough)")
    plt.xlabel("Hora")
    plt.ylabel("Avg num_trips")
    plt.xticks(range(0, 24, 2))
    plt.legend()
    save_fig(fig, outpath)


# =========================
# MAIN
# =========================
def main(year: int, focus_zones: list[int]):
    # crea carpeta de salida si no existe (y si existe, no falla)
    Path("outputs/viz_conjuntas").mkdir(parents=True, exist_ok=True)

    spark = get_spark(f"Viz-Meteo-Eventos-{year}")

    # TLC capa3: usamos df2b (zona+hora+día+servicio)
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

    # borough para unir eventos
    zones = read_zones_lookup(spark)
    base = base.join(F.broadcast(zones), on="pu_location_id", how="left")

    meteo = read_meteo_c3(spark, year)
    events = read_events_c3(spark, year)

    # joins ligeros
    dfj = (
        base.join(F.broadcast(meteo), on=["date", "hour"], how="left")
        .join(F.broadcast(events), on=["pu_borough", "date", "hour"], how="left")
        .fillna({"event_flag": 0, "n_events": 0, "n_event_types": 0})
        .cache()
    )
    dfj.count()  # materializa una vez

    # -------------------------
    # (1) Demanda vs lluvia (absoluto)
    # -------------------------
    rain_demand = (
        dfj.groupBy("service_type", "hour", "rain_flag")
        .agg(F.avg("num_trips").alias("avg_num_trips"))
        .orderBy("service_type", "hour", "rain_flag")
    )
    pdf_rain_demand = rain_demand.toPandas()
    plot_demand_by_hour_rainflag(
        pdf_rain_demand,
        year,
        outpath=f"outputs/viz_conjuntas/09_{year}_demand_by_hour_rainflag.png",
    )

    # -------------------------
    # (2) % cambio de demanda por lluvia
    # -------------------------
    plot_pct_change_demand_rain(
        pdf_rain_demand,
        year,
        outpath=f"outputs/viz_conjuntas/09b_{year}_pct_change_demand_rain.png",
    )

    # -------------------------
    # (3) Precio por hora en 2 zonas (no_rain vs heavy)
    # -------------------------
    df_price = (
        dfj.where(F.col("pu_location_id").isin(focus_zones))
        .groupBy("pu_location_id", "service_type", "hour", "rain_bin")
        .agg(F.avg("avg_price").alias("avg_price2"))
        .orderBy("pu_location_id", "service_type", "hour")
    )
    pdf_price = df_price.toPandas()

    for z in focus_zones:
        plot_price_focus_zones_no_rain_vs_heavy(
            pdf_price,
            year,
            zone_id=z,
            outpath=f"outputs/viz_conjuntas/10_{year}_price_by_hour_no_rain_vs_heavy_zone_{z}.png",
        )

    # -------------------------
    # (4) Eventos binario: demanda vs eventos
    # -------------------------
    ev_demand = (
        dfj.groupBy("service_type", "hour", "event_flag")
        .agg(F.avg("num_trips").alias("avg_num_trips"))
        .orderBy("service_type", "hour", "event_flag")
    )
    pdf_ev_demand = ev_demand.toPandas()
    plot_demand_by_hour_events_binary(
        pdf_ev_demand,
        year,
        outpath=f"outputs/viz_conjuntas/11_{year}_demand_by_hour_events_binary.png",
    )

    dfj.unpersist()
    spark.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Viz TLC + Meteo + Eventos (por año).")
    p.add_argument("--year", type=int, default=2024, help="Año a analizar (default: 2024)")
    p.add_argument(
        "--zones",
        type=int,
        nargs="+",
        default=DEFAULT_FOCUS_ZONES,
        help="Lista de 2 zonas foco (pu_location_id). Default: 79 138",
    )
    args = p.parse_args()

    # Forzamos 2 zonas para que sea legible (si te pasan 3+, recortamos)
    zones = list(args.zones)[:2]
    main(args.year, zones)
