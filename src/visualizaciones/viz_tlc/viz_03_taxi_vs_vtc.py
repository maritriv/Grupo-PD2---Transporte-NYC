# notebooks/faseB/viz_03_taxi_vs_vtc.py
"""
VIZ 03 — TAXI VS VTC EN ZONAS CLAVE (POR AÑO)

FINALIDAD
---------
Este script compara patrones horarios entre tipos de servicio en “zonas importantes”,
seleccionadas por volumen dentro de un servicio específico (fhvhv).

Sirve para ver de forma clara:
- cómo cambia la demanda por hora
- cómo cambia el precio por hora
y si la dinámica es distinta entre servicios (taxi vs VTC).

CONTROL DEL AÑO
---------------
El año se controla con --year (por defecto 2024). La selección de zonas “top”
y los promedios horarios se calculan únicamente con datos del año elegido.

VISUALIZACIONES QUE CREA (y qué pregunta responde)
--------------------------------------------------
(1) Demanda media por hora en 3 zonas top (líneas por service_type)
    - ¿En las zonas con mayor volumen, qué servicio domina en cada hora?
    - ¿Hay horas donde la demanda de un servicio supera claramente a los demás?

(2) Precio medio por hora en esas mismas 3 zonas (líneas por service_type)
    - ¿Hay diferencias sistemáticas de precio entre servicios en zonas clave?
    - ¿Aparecen “picos” horarios de precio distintos según el servicio?

SALIDAS
-------
Genera 3 figuras por (1) y 3 figuras por (2), una por cada zona top.
"""

from __future__ import annotations

import argparse
import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from src.visualizaciones.viz_tlc.viz_common import get_spark, read_capa3, save_fig


def main(year: int):
    spark = get_spark(f"FaseB-Viz-03-{year}")
    _, _, df2b, _ = read_capa3(spark)

    # ✅ SOLO el año elegido (por defecto 2024)
    df2b = df2b.where(F.year("date") == year)

    # Elegimos 3 zonas “importantes” por volumen (en fhvhv)
    zone_rank = (
        df2b.filter(F.col("service_type") == "fhvhv")
        .groupBy("pu_location_id")
        .agg(F.sum("num_trips").alias("total_trips"))
        .orderBy(F.desc("total_trips"))
        .limit(3)
        .select("pu_location_id")
        .toPandas()["pu_location_id"]
        .tolist()
    )

    # Agregamos por hora y servicio en esas zonas
    hourly = (
        df2b.filter(F.col("pu_location_id").isin(zone_rank))
        .groupBy("pu_location_id", "hour", "service_type")
        .agg(
            F.avg("num_trips").alias("avg_num_trips"),
            F.avg("avg_price").alias("avg_price"),
        )
        .orderBy("pu_location_id", "hour", "service_type")
        .toPandas()
    )

    # Plot 1: demanda por hora (comparando servicios) en cada zona (3 figuras)
    for zid in zone_rank:
        z = hourly[hourly["pu_location_id"] == zid]

        fig = plt.figure()
        for svc, g in z.groupby("service_type"):
            plt.plot(g["hour"], g["avg_num_trips"], label=svc)

        plt.title(f"{year} | Demanda por hora (media) en zona {zid} | Taxi vs VTC")
        plt.xlabel("Hora")
        plt.ylabel("Avg num_trips")
        plt.xticks(range(0, 24, 2))
        plt.legend()
        save_fig(fig, f"outputs/viz_tlc/05_{year}_zone_{zid}_demand_by_hour_service.png")

    # Plot 2: precio por hora (comparando servicios) en cada zona
    for zid in zone_rank:
        z = hourly[hourly["pu_location_id"] == zid]

        fig = plt.figure()
        for svc, g in z.groupby("service_type"):
            plt.plot(g["hour"], g["avg_price"], label=svc)

        plt.title(f"{year} | Precio por hora (medio) en zona {zid} | Taxi vs VTC")
        plt.xlabel("Hora")
        plt.ylabel("Avg price")
        plt.xticks(range(0, 24, 2))
        plt.legend()
        save_fig(fig, f"outputs/viz_tlc/06_{year}_zone_{zid}_price_by_hour_service.png")

    spark.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Viz 03 - Taxi vs VTC (por año)")
    p.add_argument("--year", type=int, default=2024, help="Año a analizar (default: 2024)")
    args = p.parse_args()
    main(args.year)
