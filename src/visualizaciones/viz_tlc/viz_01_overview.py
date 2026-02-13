# notebooks/faseB/viz_01_overview.py
"""
VIZ 01 — OVERVIEW (TLC) POR AÑO

FINALIDAD
---------
Este script genera una visión general (exploratoria) del transporte urbano en NYC
a lo largo de un año concreto, diferenciando por tipo de servicio (yellow, green, fhvhv).

No es un modelo predictivo ni un pipeline de producción: su objetivo es
entender tendencias temporales de nivel “macro” y comparar servicios.

CONTROL DEL AÑO
---------------
El año se controla con --year (por defecto 2024). Todas las series se filtran
a ese año para evitar mezclar patrones de distintos años.

VISUALIZACIONES QUE CREA (y qué pregunta responde)
--------------------------------------------------
(1) Serie temporal: número de viajes diario por servicio
    - ¿Cómo evoluciona la demanda a lo largo del año?
    - ¿Se observan estacionalidades (picos/vales) y diferencias entre servicios?

(2) Serie temporal: precio medio diario por servicio
    - ¿Cómo evoluciona el precio medio a lo largo del año?
    - ¿Hay periodos con “premium” o bajadas de precio y se comportan igual los servicios?

SALIDAS
-------
Guarda imágenes en outputs/viz_tlc/ con sufijo del año (_{year}).
"""

from __future__ import annotations

import argparse
import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from src.visualizaciones.viz_tlc.viz_common import get_spark, read_capa3, save_fig, ensure_local_date


def plot_num_trips(df_daily):
    pdf = (
        ensure_local_date(df_daily, "date")
        .select("date", "service_type", "num_trips")
        .orderBy("date")
        .toPandas()
    )

    services = sorted(pdf["service_type"].unique())
    fig, axes = plt.subplots(len(services), 1, sharex=True, figsize=(10, 7))

    if len(services) == 1:
        axes = [axes]

    for ax, svc in zip(axes, services):
        g = pdf[pdf["service_type"] == svc]
        ax.plot(g["date"], g["num_trips"])
        ax.set_title(f"Viajes diarios - {svc}")
        ax.set_ylabel("num_trips")
    
    for label in axes[-1].get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

    axes[-1].set_xlabel("Fecha")
    fig.suptitle("Número de viajes diario por servicio", y=0.98)
    fig.tight_layout()
    return fig



def plot_avg_price(df_daily):
    pdf = (
        ensure_local_date(df_daily, "date")
        .select("date", "service_type", "avg_price")
        .orderBy("date")
        .toPandas()
    )

    fig = plt.figure()
    for svc, g in pdf.groupby("service_type"):
        plt.plot(g["date"], g["avg_price"], label=svc)

    plt.title("Precio medio diario por servicio (total_amount_std)")
    plt.xlabel("Fecha")
    plt.ylabel("Precio medio (avg_price)")
    plt.legend()
    return fig


def main(year: int):
    spark = get_spark(f"FaseB-Viz-01-{year}")
    df1, _, _, _ = read_capa3(spark)

    # ✅ SOLO el año elegido (por defecto 2024)
    df1 = df1.where(F.year("date") == year)

    fig1 = plot_num_trips(df1)
    save_fig(fig1, f"outputs/viz_tlc/01_num_trips_by_service_{year}.png")

    fig2 = plot_avg_price(df1)
    save_fig(fig2, f"outputs/viz_tlc/02_avg_price_by_service_{year}.png")

    spark.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Viz 01 - Overview (por año)")
    p.add_argument("--year", type=int, default=2024, help="Año a analizar (default: 2024)")
    args = p.parse_args()
    main(args.year)
