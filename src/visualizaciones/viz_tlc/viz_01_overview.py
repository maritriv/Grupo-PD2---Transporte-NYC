# notebooks/faseB/viz_01_overview.py
from __future__ import annotations

import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from src.visualizaciones.viz_tlc.viz_common import get_spark, read_capa3, save_fig, ensure_local_date


def plot_num_trips(df_daily):
    # df_daily: date, service_type, num_trips
    pdf = (
        ensure_local_date(df_daily, "date")
        .select("date", "service_type", "num_trips")
        .orderBy("date")
        .toPandas()
    )

    fig = plt.figure()
    for svc, g in pdf.groupby("service_type"):
        plt.plot(g["date"], g["num_trips"], label=svc)

    plt.title("Número de viajes diario por servicio")
    plt.xlabel("Fecha")
    plt.ylabel("Viajes (num_trips)")
    plt.legend()
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


def main():
    spark = get_spark("FaseB-Viz-01")
    df1, _, _, _ = read_capa3(spark)

    fig1 = plot_num_trips(df1)
    save_fig(fig1, "outputs/viz_tlc/01_num_trips_by_service.png")

    fig2 = plot_avg_price(df1)
    save_fig(fig2, "outputs/viz_tlc/02_avg_price_by_service.png")

    spark.stop()


if __name__ == "__main__":
    main()
