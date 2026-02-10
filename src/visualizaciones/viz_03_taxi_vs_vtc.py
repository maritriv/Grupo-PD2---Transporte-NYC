# notebooks/faseB/viz_03_taxi_vs_vtc.py
from __future__ import annotations

import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from notebooks.faseB.viz_common import get_spark, read_capa3, save_fig


def main():
    spark = get_spark("FaseB-Viz-03")
    _, _, df2b, _ = read_capa3(spark)

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

        plt.title(f"Demanda por hora (media) en zona {zid} | Taxi vs VTC")
        plt.xlabel("Hora")
        plt.ylabel("Avg num_trips")
        plt.xticks(range(0, 24, 2))
        plt.legend()
        save_fig(fig, f"outputs/faseB/05_zone_{zid}_demand_by_hour_service.png")

    # Plot 2: precio por hora (comparando servicios) en cada zona
    for zid in zone_rank:
        z = hourly[hourly["pu_location_id"] == zid]

        fig = plt.figure()
        for svc, g in z.groupby("service_type"):
            plt.plot(g["hour"], g["avg_price"], label=svc)

        plt.title(f"Precio por hora (medio) en zona {zid} | Taxi vs VTC")
        plt.xlabel("Hora")
        plt.ylabel("Avg price")
        plt.xticks(range(0, 24, 2))
        plt.legend()
        save_fig(fig, f"outputs/faseB/06_zone_{zid}_price_by_hour_service.png")

    spark.stop()


if __name__ == "__main__":
    main()
