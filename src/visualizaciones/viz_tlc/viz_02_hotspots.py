# notebooks/faseB/viz_02_hotspots.py
from __future__ import annotations

import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from src.visualizaciones.viz_tlc.viz_common import get_spark, read_capa3, save_fig


def make_heatmap_pivot(pdf, title: str, value_col: str):
    # pdf columns: pu_location_id, hour, value_col
    pivot = pdf.pivot_table(index="pu_location_id", columns="hour", values=value_col, aggfunc="mean")

    fig = plt.figure()
    plt.imshow(pivot.values, aspect="auto")
    plt.title(title)
    plt.xlabel("Hora (0-23)")
    plt.ylabel("pu_location_id (ordenado)")
    plt.xticks(range(0, 24, 2), range(0, 24, 2))
    return fig


def main():
    spark = get_spark("FaseB-Viz-02")
    _, df2a, _, _ = read_capa3(spark)

    # Reducimos: media por zona y hora (sobre todos los días)
    base = (
        df2a.groupBy("pu_location_id", "hour")
        .agg(
            F.avg("num_trips").alias("avg_num_trips"),
            F.avg("avg_price").alias("avg_price"),
        )
    )

    # Para que no sea un heatmap gigante: quédate con TOP 60 zonas por demanda media
    top_zones = (
        base.groupBy("pu_location_id")
        .agg(F.avg("avg_num_trips").alias("z_demand"))
        .orderBy(F.desc("z_demand"))
        .limit(60)
        .select("pu_location_id")
    )

    base_top = base.join(F.broadcast(top_zones), "pu_location_id", "inner")

    pdf = base_top.toPandas()

    fig1 = make_heatmap_pivot(
        pdf,
        title="Hotspots: demanda media (num_trips) por zona y hora [TOP 60 zonas]",
        value_col="avg_num_trips",
    )
    save_fig(fig1, "outputs/viz_tlc/03_heatmap_demand_zone_hour.png")

    fig2 = make_heatmap_pivot(
        pdf,
        title="Hotspots: precio medio (avg_price) por zona y hora [TOP 60 zonas]",
        value_col="avg_price",
    )
    save_fig(fig2, "outputs/viz_tlc/04_heatmap_price_zone_hour.png")

    spark.stop()


if __name__ == "__main__":
    main()
