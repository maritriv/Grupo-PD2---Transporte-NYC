# notebooks/faseB/viz_04_tensions.py
from __future__ import annotations

import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from notebooks.faseB.viz_common import get_spark, read_capa3, save_fig


def main():
    spark = get_spark("FaseB-Viz-04")
    _, _, _, df3 = read_capa3(spark)

    # Scatter: volumen vs variabilidad (muestra top por volumen para que sea legible)
    top = (
        df3.orderBy(F.desc("num_trips"))
        .limit(1500)  # ajusta si quieres
        .select("num_trips", "price_variability", "service_type")
        .toPandas()
    )

    fig = plt.figure()
    for svc, g in top.groupby("service_type"):
        plt.scatter(g["num_trips"], g["price_variability"], label=svc, alpha=0.5)

    plt.title("Tensión: volumen vs variabilidad (IQR) [top 1500 por volumen]")
    plt.xlabel("num_trips")
    plt.ylabel("price_variability (IQR)")
    plt.legend()
    save_fig(fig, "outputs/faseB/07_scatter_volume_vs_variability.png")

    # Ranking negocio: top 15 (gráfico de barras)
    rank = (
        df3.orderBy(F.desc("biz_score"))
        .limit(15)
        .select("pu_location_id", "hour", "service_type", "biz_score", "num_trips", "price_variability", "avg_price")
        .toPandas()
    )

    labels = [
        f"{r.pu_location_id}-h{int(r.hour)}-{r.service_type}"
        for r in rank.itertuples(index=False)
    ]

    fig2 = plt.figure()
    plt.barh(labels, rank["biz_score"])
    plt.title("Top 15 oportunidades (biz_score = variabilidad * log(1+volumen))")
    plt.xlabel("biz_score")
    plt.gca().invert_yaxis()
    save_fig(fig2, "outputs/faseB/08_top15_biz_score.png")

    spark.stop()


if __name__ == "__main__":
    main()
