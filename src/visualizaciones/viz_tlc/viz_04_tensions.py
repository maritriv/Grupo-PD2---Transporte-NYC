# notebooks/faseB/viz_04_tensions.py
"""
VIZ 04 — TENSIONES: VOLUMEN vs VARIABILIDAD (POR AÑO)

FINALIDAD
---------
Este script busca “tensiones” u oportunidades operativas/comerciales combinando:
- volumen (num_trips)
- variabilidad de precio (price_variability, p.ej. IQR)
por servicio, zona y hora (según el dataframe de variabilidad).

La idea es detectar combinaciones zona-hora con:
- suficiente volumen para importar,
- y suficiente variabilidad para que haya “espacio” de optimización.

CONTROL DEL AÑO
---------------
El año se controla con --year (por defecto 2024).
Nota: el filtrado por año solo se aplica si el dataframe df3 incluye columna "date".
Si no existe, el script avisa y trabaja con el agregado disponible.

VISUALIZACIONES QUE CREA (y qué pregunta responde)
--------------------------------------------------
(1) Scatter: volumen vs variabilidad (top N por volumen)
    - ¿Dónde hay simultáneamente mucho volumen y mucha variabilidad?
    - ¿Se comportan distinto los servicios (yellow/green/fhvhv)?

(2) Ranking Top 15 por “biz_score”
    - ¿Cuáles son las combinaciones zona-hora-servicio con mayor “oportunidad”
      bajo una métrica simple (biz_score)?
    - ¿Qué zonas/horas conviene mirar primero para decisiones de negocio?

NOTAS
-----
Se limita el scatter a top N por volumen para que sea legible.
"""

from __future__ import annotations

import argparse
import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from src.visualizaciones.viz_tlc.viz_common import get_spark, read_capa3, save_fig


def main(year: int):
    spark = get_spark(f"FaseB-Viz-04-{year}")
    _, _, _, df3 = read_capa3(spark)

    # Filtra por año SOLO si df3 tiene columna "date"
    if "year" in df3.columns:
        df3 = df3.where(F.col("year") == year)
    elif "date" in df3.columns:
        df3 = df3.where(F.year("date") == year)
    else:
        print("⚠️ df3 no tiene 'year' ni 'date' -> no se puede filtrar por año.")


    # Scatter: volumen vs variabilidad (muestra top por volumen para que sea legible)
    top = (
        df3.orderBy(F.desc("num_trips"))
        .limit(1500)  # ajusta si quieres
        .select("num_trips", "price_variability", "service_type")
        .toPandas()
    )

    fig = plt.figure()
    for svc, g in top.groupby("service_type"):
        plt.scatter(
            g["num_trips"],
            g["price_variability"],
            label=svc,
            alpha=0.12,      # más transparente
            s=18,            # puntos pequeños
            linewidths=0,
            rasterized=True, # mejora rendimiento y legibilidad
        )

    plt.title(f"{year} | Tensión: volumen vs variabilidad (IQR) [top 1500 por volumen]")
    plt.xlabel("num_trips")
    plt.ylabel("price_variability (IQR)")
    plt.legend()
    plt.xscale("log")
    plt.xlabel("num_trips (log)")
    save_fig(fig, f"outputs/viz_tlc/07_{year}_scatter_volume_vs_variability.png")

    # Ranking negocio: top 15 (gráfico de barras)
    rank = (
        df3.orderBy(F.desc("biz_score"))
        .limit(15)
        .select("pu_location_id", "hour", "service_type", "biz_score", "num_trips", "price_variability", "avg_price")
        .toPandas()
    )

    labels = [f"{r.pu_location_id}-h{int(r.hour)}-{r.service_type}" for r in rank.itertuples(index=False)]

    fig2 = plt.figure()
    plt.barh(labels, rank["biz_score"])
    plt.title(f"{year} | Top 15 oportunidades (biz_score = variabilidad * log(1+volumen))")
    plt.xlabel("biz_score")
    plt.gca().invert_yaxis()
    save_fig(fig2, f"outputs/viz_tlc/08_{year}_top15_biz_score.png")

    spark.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Viz 04 - Tensions (por año)")
    p.add_argument("--year", type=int, default=2024, help="Año a analizar (default: 2024)")
    args = p.parse_args()
    main(args.year)
