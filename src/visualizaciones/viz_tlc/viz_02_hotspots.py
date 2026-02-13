# notebooks/faseB/viz_02_hotspots.py
"""
VIZ 02 — HOTSPOTS (ZONA x HORA) POR AÑO

FINALIDAD
---------
Este script identifica patrones “espacio-temporales” simples:
qué zonas (pickup zones) tienen más actividad y cómo varían por hora del día,
diferenciando entre demanda (num_trips) y precio medio (avg_price).

No es un análisis causal: es un mapa exploratorio para detectar “hotspots”
y horarios característicos.

CONTROL DEL AÑO
---------------
El año se controla con --year (por defecto 2024). Los agregados se calculan
solo con registros de ese año para no mezclar cambios de movilidad entre años.

VISUALIZACIONES QUE CREA (y qué pregunta responde)
--------------------------------------------------
(1) Heatmap: demanda media por zona y hora (TOP 60 zonas por demanda)
    - ¿Cuáles son las zonas más activas?
    - ¿En qué horas se concentran los viajes dentro de esas zonas?

(2) Heatmap: precio medio por zona y hora (TOP 60 zonas por demanda)
    - ¿Hay zonas/horas con precios sistemáticamente más altos?
    - ¿Coinciden los picos de precio con los picos de demanda?

NOTAS
-----
Para evitar un heatmap inmanejable, se limita a TOP 60 zonas por demanda media
(la selección de zonas se calcula dentro del año elegido).
"""

from __future__ import annotations

import argparse
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


def main(year: int):
    spark = get_spark(f"FaseB-Viz-02-{year}")
    _, df2a, _, _ = read_capa3(spark)

    # ✅ SOLO el año elegido (por defecto 2024)
    df2a = df2a.where(F.year("date") == year)

    # Reducimos: media por zona y hora (sobre todos los días del año)
    base = (
        df2a.groupBy("pu_location_id", "hour")
        .agg(
            F.avg("num_trips").alias("avg_num_trips"),
            F.avg("avg_price").alias("avg_price"),
        )
    )

    # Para que no sea un heatmap gigante: TOP 60 zonas por demanda media
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
        title=f"Hotspots: demanda media (num_trips) por zona y hora [TOP 60] | {year}",
        value_col="avg_num_trips",
    )
    save_fig(fig1, f"outputs/viz_tlc/03_heatmap_demand_zone_hour_{year}.png")

    fig2 = make_heatmap_pivot(
        pdf,
        title=f"Hotspots: precio medio (avg_price) por zona y hora [TOP 60] | {year}",
        value_col="avg_price",
    )
    save_fig(fig2, f"outputs/viz_tlc/04_heatmap_price_zone_hour_{year}.png")

    spark.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Viz 02 - Hotspots (por año)")
    p.add_argument("--year", type=int, default=2024, help="Año a analizar (default: 2024)")
    args = p.parse_args()
    main(args.year)
