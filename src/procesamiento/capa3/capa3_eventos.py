# src/procesamiento/capa3/capa3_eventos.py
from __future__ import annotations

import findspark
findspark.init()

from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from config.settings import obtener_ruta

DEBUG = False  # True para ver previews


# ---------------------------------------------------------------------
# Spark
# ---------------------------------------------------------------------
def get_spark(app_name: str = "PD2-Capa3-Eventos"):
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "6g")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )
    spark.conf.set("spark.sql.session.timeZone", "America/New_York")
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ---------------------------------------------------------------------
# Lectura capa 2 eventos + higiene mínima
# ---------------------------------------------------------------------
def read_layer2_events(
    spark,
    layer2_path = obtener_ruta("data/external/events") / "standarized",
    min_date: str = "2019-01-01",
    max_date: str = "2025-12-31",
):
    df = spark.read.parquet(str(layer2_path))

    df = df.filter(
        F.col("date").isNotNull()
        & F.col("hour").isNotNull()
        & F.col("borough").isNotNull()
        & F.col("event_type").isNotNull()
        & F.col("n_events").isNotNull()
    )

    df = df.filter(F.col("date").between(F.lit(min_date), F.lit(max_date)))
    df = df.filter((F.col("hour") >= 0) & (F.col("hour") <= 23))
    df = df.filter(F.col("n_events") >= 0)

    return df


# ---------------------------------------------------------------------
# Construcción CAPA 3 eventos
# ---------------------------------------------------------------------
def build_layer3_events(
    df_capa2_events,
    min_events_hour_day: int = 1,   # mínimo para quedarte con buckets con señal
):
    # 1) Intensidad por borough + date + hour (lo más fácil de cruzar con taxis/VTC)
    df_borough_hour_day = (
        df_capa2_events
        .groupBy("borough", "date", "hour")
        .agg(
            F.sum("n_events").alias("n_events"),
            F.countDistinct("event_type").alias("n_event_types"),
        )
        .filter(F.col("n_events") >= F.lit(min_events_hour_day))
    )

    # 2) Evolución diaria por borough
    df_daily_borough = (
        df_borough_hour_day
        .groupBy("borough", "date")
        .agg(
            F.sum("n_events").alias("n_events"),
            F.sum("n_event_types").alias("n_event_types_sum"),  # suma de tipos por hora (proxy)
            F.max("n_events").alias("max_events_in_an_hour"),
        )
    )

    # 3) Ranking de tipos (borough + date) para análisis cualitativo
    df_type_daily_borough = (
        df_capa2_events
        .groupBy("borough", "date", "event_type")
        .agg(F.sum("n_events").alias("n_events"))
    )

    # 4) Patrón horario medio (borough + hour): media de eventos a esa hora (a través de días)
    df_hourly_pattern = (
        df_borough_hour_day
        .groupBy("borough", "hour")
        .agg(
            F.avg("n_events").alias("avg_events"),
            F.stddev("n_events").alias("std_events"),
            F.countDistinct("date").alias("n_days"),
        )
    )

    return df_borough_hour_day, df_daily_borough, df_type_daily_borough, df_hourly_pattern


# ---------------------------------------------------------------------
# Guardado capa 3 eventos
# ---------------------------------------------------------------------
def save_layer3_events(
    df_borough_hour_day,
    df_daily_borough,
    df_type_daily_borough,
    df_hourly_pattern,
    out_base: Path = obtener_ruta("data/external/events") / "aggregated",
):
    (
        df_borough_hour_day
        .write
        .mode("overwrite")
        .partitionBy("date", "borough")
        .parquet(str(out_base / "df_borough_hour_day"))
    )

    (
        df_daily_borough
        .write
        .mode("overwrite")
        .partitionBy("borough")
        .parquet(str(out_base / "df_daily_borough"))
    )

    (
        df_type_daily_borough
        .write
        .mode("overwrite")
        .partitionBy("date", "borough")
        .parquet(str(out_base / "df_type_daily_borough"))
    )

    (
        df_hourly_pattern
        .write
        .mode("overwrite")
        .partitionBy("borough")
        .parquet(str(out_base / "df_hourly_pattern"))
    )

    print("\nCapa 3 EVENTOS guardada en:", out_base)
    print(" - df_borough_hour_day     ->", out_base / "df_borough_hour_day")
    print(" - df_daily_borough        ->", out_base / "df_daily_borough")
    print(" - df_type_daily_borough   ->", out_base / "df_type_daily_borough")
    print(" - df_hourly_pattern       ->", out_base / "df_hourly_pattern")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    spark = get_spark()

    df2 = read_layer2_events(
        spark,
        layer2_path = obtener_ruta("data/external/events") / "standarized",
        min_date="2019-01-01",
        max_date="2025-12-31",
    )

    df1, df2b, df3, df4 = build_layer3_events(df2, min_events_hour_day=1)

    if DEBUG:
        print("\n--- df_borough_hour_day sample ---")
        df1.orderBy(F.desc("date"), F.desc("n_events")).show(20, truncate=False)

        print("\n--- df_daily_borough sample ---")
        df2b.orderBy(F.desc("date"), F.desc("n_events")).show(20, truncate=False)

    save_layer3_events(df1, df2b, df3, df4, out_base=obtener_ruta("data/external/events") / "aggregated")
    spark.stop()


if __name__ == "__main__":
    main()
