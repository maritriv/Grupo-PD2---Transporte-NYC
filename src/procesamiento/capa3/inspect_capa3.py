# src/procesamiento/capa3/inspect_capa3.py
from __future__ import annotations
import findspark
findspark.init()

from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from config.settings import obtener_ruta

# ---------------------------------------------------------------------
# Spark
# ---------------------------------------------------------------------
def get_spark(app_name: str = "PD2-Inspect-Capa3"):
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
# Helpers
# ---------------------------------------------------------------------
def header(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def safe_exists(spark: SparkSession, path: str) -> bool:
    try:
        spark.read.parquet(path).limit(1).count()
        return True
    except Exception:
        return False


def basic_profile(df, name: str):
    header(f"{name} | SCHEMA")
    df.printSchema()

    header(f"{name} | CONTEO (rows)")
    print(df.count())

    header(f"{name} | SAMPLE (10 filas)")
    df.show(10, truncate=False)


def null_profile(df, cols: list[str], name: str, sample_fraction: float = 0.001):
    header(f"{name} | NULOS (sobre muestra fraction={sample_fraction})")
    sdf = df.sample(False, sample_fraction, seed=42)

    exprs = [F.sum(F.col(c).isNull().cast("int")).alias(f"null_{c}") for c in cols if c in df.columns]
    if not exprs:
        print("[INFO] No hay columnas para evaluar nulos.")
        return

    sdf.select(*exprs).show(truncate=False)


def temporal_range(df, name: str):
    if "date" not in df.columns:
        header(f"{name} | RANGO TEMPORAL")
        print("[INFO] No existe columna 'date' en este DF.")
        return

    header(f"{name} | RANGO TEMPORAL (min/max date)")
    df.agg(
        F.min("date").alias("min_date"),
        F.max("date").alias("max_date"),
    ).show(truncate=False)


def by_service_counts(df, name: str):
    if "service_type" not in df.columns:
        return
    header(f"{name} | CONTEO POR SERVICIO")
    df.groupBy("service_type").count().orderBy(F.desc("count")).show(truncate=False)


# ---------------------------------------------------------------------
# Inspectores por DF
# ---------------------------------------------------------------------
def inspect_df_daily_service(df):
    name = "DF1 df_daily_service (date + service_type)"
    basic_profile(df, name)
    temporal_range(df, name)
    by_service_counts(df, name)

    header("DF1 | TOP días por num_trips")
    df.orderBy(F.desc("num_trips")).show(15, truncate=False)

    header("DF1 | TOP días por avg_price")
    df.orderBy(F.desc("avg_price")).show(15, truncate=False)

    header("DF1 | TOP días por std_price")
    if "std_price" in df.columns:
        df.orderBy(F.desc("std_price")).show(15, truncate=False)

    null_profile(df, ["date", "service_type", "num_trips", "avg_price", "std_price", "unique_zones"], name)


def inspect_df_zone_hour_day_global(df):
    name = "DF2a df_zone_hour_day_global (pu_location_id + hour + date)"
    basic_profile(df, name)
    temporal_range(df, name)

    header("DF2a | TOP combinaciones por num_trips")
    df.orderBy(F.desc("num_trips")).show(20, truncate=False)

    header("DF2a | TOP combinaciones por avg_price")
    df.orderBy(F.desc("avg_price")).show(20, truncate=False)

    if "std_price" in df.columns:
        header("DF2a | TOP combinaciones por std_price")
        df.orderBy(F.desc("std_price")).show(20, truncate=False)

    header("DF2a | Distribución por hour (conteo de filas)")
    df.groupBy("hour").count().orderBy("hour").show(50, truncate=False)

    header("DF2a | Sanity: horas fuera de [0,23]")
    df.filter((F.col("hour") < 0) | (F.col("hour") > 23)).show(20, truncate=False)

    null_profile(df, ["pu_location_id", "hour", "date", "num_trips", "avg_price", "std_price"], name)


def inspect_df_zone_hour_day_service(df):
    name = "DF2b df_zone_hour_day_service (pu_location_id + hour + date + service_type)"
    basic_profile(df, name)
    temporal_range(df, name)
    by_service_counts(df, name)

    header("DF2b | TOP combinaciones por num_trips")
    df.orderBy(F.desc("num_trips")).show(20, truncate=False)

    header("DF2b | TOP combinaciones por avg_price")
    df.orderBy(F.desc("avg_price")).show(20, truncate=False)

    if "std_price" in df.columns:
        header("DF2b | TOP combinaciones por std_price")
        df.orderBy(F.desc("std_price")).show(20, truncate=False)

    header("DF2b | Sanity: horas fuera de [0,23]")
    df.filter((F.col("hour") < 0) | (F.col("hour") > 23)).show(20, truncate=False)

    null_profile(df, ["pu_location_id", "hour", "date", "service_type", "num_trips", "avg_price", "std_price"], name)


def inspect_df_variability(df):
    name = "DF3 df_variability (IQR) (pu_location_id + hour + service_type)"
    basic_profile(df, name)
    by_service_counts(df, name)

    header("DF3 | TOP por price_variability (IQR) (impredecibles)")
    df.orderBy(F.desc("price_variability")).show(25, truncate=False)

    header("DF3 | TOP por avg_price (caros)")
    df.orderBy(F.desc("avg_price")).show(25, truncate=False)

    header("DF3 | TOP por num_trips (más volumen)")
    df.orderBy(F.desc("num_trips")).show(25, truncate=False)

    header("DF3 | TOP por biz_score (ranking negocio)")
    if "biz_score" in df.columns:
        df.orderBy(F.desc("biz_score")).show(25, truncate=False)
    else:
        print("[INFO] No existe 'biz_score' en DF3 (¿estás leyendo la versión nueva?).")

    header("DF3 | Sanity: price_variability < 0")
    df.filter(F.col("price_variability") < 0).show(20, truncate=False)

    null_profile(df, ["pu_location_id", "hour", "service_type", "price_variability", "avg_price", "num_trips", "biz_score"], name)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    spark = get_spark()

    base = obtener_ruta("data/aggregated")
    paths = {
        "df_daily_service": f"{base}/df_daily_service",
        "df_zone_hour_day_global": f"{base}/df_zone_hour_day_global",
        "df_zone_hour_day_service": f"{base}/df_zone_hour_day_service",
        "df_variability": f"{base}/df_variability",
    }

    header("INSPECT CAPA 3 | Comprobando rutas")
    for k, p in paths.items():
        ok = safe_exists(spark, p)
        print(f"- {k}: {p} -> {'OK' if ok else 'NO EXISTE / ERROR'}")

    if safe_exists(spark, paths["df_daily_service"]):
        df1 = spark.read.parquet(paths["df_daily_service"])
        inspect_df_daily_service(df1)

    if safe_exists(spark, paths["df_zone_hour_day_global"]):
        df2a = spark.read.parquet(paths["df_zone_hour_day_global"])
        inspect_df_zone_hour_day_global(df2a)

    if safe_exists(spark, paths["df_zone_hour_day_service"]):
        df2b = spark.read.parquet(paths["df_zone_hour_day_service"])
        inspect_df_zone_hour_day_service(df2b)

    if safe_exists(spark, paths["df_variability"]):
        df3 = spark.read.parquet(paths["df_variability"])
        inspect_df_variability(df3)

    spark.stop()


if __name__ == "__main__":
    main()
