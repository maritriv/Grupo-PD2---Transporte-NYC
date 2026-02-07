# notebooks/faseA/capa3/inspect_capa3.py
from __future__ import annotations

from pyspark.sql import SparkSession, functions as F

DEBUG = False


# ---------------------------------------------------------------------
# Spark (igual estilo capa2/capa3)
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
    # Comprueba si la ruta parquet existe sin petar
    try:
        spark.read.parquet(path).limit(1).count()
        return True
    except Exception:
        return False


def basic_profile(df, name: str):
    header(f"📦 {name} | SCHEMA")
    df.printSchema()

    header(f"🔢 {name} | CONTEO (rows)")
    print(df.count())

    header(f"🧾 {name} | SAMPLE (10 filas)")
    df.show(10, truncate=False)


def null_profile(df, cols: list[str], name: str, sample_fraction: float = 0.001):
    # Nulos sobre muestra (para no contar todo)
    header(f"🕳️ {name} | NULOS (sobre muestra fraction={sample_fraction})")
    sdf = df.sample(False, sample_fraction, seed=42)

    exprs = [F.sum(F.col(c).isNull().cast("int")).alias(f"null_{c}") for c in cols if c in df.columns]
    if not exprs:
        print("[INFO] No hay columnas para evaluar nulos.")
        return

    sdf.select(exprs).show(truncate=False)


def temporal_range(df, name: str):
    # Rango temporal si existe "date"
    if "date" not in df.columns:
        header(f"📅 {name} | RANGO TEMPORAL")
        print("[INFO] No existe columna 'date' en este DF.")
        return

    header(f"📅 {name} | RANGO TEMPORAL (min/max date)")
    df.agg(
        F.min("date").alias("min_date"),
        F.max("date").alias("max_date"),
    ).show(truncate=False)


def by_service_counts(df, name: str):
    if "service_type" not in df.columns:
        return
    header(f"🧩 {name} | CONTEO POR SERVICIO")
    df.groupBy("service_type").count().orderBy(F.desc("count")).show(truncate=False)


# ---------------------------------------------------------------------
# Inspecciones específicas por DF
# ---------------------------------------------------------------------
def inspect_df_daily_service(df):
    name = "DF1 df_daily_service (date + service_type)"
    basic_profile(df, name)
    temporal_range(df, name)
    by_service_counts(df, name)

    header("📈 DF1 | TOP días por num_trips (global)")
    df.orderBy(F.desc("num_trips")).show(15, truncate=False)

    header("💰 DF1 | TOP días por avg_price")
    df.orderBy(F.desc("avg_price")).show(15, truncate=False)

    header("⚠️ DF1 | TOP días por std_price (variabilidad)")
    df.orderBy(F.desc("std_price")).show(15, truncate=False)

    null_profile(df, ["date", "service_type", "num_trips", "avg_price", "std_price", "unique_zones"], name)


def inspect_df_zone_hour_day(df):
    name = "DF2 df_zone_hour_day (pu_location_id + hour + date)"
    basic_profile(df, name)
    temporal_range(df, name)

    header("🔥 DF2 | TOP combinaciones por num_trips")
    df.orderBy(F.desc("num_trips")).show(20, truncate=False)

    header("💸 DF2 | TOP combinaciones por avg_price")
    df.orderBy(F.desc("avg_price")).show(20, truncate=False)

    if "std_price" in df.columns:
        header("⚠️ DF2 | TOP combinaciones por std_price")
        df.orderBy(F.desc("std_price")).show(20, truncate=False)

    header("🕒 DF2 | Distribución por hour (conteo de filas)")
    df.groupBy("hour").count().orderBy("hour").show(50, truncate=False)

    # sanity: horas fuera de rango
    header("🧪 DF2 | Sanity: horas fuera de [0,23]")
    df.filter((F.col("hour") < 0) | (F.col("hour") > 23)).show(20, truncate=False)

    null_profile(df, ["pu_location_id", "hour", "date", "num_trips", "avg_price", "std_price"], name)


def inspect_df_variability(df):
    name = "DF3 df_variability (pu_location_id + hour + service_type)"
    basic_profile(df, name)
    by_service_counts(df, name)

    header("⚠️ DF3 | TOP por price_variability (impredecibles)")
    df.orderBy(F.desc("price_variability")).show(25, truncate=False)

    header("💰 DF3 | TOP por avg_price (caros)")
    df.orderBy(F.desc("avg_price")).show(25, truncate=False)

    header("🚦 DF3 | TOP por num_trips (más relevantes)")
    df.orderBy(F.desc("num_trips")).show(25, truncate=False)

    # sanity: variabilidad negativa (no debería pasar)
    header("🧪 DF3 | Sanity: price_variability < 0")
    df.filter(F.col("price_variability") < 0).show(20, truncate=False)

    # opcional: ranking “bueno para negocio” => variabilidad alta con volumen decente
    header("🎯 DF3 | Ranking negocio (variabilidad alta + volumen) [heurística]")
    if "price_variability" in df.columns and "num_trips" in df.columns:
        scored = df.withColumn(
            "biz_score",
            F.col("price_variability") * F.log1p(F.col("num_trips"))
        )
        scored.orderBy(F.desc("biz_score")).show(25, truncate=False)

    null_profile(df, ["pu_location_id", "hour", "service_type", "price_variability", "avg_price", "num_trips"], name)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    spark = get_spark()

    base = "data/aggregated"  # cámbialo si guardas en data/aggregated/capa3
    paths = {
        "df_daily_service": f"{base}/df_daily_service",
        "df_zone_hour_day": f"{base}/df_zone_hour_day",
        "df_variability": f"{base}/df_variability",
    }

    header("📂 INSPECT CAPA 3 | Comprobando rutas")
    for k, p in paths.items():
        ok = safe_exists(spark, p)
        print(f"- {k}: {p} -> {'OK' if ok else 'NO EXISTE / ERROR'}")

    # Lee e inspecciona si existen
    if safe_exists(spark, paths["df_daily_service"]):
        df1 = spark.read.parquet(paths["df_daily_service"])
        inspect_df_daily_service(df1)

    if safe_exists(spark, paths["df_zone_hour_day"]):
        df2 = spark.read.parquet(paths["df_zone_hour_day"])
        inspect_df_zone_hour_day(df2)

    if safe_exists(spark, paths["df_variability"]):
        df3 = spark.read.parquet(paths["df_variability"])
        inspect_df_variability(df3)

    spark.stop()


if __name__ == "__main__":
    main()
