# src/procesamiento/capa2/inspect_capa2.py
from __future__ import annotations
import findspark
findspark.init()

from pyspark.sql import SparkSession, functions as F
from config.settings import obtener_ruta

LAYER2_PATH = str(obtener_ruta("data/standarized"))

# Cuánto muestrear (para que sea rápido con 45M filas)
SAMPLE_FRACTION = 0.0005  # 0.05% (~22k filas si hay 45M)
SEED = 42

# Rango "esperado"
MIN_DATE_EXPECTED = "2019-01-01"
MAX_DATE_EXPECTED = "2024-03-01"

# Umbral de outlier simple para inspección (no filtra, solo reporta)
CAP_MAX_PRICE_INSPECT = 500.0


def get_spark(app_name: str = "Inspect-Capa2"):
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .getOrCreate()
    )
    spark.conf.set("spark.sql.session.timeZone", "America/New_York")
    spark.sparkContext.setLogLevel("WARN")
    return spark


def pct(expr):
    """Convierte una agregación en porcentaje (sobre N)."""
    return (expr * 100.0)


def main():
    spark = get_spark()

    df = spark.read.parquet(LAYER2_PATH)
    print(f"\nLeyendo capa2 desde: {LAYER2_PATH}")

    # -------------------------
    # 1) Esquema + columnas
    # -------------------------
    print("\n=== SCHEMA ===")
    df.printSchema()

    print("\n=== COLUMNAS (ordenadas) ===")
    for c in sorted(df.columns):
        print("-", c)

    # -------------------------
    # 2) Conteos básicos
    # -------------------------
    print("\n=== CONTEO POR SERVICIO ===")
    if "service_type" in df.columns:
        df.groupBy("service_type").count().orderBy(F.desc("count")).show(truncate=False)
    else:
        print("[WARN] No existe 'service_type'")

    print("\n=== RANGO TEMPORAL (min/max pickup_datetime) ===")
    if "pickup_datetime" in df.columns:
        df.select(
            F.min("pickup_datetime").alias("min_pickup"),
            F.max("pickup_datetime").alias("max_pickup"),
        ).show(truncate=False)
    else:
        print("[WARN] No existe 'pickup_datetime'")

    print("\n=== RANGO TEMPORAL (min/max date) ===")
    if "date" in df.columns:
        df.select(
            F.min("date").alias("min_date"),
            F.max("date").alias("max_date"),
        ).show(truncate=False)

        print(f"\n=== FECHAS FUERA DE RANGO ESPERADO [{MIN_DATE_EXPECTED}, {MAX_DATE_EXPECTED}] ===")
        out_of_range = df.filter(
            (F.col("date") < F.lit(MIN_DATE_EXPECTED)) | (F.col("date") > F.lit(MAX_DATE_EXPECTED))
        ).count()
        total = df.count()
        print(f"Fuera de rango: {out_of_range} / {total} = {out_of_range/total*100:.6f}%")
    else:
        print("[WARN] No existe 'date'")

    # -------------------------
    # 3) Muestra para inspección humana
    # -------------------------
    sample = df.sample(withReplacement=False, fraction=SAMPLE_FRACTION, seed=SEED)

    print(f"\n=== SAMPLE HEAD (10 filas) | fraction={SAMPLE_FRACTION} ===")
    key_cols = [
        "service_type",
        "pickup_datetime", "dropoff_datetime",
        "date", "hour", "day_of_week", "is_weekend", "week_of_year",
        "pu_location_id", "pu_borough", "pu_zone",
        "do_location_id", "do_borough", "do_zone",
        "total_amount_std", "trip_duration_min",
        "year", "month",
    ]
    cols_to_show = [c for c in key_cols if c in df.columns]
    (sample.select(*cols_to_show) if cols_to_show else sample).show(10, truncate=False)

    # -------------------------
    # 4) Nulos / sanity checks (sobre la muestra)
    # -------------------------
    print("\n=== NULOS (sobre la muestra) ===")
    null_cols = [
        "pickup_datetime",
        "date",
        "hour",
        "pu_location_id",
        "do_location_id",
        "total_amount_std",
        "trip_duration_min",
        "service_type",
        "pu_borough",
        "pu_zone",
    ]
    null_checks = [F.sum(F.col(c).isNull().cast("int")).alias(f"null_{c}") for c in null_cols if c in df.columns]
    if null_checks:
        sample.select(*null_checks).show(truncate=False)
    else:
        print("[INFO] No hay columnas típicas para revisar nulos.")

    print("\n=== SANITY CHECKS (sobre la muestra) ===")
    sanity_exprs = []

    if "total_amount_std" in df.columns:
        sanity_exprs += [
            F.sum((F.col("total_amount_std") <= 0).cast("int")).alias("le_0_total_amount_std"),
            F.sum((F.col("total_amount_std") > F.lit(CAP_MAX_PRICE_INSPECT)).cast("int")).alias("gt_cap_total_amount_std"),
            F.percentile_approx("total_amount_std", 0.5).alias("median_total_amount_std"),
            F.percentile_approx("total_amount_std", 0.95).alias("p95_total_amount_std"),
            F.percentile_approx("total_amount_std", 0.99).alias("p99_total_amount_std"),
        ]

    if "trip_duration_min" in df.columns:
        sanity_exprs += [
            F.sum((F.col("trip_duration_min") < 0).cast("int")).alias("neg_trip_duration_min"),
            F.sum((F.col("trip_duration_min") > 360).cast("int")).alias("gt_360_trip_duration_min"),
            F.percentile_approx("trip_duration_min", 0.5).alias("median_trip_duration_min"),
            F.percentile_approx("trip_duration_min", 0.95).alias("p95_trip_duration_min"),
        ]

    if "hour" in df.columns:
        sanity_exprs += [
            F.sum(((F.col("hour") < 0) | (F.col("hour") > 23)).cast("int")).alias("bad_hour_outside_0_23"),
        ]

    if sanity_exprs:
        sample.select(*sanity_exprs).show(truncate=False)
    else:
        print("[INFO] No se pudieron calcular sanity checks (faltan columnas).")

    # -------------------------
    # 5) Mini-perfil por servicio (sobre muestra)
    # -------------------------
    if "service_type" in df.columns and "total_amount_std" in df.columns:
        print("\n=== PERFIL POR SERVICIO (sobre la muestra) ===")
        (
            sample.groupBy("service_type")
            .agg(
                F.count("*").alias("n"),
                F.avg("total_amount_std").alias("avg_total_amount_std"),
                F.percentile_approx("total_amount_std", 0.5).alias("median_total_amount_std"),
                F.percentile_approx("total_amount_std", 0.95).alias("p95_total_amount_std"),
                F.sum((F.col("total_amount_std") <= 0).cast("int")).alias("n_le_0"),
                F.sum((F.col("total_amount_std") > F.lit(CAP_MAX_PRICE_INSPECT)).cast("int")).alias("n_gt_cap"),
            )
            .orderBy(F.desc("n"))
            .show(truncate=False)
        )

    # -------------------------
    # 6) Conteo total (caro)
    # -------------------------
    print("\n=== CONTEO TOTAL (puede tardar) ===")
    print(df.count())

    spark.stop()


if __name__ == "__main__":
    main()
