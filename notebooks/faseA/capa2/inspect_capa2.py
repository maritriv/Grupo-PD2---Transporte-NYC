# notebooks/faseA/capa2/inspect_capa2.py
from __future__ import annotations

from pyspark.sql import SparkSession, functions as F

# ✅ Ruta nueva (tu capa2 ahora guarda en data/standarized)
LAYER2_PATH = "data/standarized"

# Cuánto muestrear (para que sea rápido con 45M filas)
SAMPLE_FRACTION = 0.0005  # 0.05% (~22k filas si hay 45M)
SEED = 42


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


def main():
    spark = get_spark()

    # Leer parquet (carpeta con particiones)
    df = spark.read.parquet(LAYER2_PATH)

    print(f"\n📦 Leyendo capa2 desde: {LAYER2_PATH}")

    # -------------------------
    # 1) Esquema + columnas
    # -------------------------
    print("\n=== SCHEMA ===")
    df.printSchema()

    print("\n=== COLUMNAS (ordenadas) ===")
    for c in sorted(df.columns):
        print("-", c)

    # -------------------------
    # 2) Conteos básicos (rápidos)
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

    # -------------------------
    # 3) Muestra para inspección humana (sin petar RAM)
    # -------------------------
    sample = df.sample(withReplacement=False, fraction=SAMPLE_FRACTION, seed=SEED)

    print(f"\n=== SAMPLE HEAD (10 filas) | fraction={SAMPLE_FRACTION} ===")
    # Mostrar solo columnas “clave” si existen, para que sea legible
    key_cols = [
        "service_type",
        "pickup_datetime", "dropoff_datetime",
        "date", "hour", "day_of_week", "is_weekend", "week_of_year",
        "pu_location_id", "do_location_id",
        "total_amount_std", "trip_duration_min",
        "year", "month",
    ]
    cols_to_show = [c for c in key_cols if c in df.columns]
    if cols_to_show:
        sample.select(*cols_to_show).show(10, truncate=False)
    else:
        sample.show(10, truncate=False)

    # -------------------------
    # 4) Nulos / sanity checks (sobre la muestra)
    # -------------------------
    print("\n=== NULOS (sobre la muestra) ===")
    null_checks = []
    for c in ["pickup_datetime", "pu_location_id", "do_location_id", "total_amount_std", "trip_duration_min"]:
        if c in df.columns:
            null_checks.append(F.sum(F.col(c).isNull().cast("int")).alias(f"null_{c}"))
    if null_checks:
        sample.select(*null_checks).show(truncate=False)
    else:
        print("[INFO] No hay columnas típicas para revisar nulos.")

    print("\n=== SANITY CHECKS (sobre la muestra) ===")
    sanity_exprs = []
    if "total_amount_std" in df.columns:
        sanity_exprs += [
            F.sum((F.col("total_amount_std") < 0).cast("int")).alias("neg_total_amount_std"),
            F.percentile_approx("total_amount_std", 0.5).alias("median_total_amount_std"),
            F.percentile_approx("total_amount_std", 0.95).alias("p95_total_amount_std"),
        ]
    if "trip_duration_min" in df.columns:
        sanity_exprs += [
            F.sum((F.col("trip_duration_min") < 0).cast("int")).alias("neg_trip_duration_min"),
            F.sum((F.col("trip_duration_min") > 360).cast("int")).alias("gt_360_trip_duration_min"),
            F.percentile_approx("trip_duration_min", 0.5).alias("median_trip_duration_min"),
            F.percentile_approx("trip_duration_min", 0.95).alias("p95_trip_duration_min"),
        ]

    if sanity_exprs:
        sample.select(*sanity_exprs).show(truncate=False)
    else:
        print("[INFO] No se pudieron calcular sanity checks (faltan columnas).")

    # -------------------------
    # 5) Conteo total (caro, pero útil)
    # -------------------------
    print("\n=== CONTEO TOTAL (puede tardar) ===")
    print(df.count())

    spark.stop()


if __name__ == "__main__":
    main()
