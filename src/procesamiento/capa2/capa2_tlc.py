# src/procesamiento/capa2/capa2_tlc_skip_existing.py
"""
CAPA 2 (TLC NYC) — Igual que tu script, pero:
- SKIP de ficheros RAW cuya partición (year, month, service_type) ya existe en data/standarized
- Escribe en modo APPEND (no sobreescribe lo ya creado)

Requisito: que los parquets RAW TLC tengan YYYY-MM en el nombre (lo normal).
Si algún parquet no lo tiene, se procesará (no se puede skipear sin leer).
"""

from __future__ import annotations

import re
from pathlib import Path
from pyspark.sql import SparkSession, functions as F

from config.settings import obtener_ruta  # type: ignore

DEBUG = False  # True -> imprime schemas y previews


# =============================================================================
# 1) Spark session
# =============================================================================
def get_spark(app_name: str = "PD2-Capa2-TLC-SkipExisting") -> SparkSession:
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


# =============================================================================
# 2) Lectura RAW robusta (evita mismatches de tipos entre parquets)
# =============================================================================
CANONICAL_CASTS = {
    "airport_fee": "double",
    "Airport_fee": "double",
    "congestion_surcharge": "double",
    "total_amount": "double",
    "fare_amount": "double",
    "tip_amount": "double",
    "tips": "double",
    "tolls_amount": "double",
    "tolls": "double",
    "trip_distance": "double",
    "trip_miles": "double",
    "PULocationID": "int",
    "DOLocationID": "int",
    "base_passenger_fare": "double",
    "sales_tax": "double",
    "bcf": "double",
    "driver_pay": "double",
}


def normalize_problem_columns(df):
    for col_name, spark_type in CANONICAL_CASTS.items():
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast(spark_type))

    if "Airport_fee" in df.columns and "airport_fee" not in df.columns:
        df = df.withColumnRenamed("Airport_fee", "airport_fee")

    if "Airport_fee" in df.columns and "airport_fee" in df.columns:
        df = df.drop("Airport_fee")

    return df


# =============================================================================
# 2.1) Helpers para SKIP por partición ya existente
# =============================================================================
_YYYY_MM = re.compile(r"(20\d{2})-(\d{2})")  # 2000-2099


def parse_year_month_from_filename(filename: str) -> tuple[int | None, int | None]:
    """
    Intenta sacar year/month del nombre del fichero (TLC típico: ..._YYYY-MM.parquet)
    """
    m = _YYYY_MM.search(filename)
    if not m:
        return None, None
    year = int(m.group(1))
    month = int(m.group(2))  # "01" -> 1
    return year, month


def partition_exists(base_out: Path, year: int, month: int, service: str) -> bool:
    part = base_out / f"year={year}" / f"month={month}" / f"service_type={service}"
    return part.exists() and any(part.iterdir())


# =============================================================================
# 2.2) Lectura RAW robusta + SKIP de los que ya están escritos
# =============================================================================
def read_raw_services(
    spark: SparkSession,
    base_path: Path = obtener_ruta("data/raw"),
    out_path: Path = obtener_ruta("data/standarized"),
    services: tuple[str, ...] = ("yellow", "green", "fhvhv"),
):
    """
    Igual que tu read_raw_services, pero:
    - Antes de leer cada parquet, mira si su partición year/month/service_type ya existe en out_path
    - Si existe -> SKIP (no lo lee, no lo une)
    """
    out = None

    for service in services:
        folder = Path(base_path) / service
        files = sorted(folder.glob("*.parquet"))

        if not files:
            print(f"[WARN] No hay parquet en {folder}. Se omite {service}.")
            continue

        for f in files:
            y, m = parse_year_month_from_filename(f.name)

            # Si podemos inferir year/month por nombre y ya existe la partición -> skip sin leer
            if y is not None and m is not None and partition_exists(Path(out_path), y, m, service):
                print(f"[SKIP] {service} | {f.name} -> ya existe year={y}/month={m}")
                continue

            # Si no podemos inferir por nombre, lo leemos (no podemos skipear sin mirar contenido)
            df = spark.read.parquet(str(f)).withColumn("service_type", F.lit(service))
            df = normalize_problem_columns(df)

            if DEBUG:
                print(f"\n--- {service.upper()} file={f.name} ---")
                df.printSchema()
                df.show(2, truncate=False)

            out = df if out is None else out.unionByName(df, allowMissingColumns=True)

    if out is None:
        raise RuntimeError(
            "No hay nada nuevo que procesar (o no se encontró ningún parquet en data/raw/*)."
        )

    if DEBUG:
        print("\n--- RAW UNION (conteo por servicio) ---")
        out.groupBy("service_type").count().show()

    return out


# =============================================================================
# 3) Construcción CAPA 2: timestamps, features temporales, precio estándar
# =============================================================================
def build_layer2(df):
    pickup_dt = F.coalesce(
        F.col("tpep_pickup_datetime"),
        F.col("lpep_pickup_datetime"),
        F.col("pickup_datetime"),
    )

    dropoff_dt = F.coalesce(
        F.col("tpep_dropoff_datetime"),
        F.col("lpep_dropoff_datetime"),
        F.col("dropoff_datetime"),
    )

    df2 = (
        df.withColumn("pickup_datetime", pickup_dt.cast("timestamp"))
          .withColumn("dropoff_datetime", dropoff_dt.cast("timestamp"))
          .withColumn("pu_location_id", F.col("PULocationID").cast("int"))
          .withColumn("do_location_id", F.col("DOLocationID").cast("int"))
    )

    airport_fee_any = F.coalesce(F.col("airport_fee"), F.lit(0.0))
    tips_any = F.coalesce(F.col("tip_amount"), F.col("tips"), F.lit(0.0))
    tolls_any = F.coalesce(F.col("tolls_amount"), F.col("tolls"), F.lit(0.0))
    congestion_any = F.coalesce(F.col("congestion_surcharge"), F.lit(0.0))

    total_amount_std = F.coalesce(
        F.col("total_amount"),
        F.coalesce(F.col("base_passenger_fare"), F.lit(0.0))
        + tips_any
        + tolls_any
        + airport_fee_any
        + congestion_any
    )

    df2 = df2.withColumn("total_amount_std", total_amount_std.cast("double"))

    df2 = (
        df2.withColumn("date", F.to_date("pickup_datetime"))
           .withColumn("year", F.year("pickup_datetime"))
           .withColumn("month", F.month("pickup_datetime"))
           .withColumn("hour", F.hour("pickup_datetime"))
           .withColumn("day_of_week", F.dayofweek("pickup_datetime"))
           .withColumn("is_weekend", F.col("day_of_week").isin([1, 7]).cast("int"))
           .withColumn("week_of_year", F.weekofyear("pickup_datetime"))
           .withColumn(
               "trip_duration_min",
               F.when(
                   F.col("pickup_datetime").isNotNull() & F.col("dropoff_datetime").isNotNull(),
                   (F.unix_timestamp("dropoff_datetime") - F.unix_timestamp("pickup_datetime")) / 60.0,
               )
           )
           .withColumn(
               "trip_duration_min",
               F.when(
                   (F.col("trip_duration_min") < 0) | (F.col("trip_duration_min") > 360),
                   None,
               ).otherwise(F.col("trip_duration_min"))
           )
    )

    if DEBUG:
        print("\n--- CAPA 2 preview ---")
        df2.select(
            "service_type", "pickup_datetime", "date", "hour",
            "pu_location_id", "do_location_id",
            "total_amount_std", "trip_duration_min"
        ).show(10, truncate=False)

    return df2


# =============================================================================
# 4) Lookup de zonas TLC (borough + zone)
# =============================================================================
def add_zone_lookup(
    spark: SparkSession,
    df,
    zone_csv_path: Path = obtener_ruta("data/external") / "taxi_zone_lookup.csv",
):
    try:
        zones = (
            spark.read.option("header", True).csv(str(zone_csv_path))
            .select(
                F.col("LocationID").cast("int").alias("location_id"),
                F.col("Borough").alias("borough"),
                F.col("Zone").alias("zone"),
            )
        )

        df = (
            df.join(F.broadcast(zones), df.pu_location_id == zones.location_id, "left")
              .drop("location_id")
              .withColumnRenamed("borough", "pu_borough")
              .withColumnRenamed("zone", "pu_zone")
        )

        df = (
            df.join(F.broadcast(zones), df.do_location_id == zones.location_id, "left")
              .drop("location_id")
              .withColumnRenamed("borough", "do_borough")
              .withColumnRenamed("zone", "do_zone")
        )

        if DEBUG:
            print("\n--- CAPA 2 + zonas preview ---")
            df.select(
                "service_type", "date", "hour", "pu_borough", "pu_zone", "do_borough", "do_zone"
            ).show(10, truncate=False)

        return df

    except Exception as e:
        print("\n[INFO] No se pudo aplicar taxi_zone_lookup (no es fatal).")
        print("Motivo:", str(e))
        return df


# =============================================================================
# 5) Selección de columnas finales
# =============================================================================
def select_layer2_columns(df):
    cols = [
        "service_type",

        "pickup_datetime",
        "dropoff_datetime",
        "date",
        "year",
        "month",
        "hour",
        "day_of_week",
        "is_weekend",
        "week_of_year",
        "trip_duration_min",

        "pu_location_id",
        "do_location_id",
        "pu_borough",
        "pu_zone",
        "do_borough",
        "do_zone",

        "total_amount_std",

        "total_amount",
        "fare_amount",
        "tip_amount",
        "tips",
        "tolls_amount",
        "tolls",
        "congestion_surcharge",
        "airport_fee",
        "base_passenger_fare",
        "trip_distance",
        "trip_miles",

        "VendorID",
        "passenger_count",
        "RatecodeID",
        "payment_type",
    ]

    existing = set(df.columns)
    keep = [c for c in cols if c in existing]

    if DEBUG:
        missing = [c for c in cols if c not in existing]
        if missing:
            print("\n[DEBUG] Columnas no presentes (esperable según servicio/año):")
            print(missing)
        print(f"\n[DEBUG] Guardando {len(keep)} columnas de {len(df.columns)} totales.")

    return df.select(*keep)


# =============================================================================
# 6) Guardado CAPA 2 (APPEND)
# =============================================================================
def save_layer2(df, out_path: Path = obtener_ruta("data/standarized")):
    """
    Cambios vs tu versión:
    - mode("append") en lugar de overwrite
    """
    (
        df.write
          .mode("append")
          .partitionBy("year", "month", "service_type")
          .parquet(str(out_path))
    )
    print("\nCapa 2 guardada (append) en:", str(out_path))


# =============================================================================
# 7) Main
# =============================================================================
def main():
    spark = get_spark()

    raw_base = obtener_ruta("data/raw")
    out_base = obtener_ruta("data/standarized")

    # 1) RAW robusto, PERO saltando los ficheros cuyo month/año/servicio ya existe en salida
    raw = read_raw_services(
        spark,
        base_path=raw_base,
        out_path=out_base,
        services=("yellow", "green", "fhvhv"),
    )

    # 2) Construye capa 2 (features + precio estándar)
    layer2 = build_layer2(raw)

    # 3) Añade zonas si existe taxi_zone_lookup.csv
    layer2 = add_zone_lookup(spark, layer2)

    # 4) Selecciona columnas finales
    layer2 = select_layer2_columns(layer2)

    # 5) Guardado incremental
    save_layer2(layer2, out_path=out_base)

    spark.stop()


if __name__ == "__main__":
    main()
