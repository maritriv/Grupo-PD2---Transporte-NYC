# src/procesamiento/capa2/capa2.py
from __future__ import annotations
import findspark
findspark.init()

from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from config.settings import obtener_ruta

DEBUG = False  # pon True para ver schemas/previews


# ---------------------------------------------------------------------
# Spark (WSL / Linux friendly)
# ---------------------------------------------------------------------
def get_spark(app_name: str = "PD2-Capa2"):
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
# Lectura capa RAW (yellow / green / fhvhv)
# ---------------------------------------------------------------------
def read_raw_services(
    spark,
    base_path: Path = obtener_ruta("data/raw"),
    services: tuple[str, ...] = ("yellow", "green", "fhvhv"),
):
    dfs = []

    for service in services:
        folder = Path(base_path) / service
        files = sorted(folder.glob("*.parquet"))

        if not files:
            print(f"[WARN] No hay parquet en {folder}. Se omite {service}.")
            continue

        df = spark.read.parquet(*map(str, files)).withColumn("service_type", F.lit(service))

        if DEBUG:
            print(f"\n--- {service.upper()} ({len(files)} archivos) ---")
            df.printSchema()
            df.show(3, truncate=False)

        dfs.append(df)

    if not dfs:
        raise RuntimeError("No se encontró ningún parquet en data/raw/*")

    out = dfs[0]
    for d in dfs[1:]:
        out = out.unionByName(d, allowMissingColumns=True)

    if DEBUG:
        print("\n--- RAW UNION (conteo por servicio) ---")
        out.groupBy("service_type").count().show()

    return out


# ---------------------------------------------------------------------
# Construcción CAPA 2
# ---------------------------------------------------------------------
def build_layer2(df):
    # Normalizar timestamps (cada servicio usa nombres distintos)
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

    # ---- precio estandarizado (robusto a nombres distintos) ----
    airport_fee_any = F.coalesce(F.col("airport_fee"), F.col("Airport_fee"), F.lit(0.0))
    tips_any = F.coalesce(F.col("tip_amount"), F.col("tips"), F.lit(0.0))
    tolls_any = F.coalesce(F.col("tolls_amount"), F.col("tolls"), F.lit(0.0))

    total_amount_std = F.coalesce(
        F.col("total_amount"),
        F.coalesce(F.col("base_passenger_fare"), F.lit(0.0))
        + tips_any
        + tolls_any
        + airport_fee_any
        + F.coalesce(F.col("congestion_surcharge"), F.lit(0.0))
    )

    df2 = df2.withColumn("total_amount_std", total_amount_std.cast("double"))

    # Variables temporales + duración
    df2 = (
        df2.withColumn("date", F.to_date("pickup_datetime"))
           .withColumn("year", F.year("pickup_datetime"))
           .withColumn("month", F.month("pickup_datetime"))
           .withColumn("hour", F.hour("pickup_datetime"))
           .withColumn("day_of_week", F.dayofweek("pickup_datetime"))
           .withColumn("is_weekend", F.col("day_of_week").isin([6, 7]).cast("int"))
           .withColumn("week_of_year", F.weekofyear("pickup_datetime"))
           .withColumn(
               "trip_duration_min",
               F.when(
                   (F.col("pickup_datetime").isNotNull()) & (F.col("dropoff_datetime").isNotNull()),
                   (F.unix_timestamp("dropoff_datetime") - F.unix_timestamp("pickup_datetime")) / 60.0,
               ),
           )
           .withColumn(
               "trip_duration_min",
               F.when(
                   (F.col("trip_duration_min") < 0) | (F.col("trip_duration_min") > 360),
                   None,
               ).otherwise(F.col("trip_duration_min")),
           )
    )

    if DEBUG:
        print("\n--- CAPA 2 preview ---")
        df2.select(
            "service_type",
            "pickup_datetime",
            "date",
            "hour",
            "pu_location_id",
            "do_location_id",
            "total_amount_std",
            "trip_duration_min",
        ).show(10, truncate=False)

    return df2


# ---------------------------------------------------------------------
# Lookup zonas TLC (borough + zone)
# ---------------------------------------------------------------------
def add_zone_lookup(
    spark,
    df,
    zone_csv_path: Path = obtener_ruta("data/external") / "taxi_zone_lookup.csv",
):
    try:
        zones = (
            spark.read.option("header", True).csv(zone_csv_path)
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
                "service_type",
                "date",
                "hour",
                "pu_borough",
                "pu_zone",
                "do_borough",
                "do_zone",
            ).show(10, truncate=False)

        return df

    except Exception as e:
        print("\n[INFO] No se pudo aplicar taxi_zone_lookup.")
        print("Motivo:", str(e))
        return df


# ---------------------------------------------------------------------
# SELECT final (schema canónico)
# ---------------------------------------------------------------------
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

        # opcionales (si existen)
        "total_amount",
        "fare_amount",
        "extra",
        "mta_tax",
        "tip_amount",
        "tips",
        "tolls_amount",
        "tolls",
        "improvement_surcharge",
        "congestion_surcharge",
        "Airport_fee",
        "airport_fee",
        "sales_tax",
        "bcf",
        "driver_pay",
        "base_passenger_fare",

        "trip_distance",
        "trip_miles",
        "trip_time",

        "VendorID",
        "passenger_count",
        "RatecodeID",
        "payment_type",
        "store_and_fwd_flag",
        "trip_type",
        "ehail_fee",

        "hvfhs_license_num",
        "dispatching_base_num",
        "originating_base_num",
        "request_datetime",
        "on_scene_datetime",
        "shared_request_flag",
        "shared_match_flag",
        "access_a_ride_flag",
        "wav_request_flag",
        "wav_match_flag",
    ]

    existing = set(df.columns)
    keep = [c for c in cols if c in existing]

    if DEBUG:
        missing = [c for c in cols if c not in existing]
        if missing:
            print("\n[DEBUG] Columnas no presentes (ok según servicio/version):")
            print(missing)
        print(f"\n[DEBUG] Guardando {len(keep)} columnas de {len(df.columns)} totales.")

    return df.select(*keep)


# ---------------------------------------------------------------------
# Guardado
# ---------------------------------------------------------------------
def save_layer2(df, out_path: Path = obtener_ruta("data/standarized")):
    (
        df.write
          .mode("overwrite")
          .partitionBy("year", "month", "service_type")
          .parquet(out_path)
    )
    print("\n✅ Capa 2 guardada en:", out_path)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    spark = get_spark()

    raw = read_raw_services(spark)
    layer2 = build_layer2(raw)
    layer2 = add_zone_lookup(spark, layer2)
    layer2 = select_layer2_columns(layer2)

    save_layer2(layer2)
    spark.stop()


if __name__ == "__main__":
    main()
