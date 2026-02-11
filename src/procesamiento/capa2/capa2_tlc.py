# src/procesamiento/capa2/capa2_tlc.py
"""
CAPA 2 (TLC NYC) — Limpieza + estandarización + features temporales/espaciales
-----------------------------------------------------------------------------

Qué hace este script (en lenguaje “para el profe”):
1) Lee los parquets RAW de TLC (yellow, green, fhvhv) desde data/raw/<service>/*.parquet
2) Evita el típico error de Spark al leer muchos parquets con esquemas distintos:
   - Algunos meses/años cambian nombres de columnas (Airport_fee vs airport_fee)
   - Algunas columnas aparecen con todo NULL (Spark infiere NullType/void) y luego revienta
   - Solución robusta: leer fichero a fichero + castear explícitamente columnas problemáticas
3) Normaliza timestamps a un nombre común:
   - pickup_datetime / dropoff_datetime (coalesce entre nombres según servicio)
4) Crea variables “inteligentes” para análisis:
   - date, year, month, hour, day_of_week, is_weekend, week_of_year, trip_duration_min
5) Estandariza una métrica de precio:
   - total_amount_std = total_amount si existe
   - si no, para fhvhv lo aproxima sumando base_passenger_fare + tips + tolls + airport_fee + congestion
6) (Opcional pero recomendable) Enlaza zonas TLC para añadir borough/zone de pickup y dropoff:
   - Requiere data/external/taxi_zone_lookup.csv
7) Guarda CAPA 2 en data/standarized particionado por year, month, service_type

Ejecución:
    uv run -m src.procesamiento.capa2.capa2_tlc
o bien:
    python -m src.procesamiento.capa2.capa2_tlc

Requisito para zonas:
    wget -P data/external/ https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv
"""

from __future__ import annotations

from pathlib import Path
from pyspark.sql import SparkSession, functions as F

# Si usas tu config.settings, mantenlo.
# Si NO lo tienes, comenta esta importación y usa paths directos (Path("data/raw"), etc.)
from config.settings import obtener_ruta  # type: ignore

DEBUG = False  # True -> imprime schemas y previews


# =============================================================================
# 1) Spark session
# =============================================================================
def get_spark(app_name: str = "PD2-Capa2-TLC") -> SparkSession:
    """
    Crea la sesión de Spark con una configuración razonable para local/WSL.

    Nota:
    - Aumentamos memoria porque TLC pesa.
    - Ajustamos timezone a America/New_York para que los features temporales sean correctos.
    """
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
# Columnas que suelen dar guerra por cambios entre años/meses o por venir todo NULL
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
    """
    Normaliza columnas problemáticas para que el union de parquets NO reviente.

    - Castea explícitamente columnas conocidas (evita NullType/void vs double).
    - Unifica Airport_fee vs airport_fee (se queda con airport_fee).
    """
    # 1) Cast explícito si existe la columna
    for col_name, spark_type in CANONICAL_CASTS.items():
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast(spark_type))

    # 2) Unificación Airport_fee -> airport_fee
    if "Airport_fee" in df.columns and "airport_fee" not in df.columns:
        df = df.withColumnRenamed("Airport_fee", "airport_fee")

    # Si por cualquier motivo están las dos, prioriza airport_fee
    if "Airport_fee" in df.columns and "airport_fee" in df.columns:
        df = df.drop("Airport_fee")

    return df


def read_raw_services(
    spark: SparkSession,
    base_path: Path = obtener_ruta("data/raw"),
    services: tuple[str, ...] = ("yellow", "green", "fhvhv"),
):
    """
    Lee RAW de cada servicio (yellow/green/fhvhv) de forma robusta:
    - Lee fichero a fichero
    - Normaliza tipos antes de hacer unionByName

    Esto evita el error:
    FAILED_READ_FILE.PARQUET_COLUMN_DATA_TYPE_MISMATCH
    """
    out = None

    for service in services:
        folder = Path(base_path) / service
        files = sorted(folder.glob("*.parquet"))

        if not files:
            print(f"[WARN] No hay parquet en {folder}. Se omite {service}.")
            continue

        for f in files:
            df = spark.read.parquet(str(f)).withColumn("service_type", F.lit(service))
            df = normalize_problem_columns(df)

            if DEBUG:
                print(f"\n--- {service.upper()} file={f.name} ---")
                df.printSchema()
                df.show(2, truncate=False)

            out = df if out is None else out.unionByName(df, allowMissingColumns=True)

    if out is None:
        raise RuntimeError("No se encontró ningún parquet en data/raw/*")

    if DEBUG:
        print("\n--- RAW UNION (conteo por servicio) ---")
        out.groupBy("service_type").count().show()

    return out


# =============================================================================
# 3) Construcción CAPA 2: timestamps, features temporales, precio estándar
# =============================================================================
def build_layer2(df):
    """
    Capa 2 = Capa RAW + columnas “inteligentes” pero SIN perder granularidad (1 fila = 1 viaje).

    Aquí:
    - Normalizamos timestamps (cada servicio usa nombres distintos)
    - Generamos features temporales (date, hour, etc.)
    - Creamos total_amount_std (comparabilidad básica entre servicios)
    """
    # --- timestamps unificados ---
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

    # --- precio estandarizado ---
    # Para taxi (yellow/green) normalmente existe total_amount.
    # Para fhvhv a veces no existe total_amount, pero sí base_passenger_fare + varios extras.
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

    # --- features temporales + duración ---
    df2 = (
        df2.withColumn("date", F.to_date("pickup_datetime"))
           .withColumn("year", F.year("pickup_datetime"))
           .withColumn("month", F.month("pickup_datetime"))
           .withColumn("hour", F.hour("pickup_datetime"))
           .withColumn("day_of_week", F.dayofweek("pickup_datetime"))  # 1=Dom ... 7=Sáb
           .withColumn("is_weekend", F.col("day_of_week").isin([1, 7]).cast("int"))
           .withColumn("week_of_year", F.weekofyear("pickup_datetime"))
           .withColumn(
               "trip_duration_min",
               F.when(
                   F.col("pickup_datetime").isNotNull() & F.col("dropoff_datetime").isNotNull(),
                   (F.unix_timestamp("dropoff_datetime") - F.unix_timestamp("pickup_datetime")) / 60.0,
               )
           )
           # Higiene básica duración: quita negativos o viajes absurdamente largos
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
    """
    Añade borough/zone para pickup y dropoff usando taxi_zone_lookup.csv.

    - Es un join pequeño, por eso lo hacemos con broadcast (más rápido).
    - Si el CSV no existe, el script NO revienta: solo devuelve df sin zonas.
    """
    try:
        zones = (
            spark.read.option("header", True).csv(str(zone_csv_path))
            .select(
                F.col("LocationID").cast("int").alias("location_id"),
                F.col("Borough").alias("borough"),
                F.col("Zone").alias("zone"),
            )
        )

        # Pickup
        df = (
            df.join(F.broadcast(zones), df.pu_location_id == zones.location_id, "left")
              .drop("location_id")
              .withColumnRenamed("borough", "pu_borough")
              .withColumnRenamed("zone", "pu_zone")
        )

        # Dropoff
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
# 5) Selección de columnas finales (schema “canónico” + opcionales si existen)
# =============================================================================
def select_layer2_columns(df):
    """
    Selecciona un conjunto de columnas final:
    - columnas canónicas que usaremos sí o sí
    - columnas opcionales que pueden existir dependiendo del servicio/año
    """
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

        # (opcionales)
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
# 6) Guardado CAPA 2
# =============================================================================
def save_layer2(df, out_path: Path = obtener_ruta("data/standarized")):
    """
    Guarda CAPA 2 particionada.

    Importante:
    - PartitionBy(year, month, service_type) crea carpetas tipo:
        year=2023/month=4/service_type=yellow/part-...
    - Si ves años raros (ej: 2028), normalmente es porque hay timestamps corruptos.
      Se soluciona filtrando por rangos de fecha ANTES de escribir (si lo necesitas).
    """
    (
        df.write
          .mode("overwrite")
          .partitionBy("year", "month", "service_type")
          .parquet(str(out_path))
    )
    print("\nCapa 2 guardada en:", str(out_path))


# =============================================================================
# 7) Main
# =============================================================================
def main():
    spark = get_spark()

    # 1) RAW robusto (lee fichero a fichero, castea y une)
    raw = read_raw_services(spark, base_path=obtener_ruta("data/raw"))

    # 2) Construye capa 2 (features + precio estándar)
    layer2 = build_layer2(raw)

    # 3) Añade zonas si existe taxi_zone_lookup.csv
    layer2 = add_zone_lookup(spark, layer2)

    # 4) Selecciona columnas finales
    layer2 = select_layer2_columns(layer2)

    # 5) Guardado
    save_layer2(layer2, out_path=obtener_ruta("data/standarized"))

    spark.stop()


if __name__ == "__main__":
    main()
