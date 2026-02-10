# src/procesamiento/capa2/capa2_eventos.py
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
def get_spark(app_name: str = "PD2-Capa2-Eventos"):
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
# Lectura RAW eventos (csv o parquet)
# ---------------------------------------------------------------------
def read_raw_events(
    spark,
    in_dir: Path = obtener_ruta("data/external/events"),
    base_name: str = "events_daily_borough_type",
):
    parquet_path = in_dir / f"{base_name}.parquet"
    csv_path = in_dir / f"{base_name}.csv"

    if parquet_path.exists():
        df = spark.read.parquet(str(parquet_path))
        # 🔥 si es un parquet viejo o mal generado (sin hour), usa CSV
        if "hour" not in df.columns and csv_path.exists():
            print("[WARN] Parquet RAW de eventos no tiene 'hour'. Leyendo CSV en su lugar.")
            df = spark.read.option("header", True).csv(str(csv_path))
        return df

    if csv_path.exists():
        return spark.read.option("header", True).csv(str(csv_path))

    raise FileNotFoundError(f"No encuentro {base_name}.parquet ni {base_name}.csv en {in_dir}")



# ---------------------------------------------------------------------
# Construcción CAPA 2 eventos (tipado + variables temporales)
# ---------------------------------------------------------------------
def build_layer2_events(df):
    # Normaliza columnas esperadas (por si vienen con otros nombres)
    # date: "YYYY-MM-DD" (string) -> date
    # hour: string/int -> int
    df2 = (
        df
        .withColumn("date", F.to_date(F.col("date")))
        .withColumn("hour", F.col("hour").cast("int"))
        .withColumn("borough", F.trim(F.col("borough")))
        .withColumn("event_type", F.trim(F.col("event_type")))
        .withColumn("n_events", F.col("n_events").cast("int"))
    )

    # Higiene mínima
    df2 = df2.filter(
        F.col("date").isNotNull()
        & F.col("hour").isNotNull()
        & (F.col("hour") >= 0) & (F.col("hour") <= 23)
        & F.col("borough").isNotNull()
        & F.col("event_type").isNotNull()
        & F.col("n_events").isNotNull()
        & (F.col("n_events") >= 0)
    )

    # Normalización estética de borough (opcional)
    df2 = df2.withColumn("borough", F.initcap(F.col("borough")))

    # Variables temporales (análogo a taxis)
    df2 = (
        df2
        .withColumn("year", F.year("date"))
        .withColumn("month", F.month("date"))
        .withColumn("day_of_week", F.dayofweek("date"))  # 1=Dom ... 7=Sáb
        .withColumn("is_weekend", F.col("day_of_week").isin([1, 7]).cast("int"))
        .withColumn("week_of_year", F.weekofyear("date"))
    )

    if DEBUG:
        print("\n--- CAPA 2 EVENTOS preview ---")
        df2.orderBy(F.desc("date"), F.asc("hour")).show(20, truncate=False)

    return df2


# ---------------------------------------------------------------------
# SELECT final (schema canónico eventos)
# ---------------------------------------------------------------------
def select_layer2_events_columns(df):
    cols = [
        "date",
        "hour",
        "year",
        "month",
        "day_of_week",
        "is_weekend",
        "week_of_year",
        "borough",
        "event_type",
        "n_events",
    ]
    existing = set(df.columns)
    keep = [c for c in cols if c in existing]
    return df.select(*keep)


# ---------------------------------------------------------------------
# Guardado
# ---------------------------------------------------------------------
def save_layer2_events(df, out_path: Path = obtener_ruta("data/standarized") / "events"):
    (
        df.write
        .mode("overwrite")
        .partitionBy("year", "month", "borough")
        .parquet(str(out_path))
    )
    print("\nCapa 2 EVENTOS guardada en:", out_path)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print(">>> entrando en main capa2_eventos")
    spark = get_spark()

    raw = read_raw_events(spark)
    layer2 = build_layer2_events(raw)
    layer2 = select_layer2_events_columns(layer2)

    save_layer2_events(layer2)
    spark.stop()


if __name__ == "__main__":
    main()
