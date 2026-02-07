# notebooks/faseA/capa3/capa3.py
from __future__ import annotations

from pyspark.sql import SparkSession, functions as F

DEBUG = False  # True para ver previews


# ---------------------------------------------------------------------
# Spark (igual que tu capa2)
# ---------------------------------------------------------------------
def get_spark(app_name: str = "PD2-Capa3"):
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
# Lectura capa 2 (tu output: data/standarized) + higiene mínima
# ---------------------------------------------------------------------
def read_layer2(
    spark,
    layer2_path: str = "data/standarized",
    min_date: str = "2019-01-01",
    max_date: str = "2024-12-31",
):
    df = spark.read.parquet(layer2_path)

    # defensivo: quita registros rotos para agregación
    df = df.filter(
        F.col("date").isNotNull()
        & F.col("hour").isNotNull()
        & F.col("service_type").isNotNull()
        & F.col("pu_location_id").isNotNull()
        & F.col("total_amount_std").isNotNull()
    )

    # ✅ filtro de fechas razonables (evita cosas tipo 2002-12-31)
    df = df.filter(F.col("date").between(F.lit(min_date), F.lit(max_date)))

    return df


# ---------------------------------------------------------------------
# Construcción CAPA 3 (CLEAN)
# ---------------------------------------------------------------------
def build_layer3(
    df_capa2,
    min_trips_df2: int = 30,   # zona+hora+dia: mínimo para stats estables
    min_trips_df3: int = 100,  # zona+hora+servicio: mínimo para variabilidad fiable
):
    # 📊 DF1: Evolución diaria por servicio (date + service_type)
    df_daily_service = (
        df_capa2
        .groupBy("date", "service_type")
        .agg(
            F.count("*").alias("num_trips"),
            F.avg("total_amount_std").alias("avg_price"),
            F.stddev("total_amount_std").alias("std_price"),
            F.countDistinct("pu_location_id").alias("unique_zones"),
        )
    )

    # 📍 DF2: Hotspots (zona + hora + día)
    df_zone_hour_day = (
        df_capa2
        .groupBy("pu_location_id", "hour", "date")
        .agg(
            F.count("*").alias("num_trips"),
            F.avg("total_amount_std").alias("avg_price"),
            F.stddev("total_amount_std").alias("std_price"),
        )
        # ✅ quita combinaciones con pocos viajes (evita tops absurdos)
        .filter(F.col("num_trips") >= F.lit(min_trips_df2))
    )

    # 🔥 DF3: Variabilidad (zona + hora + servicio)
    df_variability = (
        df_capa2
        .groupBy("pu_location_id", "hour", "service_type")
        .agg(
            F.stddev("total_amount_std").alias("price_variability"),
            F.avg("total_amount_std").alias("avg_price"),
            F.count("*").alias("num_trips"),
        )
        # ✅ quita combinaciones con pocos viajes (variabilidad poco fiable)
        .filter(F.col("num_trips") >= F.lit(min_trips_df3))
    )

    return df_daily_service, df_zone_hour_day, df_variability


# ---------------------------------------------------------------------
# Guardado capa 3
# ---------------------------------------------------------------------
def save_layer3(
    df_daily_service,
    df_zone_hour_day,
    df_variability,
    out_base: str = "data/aggregated",
):
    (
        df_daily_service
        .write
        .mode("overwrite")
        .partitionBy("service_type")
        .parquet(f"{out_base}/df_daily_service")
    )

    (
        df_zone_hour_day
        .write
        .mode("overwrite")
        .partitionBy("date")
        .parquet(f"{out_base}/df_zone_hour_day")
    )

    (
        df_variability
        .write
        .mode("overwrite")
        .partitionBy("service_type")
        .parquet(f"{out_base}/df_variability")
    )

    print("\n✅ Capa 3 (CLEAN) guardada en:", out_base)
    print(" - df_daily_service  ->", f"{out_base}/df_daily_service")
    print(" - df_zone_hour_day  ->", f"{out_base}/df_zone_hour_day")
    print(" - df_variability    ->", f"{out_base}/df_variability")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    spark = get_spark()

    # ajusta max_date si quieres clavar tu histórico real
    df_capa2 = read_layer2(
        spark,
        layer2_path="data/standarized",
        min_date="2019-01-01",
        max_date="2024-03-01",  # por tu output actual, acaba en 2024-03-01
    )

    df1, df2, df3 = build_layer3(
        df_capa2,
        min_trips_df2=30,
        min_trips_df3=100,
    )

    if DEBUG:
        print("\n--- DF1 daily_service sample ---")
        df1.orderBy(F.desc("date")).show(10, truncate=False)

        print("\n--- DF2 zone_hour_day sample ---")
        df2.orderBy(F.desc("date"), F.desc("num_trips")).show(10, truncate=False)

        print("\n--- DF3 variability sample ---")
        df3.orderBy(F.desc("price_variability")).show(10, truncate=False)

    save_layer3(df1, df2, df3, out_base="data/aggregated")
    spark.stop()


if __name__ == "__main__":
    main()
