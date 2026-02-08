# notebooks/faseB/viz_common.py
from __future__ import annotations

from pyspark.sql import SparkSession, functions as F


def get_spark(app_name: str = "PD2-FaseB-Viz"):
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


def read_capa3(spark: SparkSession, base: str = "data/aggregated"):
    df1 = spark.read.parquet(f"{base}/df_daily_service")
    df2a = spark.read.parquet(f"{base}/df_zone_hour_day_global")
    df2b = spark.read.parquet(f"{base}/df_zone_hour_day_service")
    df3 = spark.read.parquet(f"{base}/df_variability")
    return df1, df2a, df2b, df3


def ensure_local_date(df, col="date"):
    # para que pandas/matplotlib lo trate bien
    return df.withColumn(col, F.to_timestamp(F.col(col)))


def save_fig(fig, out_path: str):
    import os
    os.makedirs("outputs/faseB", exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print("✅ Guardado:", out_path)
