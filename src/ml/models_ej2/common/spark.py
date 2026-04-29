from __future__ import annotations

from pathlib import Path
from typing import Sequence

from pyspark import StorageLevel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from config.spark_manager import SparkManager


def get_spark_session(
    *,
    shuffle_partitions: int,
    enable_arrow: bool = True,
) -> SparkSession:
    spark = SparkManager.get_session()
    spark.conf.set("spark.sql.shuffle.partitions", str(shuffle_partitions))
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", str(enable_arrow).lower())
    return spark


def read_parquet(spark: SparkSession, input_dir: str | Path) -> DataFrame:
    return spark.read.parquet(str(input_dir))


def apply_time_target_not_null_filters(
    df: DataFrame,
    *,
    time_col: str,
    target_col: str,
) -> DataFrame:
    out = df.withColumn(time_col, F.to_timestamp(F.col(time_col)))
    out = out.filter(F.col(time_col).isNotNull())
    out = out.filter(F.col(target_col).isNotNull())
    return out


def infer_feature_columns(
    df: DataFrame,
    *,
    label_col: str,
) -> tuple[list[str], list[str]]:
    # Blindaje anti-leakage y anti-columnas problemáticas.
    excluded_cols = {
        label_col,
        "date",
        "datetime",
        "timestamp",
        "timestamp_hour",
        "pickup_datetime",
        "dropoff_datetime",
    }

    cols = [
        c
        for c in df.columns
        if c not in excluded_cols
        and not c.startswith("target_")
    ]

    cat_cols = [
        c
        for c, t in df.dtypes
        if c in cols and t in {"string", "boolean"}
    ]

    numeric_types = {
        "int",
        "bigint",
        "double",
        "float",
        "long",
        "short",
        "decimal",
        "tinyint",
    }

    num_cols = [
        c
        for c, t in df.dtypes
        if c in cols and any(t.startswith(nt) for nt in numeric_types)
    ]

    return num_cols, cat_cols

def materialize_on_disk(df: DataFrame) -> DataFrame:
    cached = df.persist(StorageLevel.DISK_ONLY)
    _ = cached.count()
    return cached


def evaluate_regression_predictions(
    df: DataFrame,
    *,
    label_col: str = "label",
    prediction_col: str = "prediction",
    metrics: Sequence[str] = ("mae", "rmse", "r2"),
) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric_name in metrics:
        evaluator = RegressionEvaluator(
            labelCol=label_col,
            predictionCol=prediction_col,
            metricName=metric_name,
        )
        out[metric_name] = float(evaluator.evaluate(df))
    return out
