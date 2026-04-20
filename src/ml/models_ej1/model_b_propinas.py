from __future__ import annotations

"""
Entrena modelos de propinas a nivel viaje (EX1b) usando Spark ML.

Este archivo mantiene únicamente la ruta Spark: no hay carga completa en memoria
con Pandas ni respaldo a sklearn para datasets grandes.

Modelos soportados:
- dummy_median
- elastic_net
- random_forest
- hist_gradient_boosting
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from rich.table import Table
from pyspark.ml import Pipeline as SparkPipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor as SparkRandomForestRegressor
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F

from config.pipeline_runner import console, print_done, print_stage
from config.spark_manager import SparkManager
from src.ml.models_ej1.common.io import read_partitioned_parquet_dir_spark
from src.ml.models_ej2.common.split import split_model_stress_spark
from src.ml.models_ej2.common.spark import evaluate_regression_predictions

DEFAULT_INPUT_DIR = "data/aggregated/ex1b/df_trip_level_tips"
DEFAULT_OUTPUT_DIR = "outputs/ml/propinas"
RANDOM_STATE = 42
SPARK_MODEL_NAMES = ["dummy_median", "elastic_net", "random_forest", "hist_gradient_boosting"]
SUPPORTED_MODEL_NAMES = SPARK_MODEL_NAMES

CORE_REQUIRED_COLS = [
    "date",
    "hour",
    "target_tip_amount",
]

NON_FEATURE_COLS = [
    "timestamp_hour",
    "date",
    "target_tip_amount",
    "target_tip_pct",
    "has_tip",
]

STRICT_APRIORI_DROP_COLS = [
    "pickup_datetime",
    "dropoff_datetime",
    "trip_duration_min",
    "fare_amount",
    "total_amount_std",
    "target_tip_amount",
    "target_tip_pct",
    "tip_amount",
    "tip_pct",
    "has_tip",
]

SOFT_DROP_COLS = [
    "pickup_datetime",
    "dropoff_datetime",
    "target_tip_amount",
    "target_tip_pct",
    "tip_amount",
    "tip_pct",
    "has_tip",
]


def ensure_columns(df: Any, cols: list[str], name: str) -> None:
    if hasattr(df, "schema"):
        existing_cols = [field.name for field in df.schema.fields]
    else:
        existing_cols = list(df.columns)

    missing = [c for c in cols if c not in existing_cols]
    if missing:
        raise ValueError(f"[{name}] Faltan columnas requeridas: {missing}")


def _select_spark_models(model_names: list[str] | None) -> list[str]:
    if model_names is None:
        return SPARK_MODEL_NAMES

    selected: list[str] = []
    for name in model_names:
        if name not in SPARK_MODEL_NAMES:
            raise ValueError(
                f"El modelo '{name}' no es compatible con Spark. Usa: {sorted(SPARK_MODEL_NAMES)}"
            )
        selected.append(name)
    return selected


def _spark_prepare_sdf(
    sdf: SparkDataFrame,
    target_col: str,
    min_date: str | None,
    max_date: str | None,
) -> SparkDataFrame:
    ensure_columns(sdf, CORE_REQUIRED_COLS + [target_col], "EX1B")

    # Normalizar tipo de columnas y construir timestamp_hour en Spark.
    sdf = sdf.withColumn("date", F.to_date(F.col("date"), "yyyy-MM-dd"))
    sdf = sdf.withColumn("hour", F.col("hour").cast("int"))

    # Aplicar pruning de particiones si existen las columnas year/month.
    if min_date is not None and max_date is not None and "year" in sdf.columns and "month" in sdf.columns:
        min_year, min_month = map(int, min_date.split("-")[:2])
        max_year, max_month = map(int, max_date.split("-")[:2])
        ranges: list[tuple[int, int]] = []
        year = min_year
        month = min_month
        while year < max_year or (year == max_year and month <= max_month):
            ranges.append((year, month))
            month += 1
            if month > 12:
                month = 1
                year += 1

        if ranges:
            partition_filter = None
            for y, m in ranges:
                condition = (F.col("year") == y) & (F.col("month") == m)
                partition_filter = condition if partition_filter is None else partition_filter | condition
            sdf = sdf.filter(partition_filter)

    if min_date is not None:
        sdf = sdf.filter(F.col("date") >= F.lit(min_date))
    if max_date is not None:
        sdf = sdf.filter(F.col("date") <= F.lit(max_date))

    sdf = sdf.withColumn(
        "timestamp_hour",
        F.expr("make_timestamp(year(date), month(date), day(date), hour, 0, 0)").cast("timestamp"),
    )

    sdf = sdf.filter(F.col(target_col).isNotNull())
    sdf = sdf.filter(F.col(target_col) >= 0)
    sdf = sdf.filter(F.col("timestamp_hour").isNotNull())

    return sdf


def _spark_select_feature_columns(
    sdf: SparkDataFrame,
    feature_scope: str = "strict_apriori",
) -> tuple[list[str], list[str], list[str], dict[str, Any]]:
    candidate_cols = [c for c in sdf.columns if c not in NON_FEATURE_COLS]

    if feature_scope == "all":
        selected = [c for c in candidate_cols if c not in SOFT_DROP_COLS]
    elif feature_scope == "strict_apriori":
        selected = [c for c in candidate_cols if c not in STRICT_APRIORI_DROP_COLS]
    else:
        raise ValueError("feature_scope no soportado. Usa 'strict_apriori' o 'all'.")

    numeric_cols = [c for c in selected if c != "timestamp_hour" and c != "date"]
    cat_cols = [c for c in numeric_cols if str(sdf.schema[c].dataType).startswith("String")]
    num_cols = [c for c in numeric_cols if c not in cat_cols]

    return selected, num_cols, cat_cols, {
        "feature_scope": feature_scope,
        "n_selected_feature_columns": len(selected),
        "dropped_columns": [c for c in candidate_cols if c not in selected],
    }


def _spark_build_pipeline(
    num_cols: list[str],
    cat_cols: list[str],
) -> tuple[SparkPipeline, list[str]]:
    stages: list[Any] = []
    indexed_cat_cols: list[str] = []

    for cat_col in cat_cols:
        idx_col = f"{cat_col}__idx"
        stages.append(StringIndexer(inputCol=cat_col, outputCol=idx_col, handleInvalid="keep"))
        indexed_cat_cols.append(idx_col)

    assembler = VectorAssembler(
        inputCols=num_cols + indexed_cat_cols,
        outputCol="features",
        handleInvalid="skip",
    )
    stages.append(assembler)

    return SparkPipeline(stages=stages), num_cols + indexed_cat_cols


def _spark_build_regressor(model_name: str) -> Any:
    if model_name == "elastic_net":
        return LinearRegression(labelCol="label", featuresCol="features", maxIter=100, regParam=0.1, elasticNetParam=0.25)
    if model_name == "random_forest":
        return SparkRandomForestRegressor(labelCol="label", featuresCol="features", numTrees=80, maxDepth=10)
    if model_name == "hist_gradient_boosting":
        return GBTRegressor(labelCol="label", featuresCol="features", maxIter=80, maxDepth=8)
    raise ValueError(f"Modelo Spark no soportado: {model_name}")


def _spark_feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame:
    if not hasattr(model, "stages"):
        return pd.DataFrame(columns=["feature", "importance"])

    estimators = [stage for stage in model.stages if hasattr(stage, "featureImportances")]
    if not estimators:
        return pd.DataFrame(columns=["feature", "importance"])

    importance = estimators[-1].featureImportances.toArray()
    return pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)


def _spark_predict_constant(
    df: SparkDataFrame,
    value: float,
) -> SparkDataFrame:
    return df.withColumn("prediction", F.lit(value).cast("double"))


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _spark_train_single_model(
    model_name: str,
    train_df: SparkDataFrame,
    val_df: SparkDataFrame,
    test_df: SparkDataFrame,
    feature_cols: list[str],
    num_cols: list[str],
    cat_cols: list[str],
    out_base: Path,
    target_col: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "model_name": model_name,
        "target_col": target_col,
        "feature_columns": feature_cols,
    }

    if model_name == "dummy_median":
        median_value = float(train_df.approxQuantile(target_col, [0.5], 0.01)[0])
        train_pred = _spark_predict_constant(train_df, median_value)
        val_pred = _spark_predict_constant(val_df, median_value)
        test_pred = _spark_predict_constant(test_df, median_value)
        model_artifact_path = None
    else:
        pipeline, assembled_feature_cols = _spark_build_pipeline(num_cols, cat_cols)
        
        # Filtrar filas con valores NaN o nulos en las columnas numéricas
        train_df_clean = train_df.dropna(subset=num_cols)
        
        regressor = _spark_build_regressor(model_name)
        fitted_pipeline = SparkPipeline(stages=pipeline.getStages() + [regressor]).fit(train_df_clean)

        train_pred = fitted_pipeline.transform(train_df)
        val_pred = fitted_pipeline.transform(val_df)
        test_pred = fitted_pipeline.transform(test_df)

        model_artifact_path = out_base / f"{model_name}_spark_model"
        fitted_pipeline.write().overwrite().save(str(model_artifact_path))
        report["artifact_model_path"] = str(model_artifact_path)
        report["feature_importance"] = _spark_feature_importance(
            fitted_pipeline, num_cols + [f"{c}__idx" for c in cat_cols]
        ).to_dict(orient="records")

    report["metrics"] = {
        "train": evaluate_regression_predictions(train_pred, label_col="label", prediction_col="prediction"),
        "val": evaluate_regression_predictions(val_pred, label_col="label", prediction_col="prediction"),
        "test": evaluate_regression_predictions(test_pred, label_col="label", prediction_col="prediction"),
    }

    preds_path = out_base / f"{model_name}_test_predictions.parquet"
    test_pred.select(
        F.col(target_col).alias("y_true"),
        F.col("prediction").alias("y_pred"),
        F.abs(F.col(target_col) - F.col("prediction")).alias("abs_error"),
    ).write.mode("overwrite").parquet(str(preds_path))

    report["artifacts"] = {
        "test_predictions_parquet": str(preds_path),
    }
    if model_artifact_path is not None:
        report["artifacts"]["spark_model_path"] = str(model_artifact_path)

    save_json(report, out_base / f"{model_name}_report.json")
    return report


def run_training_spark(
    input_dir: str = DEFAULT_INPUT_DIR,
    out_dir: str = DEFAULT_OUTPUT_DIR,
    target_col: str = "target_tip_amount",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_state: int = RANDOM_STATE,
    save_model_flag: bool = True,
    min_date: str | None = None,
    max_date: str | None = None,
    feature_scope: str = "strict_apriori",
    model_names: list[str] | None = None,
) -> dict[str, Any]:
    print_stage("ML PROPINAS", "Regresion a nivel viaje (Spark ML)", color="cyan")

    project_root = Path(__file__).resolve().parents[3]
    out_base = (project_root / out_dir).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    spark = SparkManager.get_session()
    spark.conf.set("spark.sql.shuffle.partitions", "16")
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

    input_path = project_root / input_dir
    sdf = read_partitioned_parquet_dir_spark(input_path)
    sdf = _spark_prepare_sdf(sdf, target_col, min_date, max_date)

    total_rows = int(sdf.count())
    if total_rows == 0:
        raise ValueError("No hay filas luego de aplicar los filtros Spark. Revisa min_date/max_date.")

    feature_cols, num_cols, cat_cols, feature_meta = _spark_select_feature_columns(sdf, feature_scope)
    if not feature_cols:
        raise ValueError("No hay columnas disponibles para entrenamiento Spark tras seleccionar features.")

    train_df, val_df, test_df, bounds = split_model_stress_spark(
        sdf,
        target_col=target_col,
        time_col="timestamp_hour",
        train_frac=train_frac,
        val_frac=val_frac,
        return_bounds=True,
    )

    backend_str = "Spark"
    console.print(f"[cyan]Configuracion:[/cyan] backend={backend_str}")

    model_names = _select_spark_models(model_names)
    reports: dict[str, Any] = {}
    for model_name in model_names:
        model_out = out_base / model_name
        model_out.mkdir(parents=True, exist_ok=True)
        reports[model_name] = _spark_train_single_model(
            model_name=model_name,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_cols=feature_cols,
            num_cols=num_cols,
            cat_cols=cat_cols,
            out_base=model_out,
            target_col=target_col,
        )

    best_model_name = min(reports.keys(), key=lambda name: reports[name]["metrics"]["val"]["mae"])

    dataset_profile = {
        "input_dir": str(input_path.resolve()),
        "n_source_rows": total_rows,
        "n_train_rows": int(train_df.count()),
        "n_val_rows": int(val_df.count()),
        "n_test_rows": int(test_df.count()),
        "target_col": target_col,
        "feature_columns": feature_cols,
        "feature_selection": feature_meta,
        "split": {
            "train_frac": train_frac,
            "val_frac": val_frac,
            "test_frac": float(1.0 - train_frac - val_frac),
        },
        "bounds": bounds,
    }
    save_json(dataset_profile, out_base / "dataset_profile.json")

    final_summary = {
        "task": "predict_tip_amount",
        "problem_type": "regression",
        "target_col": target_col,
        "random_state": random_state,
        "feature_scope": feature_scope,
        "selected_models": list(reports.keys()),
        "best_model_by_val_mae": best_model_name,
        "outputs_dir": str(out_base),
        "models": reports,
    }
    save_json(final_summary, out_base / "training_summary.json")

    return final_summary


def run_training(
    input_dir: str = DEFAULT_INPUT_DIR,
    out_dir: str = DEFAULT_OUTPUT_DIR,
    target_col: str = "target_tip_amount",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_state: int = RANDOM_STATE,
    save_model_flag: bool = True,
    min_date: str | None = None,
    max_date: str | None = None,
    feature_scope: str = "strict_apriori",
    model_names: list[str] | None = None,
    use_spark: bool = True,
    n_jobs: int | None = None,
) -> dict[str, Any]:
    # Mantener compatible la firma con wrappers antiguos que pasan use_spark y n_jobs.
    return run_training_spark(
        input_dir=input_dir,
        out_dir=out_dir,
        target_col=target_col,
        train_frac=train_frac,
        val_frac=val_frac,
        random_state=random_state,
        save_model_flag=save_model_flag,
        min_date=min_date,
        max_date=max_date,
        feature_scope=feature_scope,
        model_names=model_names,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena modelos para predecir la propina de un viaje (EX1b) usando Spark."
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Directorio bajo el proyecto con los datos de entrada.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directorio de salida para metricas, artefactos y modelos.",
    )
    parser.add_argument(
        "--target-col",
        choices=["target_tip_amount", "target_tip_pct"],
        default="target_tip_amount",
        help="Target de regresion a predecir.",
    )
    parser.add_argument("--train-frac", type=float, default=0.70, help="Fraccion temporal para train.")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Fraccion temporal para val.")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Semilla aleatoria.")
    parser.add_argument(
        "--min-date",
        default=None,
        help="Filtro opcional YYYY-MM-DD sobre la fecha minima del dataset.",
    )
    parser.add_argument(
        "--max-date",
        default=None,
        help="Filtro opcional YYYY-MM-DD sobre la fecha maxima del dataset.",
    )
    parser.add_argument(
        "--feature-scope",
        choices=["strict_apriori", "all"],
        default="strict_apriori",
        help=(
            "Seleccion de features. "
            "'strict_apriori' elimina columnas con posible fuga de informacion; 'all' usa todas las disponibles."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SPARK_MODEL_NAMES,
        default=None,
        help="Modelos Spark a entrenar. Si no se indica, usa todos los soportados.",
    )
    args = parser.parse_args()

    run_training(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        target_col=args.target_col,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        random_state=args.random_state,
        save_model_flag=True,
        min_date=args.min_date,
        max_date=args.max_date,
        feature_scope=args.feature_scope,
        model_names=args.models,
    )


if __name__ == "__main__":
    main()
