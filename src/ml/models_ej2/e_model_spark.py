from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import SparkSession
from rich.table import Table
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from config.pipeline_runner import console, print_done, print_stage

TARGET_REG = "stress_score"
TARGET_CLF = "is_stress"
VALID_MODES = {"operational", "predictive"}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def rmse(y_true, y_pred) -> float:
    """Calcula el RMSE entre y_true y y_pred."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_processed_splits(
    splits_dir: str = "data/ml/splits_processed",
    prefix: str = "completo",
    mode: str = "operational",
) -> dict[str, pd.DataFrame]:
    """Carga los splits procesados en pandas."""
    if mode not in VALID_MODES:
        raise ValueError(f"Modo no soportado: {mode}. Usa uno de {sorted(VALID_MODES)}")

    project_root = Path(__file__).resolve().parents[3]
    base = (project_root / splits_dir).resolve()

    train_fp = base / f"{prefix}_{mode}_train.parquet"
    val_fp = base / f"{prefix}_{mode}_val.parquet"
    test_fp = base / f"{prefix}_{mode}_test.parquet"

    for fp in [train_fp, val_fp, test_fp]:
        if not fp.exists():
            raise FileNotFoundError(f"No existe el split procesado esperado: {fp}")

    return {
        "train": pd.read_parquet(train_fp),
        "val": pd.read_parquet(val_fp),
        "test": pd.read_parquet(test_fp),
    }


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features de target de regresión."""
    if TARGET_REG not in df.columns:
        raise ValueError(f"Falta target '{TARGET_REG}' en el split.")

    x = df.drop(columns=[TARGET_REG, TARGET_CLF]).copy()
    y_reg = df[TARGET_REG].copy()
    return x, y_reg


def evaluate_regression_pandas(y_true, y_pred) -> dict[str, float]:
    """Evalúa métricas de regresión con sklearn."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_json(data: dict[str, Any], out_path: Path) -> None:
    """Guarda un diccionario como JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_categorical_features(df: pd.DataFrame) -> list[str]:
    """Identifica columnas categóricas (object dtype)."""
    return [col for col in df.columns if df[col].dtype == "object"]


def get_numeric_features(df: pd.DataFrame) -> list[str]:
    """Identifica columnas numéricas."""
    return [col for col in df.columns if df[col].dtype in ["int64", "float64"]]


# -----------------------------------------------------------------------------
# Spark GBT Regressor
# -----------------------------------------------------------------------------
def run_spark_gbt_regressor(
    splits_dir: str = "data/ml/splits_processed",
    prefix: str = "completo",
    mode: str = "operational",
    outputs_dir: str = "outputs/ml",
) -> dict[str, Any]:
    """
    Entrena un GBTRegressor con Spark y lo compara con XGBoost.
    
    Pipeline Spark:
    - StringIndexer (variables categóricas)
    - OneHotEncoder (convertir índices a vectores)
    - VectorAssembler (juntar características)
    - GBTRegressor (modelo de regresión)
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Modo no soportado: {mode}. Usa uno de {sorted(VALID_MODES)}")

    print_stage(
        "ML SPARK GBT",
        f"GBTRegressor con pipeline Spark | mode={mode}",
        color="cyan"
    )

    project_root = Path(__file__).resolve().parents[3]
    outputs_base = (project_root / outputs_dir).resolve()
    outputs_base.mkdir(parents=True, exist_ok=True)

    # Cargar splits en pandas
    splits = load_processed_splits(
        splits_dir=splits_dir,
        prefix=prefix,
        mode=mode,
    )

    train_df_pd = splits["train"]
    val_df_pd = splits["val"]
    test_df_pd = splits["test"]

    x_train_pd, y_train_reg = split_xy(train_df_pd)
    x_val_pd, y_val_reg = split_xy(val_df_pd)
    x_test_pd, y_test_reg = split_xy(test_df_pd)

    train_size = len(train_df_pd)
    val_size = len(val_df_pd)
    test_size = len(test_df_pd)
    total_size = train_size + val_size + test_size
    n_features = x_train_pd.shape[1]

    console.print(
        f"[cyan]Splits cargados[/cyan] -> "
        f"mode={mode} | train={train_size:,} | val={val_size:,} | test={test_size:,} | total={total_size:,}"
    )
    console.print(f"[cyan]Features[/cyan] -> {n_features:,}")

    # =========================================================================
    # SPARK GBTRegressor
    # =========================================================================
    console.print("\n[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]SPARK GBTRegressor[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]")

    spark = SparkSession.builder.appName("GBTRegressor").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Convertir a Spark DataFrames
    train_df_spark = spark.createDataFrame(
        train_df_pd.reset_index(drop=True)
    )
    val_df_spark = spark.createDataFrame(
        val_df_pd.reset_index(drop=True)
    )
    test_df_spark = spark.createDataFrame(
        test_df_pd.reset_index(drop=True)
    )

    # Identificar features categóricas y numéricas
    categorical_features = get_categorical_features(x_train_pd)
    numeric_features = get_numeric_features(x_train_pd)

    console.print(f"[cyan]Categorical features[/cyan] -> {len(categorical_features)}")
    console.print(f"[cyan]Numeric features[/cyan] -> {len(numeric_features)}")

    # Construir pipeline Spark
    stages = []

    # 1. StringIndexer para variables categóricas
    if categorical_features:
        for cat_col in categorical_features:
            indexer = StringIndexer(
                inputCol=cat_col,
                outputCol=f"{cat_col}_indexed",
                handleInvalid="skip"
            )
            stages.append(indexer)

    # 2. OneHotEncoder para variables categóricas indexadas
    indexed_cols = [f"{cat}_indexed" for cat in categorical_features]
    if indexed_cols:
        encoder = OneHotEncoder(
            inputCols=indexed_cols,
            outputCols=[f"{col}_encoded" for col in indexed_cols],
            handleInvalid="skip"
        )
        stages.append(encoder)

    # 3. VectorAssembler para juntar características
    assembler_inputs = (
        numeric_features + [f"{col}_encoded" for col in indexed_cols]
    )
    assembler = VectorAssembler(
        inputCols=assembler_inputs,
        outputCol="features"
    )
    stages.append(assembler)

    # 4. GBTRegressor
    gbt = GBTRegressor(
        featuresCol="features",
        labelCol=TARGET_REG,
        maxDepth=6,
        maxIter=100,
        stepSize=0.1,
        subsamplingRate=0.8,
        seed=42,
    )
    stages.append(gbt)

    # Crear pipeline
    pipeline = Pipeline(stages=stages)

    # Entrenar modelo
    console.print("[yellow]Entrenando Spark GBTRegressor...[/yellow]")
    start_time_spark = time.time()
    gbt_model = pipeline.fit(train_df_spark)
    spark_train_time = time.time() - start_time_spark

    console.print(
        f"[green]✓ Tiempo de entrenamiento (Spark GBT)[/green]: "
        f"{spark_train_time:.4f} segundos"
    )

    # Predicciones en val y test
    val_pred_spark = gbt_model.transform(val_df_spark).select("prediction").toPandas()["prediction"].values
    test_pred_spark = gbt_model.transform(test_df_spark).select("prediction").toPandas()["prediction"].values

    spark_val_metrics = evaluate_regression_pandas(y_val_reg, val_pred_spark)
    spark_test_metrics = evaluate_regression_pandas(y_test_reg, test_pred_spark)

    spark.stop()

    # =========================================================================
    # XGBoost (para comparación)
    # =========================================================================
    console.print("\n[bold yellow]═══════════════════════════════════════[/bold yellow]")
    console.print("[bold yellow]XGBoost (Comparación)[/bold yellow]")
    console.print("[bold yellow]═══════════════════════════════════════[/bold yellow]")

    console.print("[yellow]Entrenando XGBoost Regressor...[/yellow]")
    start_time_xgb = time.time()

    xgb_model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    xgb_model.fit(x_train_pd, y_train_reg)
    xgb_train_time = time.time() - start_time_xgb

    console.print(
        f"[green]✓ Tiempo de entrenamiento (XGBoost)[/green]: "
        f"{xgb_train_time:.4f} segundos"
    )

    # Predicciones
    xgb_val_pred = xgb_model.predict(x_val_pd)
    xgb_test_pred = xgb_model.predict(x_test_pd)

    xgb_val_metrics = evaluate_regression_pandas(y_val_reg, xgb_val_pred)
    xgb_test_metrics = evaluate_regression_pandas(y_test_reg, xgb_test_pred)

    # =========================================================================
    # Reporte comparativo
    # =========================================================================
    report = {
        "model_primary": "spark_gbt",
        "models_compared": ["spark_gbt", "xgboost"],
        "prefix": prefix,
        "mode": mode,
        "dataset_info": {
            "total_samples": int(total_size),
            "train_samples": int(train_size),
            "val_samples": int(val_size),
            "test_samples": int(test_size),
            "n_features": int(n_features),
            "categorical_features": int(len(categorical_features)),
            "numeric_features": int(len(numeric_features)),
        },
        "spark_gbt": {
            "framework": "pyspark",
            "algorithm": "GBTRegressor",
            "max_depth": 6,
            "max_iter": 100,
            "step_size": 0.1,
            "subsample_rate": 0.8,
            "training_time_seconds": float(spark_train_time),
            "validation": spark_val_metrics,
            "test": spark_test_metrics,
        },
        "xgboost": {
            "framework": "scikit-learn",
            "algorithm": "XGBRegressor",
            "max_depth": 6,
            "n_estimators": 100,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "training_time_seconds": float(xgb_train_time),
            "validation": xgb_val_metrics,
            "test": xgb_test_metrics,
        },
        "comparison": {
            "spark_vs_xgb_training_time_ratio": float(spark_train_time / xgb_train_time),
            "faster_model": "xgboost" if xgb_train_time < spark_train_time else "spark_gbt",
            "spark_gbt_r2_test": float(spark_test_metrics["r2"]),
            "xgboost_r2_test": float(xgb_test_metrics["r2"]),
            "better_model_r2": "spark_gbt" if spark_test_metrics["r2"] > xgb_test_metrics["r2"] else "xgboost",
        },
    }

    report_fp = outputs_base / f"spark_gbt_vs_xgboost_report_{prefix}_{mode}.json"
    save_json(report, report_fp)

    # =========================================================================
    # Mostrar tabla comparativa
    # =========================================================================
    table = Table(
        title=f"Spark GBTRegressor vs XGBoost ({prefix} | {mode})",
        header_style="bold white"
    )
    table.add_column("Aspecto", style="bold cyan")
    table.add_column("Spark GBT", style="green")
    table.add_column("XGBoost", style="yellow")

    # Tiempos
    table.add_row(
        "Tiempo entrenamiento",
        f"{spark_train_time:.4f}s",
        f"{xgb_train_time:.4f}s"
    )
    table.add_row(
        "Ratio (Spark/XGB)",
        f"{spark_train_time / xgb_train_time:.2f}x",
        "-"
    )

    # Tamaño del dataset
    table.add_row("", "", "")
    table.add_row("[bold]Dataset", "[bold]─────────────", "[bold]─────────────")
    table.add_row("Total samples", f"{total_size:,}", f"{total_size:,}")
    table.add_row("Train samples", f"{train_size:,}", f"{train_size:,}")
    table.add_row("Features", f"{n_features:,}", f"{n_features:,}")

    # Métricas de regresión (Validation)
    table.add_row("", "", "")
    table.add_row("[bold]Validation", "[bold]─────────────", "[bold]─────────────")
    table.add_row(
        "MAE",
        f"{spark_val_metrics['mae']:.4f}",
        f"{xgb_val_metrics['mae']:.4f}"
    )
    table.add_row(
        "RMSE",
        f"{spark_val_metrics['rmse']:.4f}",
        f"{xgb_val_metrics['rmse']:.4f}"
    )
    table.add_row(
        "R²",
        f"{spark_val_metrics['r2']:.4f}",
        f"{xgb_val_metrics['r2']:.4f}"
    )

    # Métricas de regresión (Test)
    table.add_row("", "", "")
    table.add_row("[bold]Test", "[bold]─────────────", "[bold]─────────────")
    table.add_row(
        "MAE",
        f"{spark_test_metrics['mae']:.4f}",
        f"{xgb_test_metrics['mae']:.4f}"
    )
    table.add_row(
        "RMSE",
        f"{spark_test_metrics['rmse']:.4f}",
        f"{xgb_test_metrics['rmse']:.4f}"
    )
    table.add_row(
        "R²",
        f"{spark_test_metrics['r2']:.4f}",
        f"{xgb_test_metrics['r2']:.4f}"
    )

    console.print(table)
    console.print(f"\n[green]✓ Reporte guardado[/green] -> {report_fp}")

    print_done(f"SPARK GBT VS XGBOOST COMPLETADO ({mode})")
    return report


def main() -> None:
    """Función principal para ejecutar desde CLI."""
    p = argparse.ArgumentParser(
        description="Entrena GBTRegressor (Spark) y lo compara con XGBoost."
    )

    p.add_argument(
        "--splits-dir",
        default="data/ml/splits_processed",
        help="Directorio de splits procesados.",
    )
    p.add_argument(
        "--prefix",
        default="completo",
        help="Prefijo de los splits.",
    )
    p.add_argument(
        "--mode",
        choices=["operational", "predictive", "both"],
        default="both",
        help="Modo de ejecución: operational, predictive o both (default).",
    )
    p.add_argument(
        "--outputs-dir",
        default="outputs/ml",
        help="Directorio de salida para reportes.",
    )

    args = p.parse_args()

    modes = (
        ["operational", "predictive"]
        if args.mode == "both"
        else [args.mode]
    )

    for mode in modes:
        run_spark_gbt_regressor(
            splits_dir=args.splits_dir,
            prefix=args.prefix,
            mode=mode,
            outputs_dir=args.outputs_dir,
        )


if __name__ == "__main__":
    main()
