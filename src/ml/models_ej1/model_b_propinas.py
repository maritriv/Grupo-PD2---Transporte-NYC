from __future__ import annotations

"""
Entrena modelos para predecir la propina de un viaje (Ejercicio 1b).

Formulacion elegida
-------------------
El builder EX1(b) deja un dataset a nivel viaje donde cada fila representa
un trayecto individual y el target principal es `target_tip_amount`.

Para mantener una formulacion coherente con el enunciado y minimizar fuga de
informacion, el script permite dos modos de seleccion de variables:

1. `strict_apriori` (recomendado)
   - elimina columnas que no deberian conocerse antes de que el viaje termine
   - por ejemplo: `dropoff_datetime`, `trip_duration_min`, `fare_amount`,
     `total_amount_std`

2. `all`
   - usa todas las columnas disponibles salvo targets y columnas no utiles

Modelos incluidos
-----------------
- dummy_median              -> baseline trivial
- elastic_net               -> baseline lineal regularizado
- random_forest             -> modelo de arboles robusto a no linealidades
- hist_gradient_boosting    -> boosting tabular potente y eficiente
- xgboost                   -> opcional, si xgboost esta instalado

La seleccion del mejor modelo se hace por MAE en validacion.
Despues se reentrena el mejor con train+val y se evalua en test.

Ejemplo:
    uv run -m src.ml.models_ej1.model_b_propinas
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rich.table import Table
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.models_ej1.split_dataset import split_model_propinas
from config.pipeline_runner import console, print_done, print_stage
from src.ml.models_ej1.common.io import read_partitioned_parquet_dir_dask, collect_dask_with_filter
from src.ml.models_ej1.common.memory import get_max_workers, get_dask_blocksize, warn_memory_config

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


DEFAULT_INPUT_DIR = "data/aggregated/ex1b/df_trip_level_tips"
DEFAULT_OUTPUT_DIR = "outputs/ml/propinas"
RANDOM_STATE = 42
SUPPORTED_MODEL_NAMES = ["dummy_median", "elastic_net", "random_forest", "hist_gradient_boosting", "xgboost"]

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


def ensure_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Faltan columnas requeridas: {missing}")


def load_ex1b_dataset(
    path: str | Path,
    use_dask: bool = True,
    dask_blocksize: str = "64MB",
    min_date: str | None = None,
    max_date: str | None = None,
) -> pd.DataFrame:
    """
    Carga el dataset EX1(b) usando Dask para datasets grandes.
    
    Args:
        path: Ruta al dataset
        use_dask: Si True, usa Dask para evitar llenar memoria
        dask_blocksize: Tamaño de bloque para Dask (ej: "32MB", "64MB", "128MB")
        min_date: Filtro de fecha mínima (se aplica antes de collect)
        max_date: Filtro de fecha máxima (se aplica antes de collect)
    """
    base = Path(path).resolve()
    if not base.exists():
        raise FileNotFoundError(
            f"No existe el input: {base}\n"
            "Genera antes EX1(b) con el builder de propinas de capa 3."
        )

    if base.is_file():
        if base.suffix != ".parquet":
            raise ValueError(f"El input debe ser un parquet o un directorio particionado: {base}")
        return pd.read_parquet(base)

    if not use_dask:
        # Cargar todo en memoria (solo para datasets pequeños)
        from src.ml.models_ej1.common.io import read_partitioned_parquet_dir
        return read_partitioned_parquet_dir(base)
    
    # Cargar con Dask (lazy)
    ddf = read_partitioned_parquet_dir_dask(base, blocksize=dask_blocksize)
    
    # Aplicar filtros ANTES de compute() para evitar cargar datos innecesarios
    # date está en formato string/object, comparar como strings (ISO format YYYY-MM-DD)
    if min_date is not None:
        min_date_str = min_date if isinstance(min_date, str) else str(min_date)
        ddf = ddf[ddf["date"] >= min_date_str]
    if max_date is not None:
        max_date_str = max_date if isinstance(max_date, str) else str(max_date)
        ddf = ddf[ddf["date"] <= max_date_str]
    
    # Collect a memoria (ahora probablemente mucho más pequeño gracias a filtros)
    return ddf.compute()


def normalize_dataset(
    df: pd.DataFrame,
    target_col: str = "target_tip_amount",
    min_date: str | None = None,
    max_date: str | None = None,
) -> pd.DataFrame:
    df = df.copy()

    ensure_columns(df, CORE_REQUIRED_COLS + [target_col], "EX1B")

    # Normalizar date y hour, luego construir timestamp_hour
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype("Int64")
    
    # Construir timestamp_hour a partir de date + hour
    df["timestamp_hour"] = df.apply(
        lambda row: pd.Timestamp(row["date"]) + pd.Timedelta(hours=int(row["hour"])) 
        if pd.notna(row["date"]) and pd.notna(row["hour"]) else pd.NaT,
        axis=1
    )

    for col in ["month", "day", "day_of_week", "is_weekend", "is_peak_hour", "is_night_hour"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in [
        "pu_location_id",
        "do_location_id",
        "passenger_count",
        "payment_type",
        "RatecodeID",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    numeric_candidates = [
        "trip_distance",
        "trip_duration_min",
        "fare_amount",
        "total_amount_std",
        "temp_c",
        "precip_mm",
        "rain_mm",
        "snowfall_mm",
        "wind_kmh",
        "borough_n_events",
        "borough_n_event_types",
        "borough_n_events_day",
        "borough_n_event_types_day",
        "city_n_events",
        "city_n_event_types",
        "restaurant_inspections_city",
        "restaurant_unique_places_city",
        "restaurant_score_mean_city",
        "restaurant_critical_count_city",
        "restaurant_good_grade_count_city",
        "restaurant_inspections_borough",
        "restaurant_unique_places_borough",
        "restaurant_score_mean_borough",
        "restaurant_critical_count_borough",
        "restaurant_good_grade_count_borough",
        "restaurant_inspections_borough_hour",
        "restaurant_score_mean_borough_hour",
        "restaurant_critical_count_borough_hour",
        "rent_listing_count_city",
        "rent_price_mean_city",
        "rent_price_median_city",
        "rent_availability_mean_city",
        "rent_min_nights_mean_city",
        "rent_listing_count_borough",
        "rent_price_mean_borough",
        "rent_price_median_borough",
        "rent_availability_mean_borough",
        "rent_accommodates_mean_borough",
        "rent_listing_count_zone",
        "rent_price_mean_zone",
        "rent_price_median_zone",
        "rent_availability_mean_zone",
        target_col,
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if target_col == "target_tip_amount":
        df = df.dropna(subset=["timestamp_hour", target_col])
        df = df[df[target_col] >= 0]
    elif target_col == "target_tip_pct":
        df = df.dropna(subset=["timestamp_hour", target_col])
        df = df[(df[target_col] >= 0) & (df[target_col] <= 100)]
    else:
        raise ValueError("target_col no soportado. Usa 'target_tip_amount' o 'target_tip_pct'.")

    if min_date is not None:
        df = df[df["date"] >= pd.to_datetime(min_date)]
    if max_date is not None:
        df = df[df["date"] <= pd.to_datetime(max_date)]

    df = df.drop_duplicates().sort_values(["timestamp_hour", target_col]).reset_index(drop=True)
    return df


def select_feature_columns(
    df: pd.DataFrame,
    feature_scope: str = "strict_apriori",
) -> tuple[list[str], dict[str, Any]]:
    all_feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    if feature_scope == "all":
        selected = [c for c in all_feature_cols if c not in SOFT_DROP_COLS]
    elif feature_scope == "strict_apriori":
        selected = [c for c in all_feature_cols if c not in STRICT_APRIORI_DROP_COLS]
    else:
        raise ValueError("feature_scope no soportado. Usa 'strict_apriori' o 'all'.")

    return selected, {
        "feature_scope": feature_scope,
        "n_all_feature_columns": int(len(all_feature_cols)),
        "n_selected_feature_columns": int(len(selected)),
        "dropped_columns": [c for c in all_feature_cols if c not in selected],
    }


def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    x = df[feature_cols].copy()

    for col in list(x.columns):
        if pd.api.types.is_datetime64_any_dtype(x[col]):
            x[col] = x[col].astype("int64") / 10**9

    cat_cols = []
    for col in x.columns:
        if x[col].dtype == "object" or str(x[col].dtype).startswith("string") or str(x[col].dtype) == "category":
            cat_cols.append(col)

    if cat_cols:
        x[cat_cols] = x[cat_cols].astype("string").fillna("<MISSING>")
        x = pd.get_dummies(x, columns=cat_cols, dummy_na=False)

    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    if train_columns is not None:
        x = x.reindex(columns=train_columns, fill_value=0)
        final_cols = train_columns
    else:
        final_cols = list(x.columns)

    return x.astype(float), final_cols


def build_models(random_state: int) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {
        "dummy_median": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DummyRegressor(strategy="median")),
            ]
        ),
        "elastic_net": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    TransformedTargetRegressor(
                        regressor=ElasticNet(alpha=0.0005, l1_ratio=0.25, max_iter=5000, random_state=random_state),
                        func=np.log1p,
                        inverse_func=np.expm1,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    TransformedTargetRegressor(
                        regressor=RandomForestRegressor(
                            n_estimators=350,
                            max_depth=None,
                            min_samples_leaf=2,
                            n_jobs=-1,
                            random_state=random_state,
                        ),
                        func=np.log1p,
                        inverse_func=np.expm1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    TransformedTargetRegressor(
                        regressor=HistGradientBoostingRegressor(
                            loss="absolute_error",
                            learning_rate=0.05,
                            max_depth=8,
                            max_iter=400,
                            min_samples_leaf=50,
                            l2_regularization=0.1,
                            random_state=random_state,
                        ),
                        func=np.log1p,
                        inverse_func=np.expm1,
                    ),
                ),
            ]
        ),
    }

    if XGBRegressor is not None:
        models["xgboost"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    TransformedTargetRegressor(
                        regressor=XGBRegressor(
                            objective="reg:absoluteerror",
                            n_estimators=500,
                            max_depth=8,
                            learning_rate=0.04,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            reg_lambda=1.0,
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                        func=np.log1p,
                        inverse_func=np.expm1,
                    ),
                ),
            ]
        )

    return models


def select_models(
    all_models: dict[str, Pipeline],
    requested_model_names: list[str] | None,
) -> dict[str, Pipeline]:
    if requested_model_names is None:
        return all_models

    selected: dict[str, Pipeline] = {}
    for model_name in requested_model_names:
        if model_name == "xgboost" and XGBRegressor is None:
            raise ImportError("Se ha pedido 'xgboost' pero xgboost no esta instalado.")
        if model_name not in all_models:
            raise ValueError(f"Modelo no disponible: {model_name}. Disponibles: {sorted(all_models)}")
        selected[model_name] = all_models[model_name]
    return selected


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "medae": float(median_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def extract_feature_importance(fitted_model: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    estimator = fitted_model.named_steps["model"]
    if isinstance(estimator, TransformedTargetRegressor):
        estimator = estimator.regressor_

    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        if coef.ndim > 1:
            importance = np.mean(np.abs(coef), axis=0)
        else:
            importance = np.abs(coef)
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    df = pd.DataFrame({"feature": feature_cols, "importance": importance})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def evaluate_and_persist_model(
    model_name: str,
    model: Pipeline,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
    target_col: str,
    save_model_flag: bool,
) -> dict[str, Any]:
    x_train, train_columns = prepare_feature_matrix(train_df, feature_cols)
    x_val, _ = prepare_feature_matrix(val_df, feature_cols, train_columns=train_columns)
    x_test, _ = prepare_feature_matrix(test_df, feature_cols, train_columns=train_columns)

    y_train = pd.to_numeric(train_df[target_col], errors="coerce").astype(float).to_numpy()
    y_val = pd.to_numeric(val_df[target_col], errors="coerce").astype(float).to_numpy()
    y_test = pd.to_numeric(test_df[target_col], errors="coerce").astype(float).to_numpy()

    model.fit(x_train, y_train)

    pred_train = np.maximum(model.predict(x_train), 0.0)
    pred_val = np.maximum(model.predict(x_val), 0.0)
    pred_test = np.maximum(model.predict(x_test), 0.0)

    split_metrics = {
        "train": compute_metrics(y_train, pred_train),
        "val": compute_metrics(y_val, pred_val),
        "test": compute_metrics(y_test, pred_test),
    }

    importance_df = extract_feature_importance(model, train_columns)
    importance_fp = out_dir / f"{model_name}_feature_importance.csv"
    importance_df.to_csv(importance_fp, index=False)

    artifact_paths: dict[str, str] = {
        "feature_importance_csv": str(importance_fp),
    }

    if save_model_flag:
        model_fp = out_dir / f"{model_name}.pkl"
        with model_fp.open("wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "feature_columns": train_columns,
                    "target_col": target_col,
                    "model_name": model_name,
                },
                f,
            )
        artifact_paths["model_pickle"] = str(model_fp)

    preds_fp = out_dir / f"{model_name}_test_predictions.parquet"
    pd.DataFrame(
        {
            "timestamp_hour": test_df["timestamp_hour"].values,
            "y_true": y_test,
            "y_pred": pred_test,
            "abs_error": np.abs(y_test - pred_test),
        }
    ).to_parquet(preds_fp, index=False)
    artifact_paths["test_predictions_parquet"] = str(preds_fp)

    report = {
        "model_name": model_name,
        "target_col": target_col,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "n_features": int(len(train_columns)),
        "metrics": split_metrics,
        "artifacts": artifact_paths,
    }
    save_json(report, out_dir / f"{model_name}_report.json")
    return report


def target_distribution_payload(df: pd.DataFrame, target_col: str) -> dict[str, float]:
    y = pd.to_numeric(df[target_col], errors="coerce")
    return {
        "count": int(y.notna().sum()),
        "mean": float(y.mean()),
        "median": float(y.median()),
        "p90": float(y.quantile(0.90)),
        "p99": float(y.quantile(0.99)),
        "zero_share": float((y.fillna(0) <= 0).mean()),
    }


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
    dask_blocksize: str | None = None,
    use_dask: bool = True,
    n_jobs: int | None = None,
) -> dict[str, Any]:
    print_stage("ML PROPINAS", "Regresion a nivel viaje", color="cyan")

    project_root = Path(__file__).resolve().parents[3]
    out_base = (project_root / out_dir).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    # Determinar configuración de memoria automáticamente
    if dask_blocksize is None:
        dask_blocksize = get_dask_blocksize()
    if n_jobs is None:
        n_jobs = get_max_workers()
    
    warn_memory_config()

    console.print(f"[cyan]Configuracion:[/cyan] dask_blocksize={dask_blocksize}, n_jobs={n_jobs}")

    df_raw = load_ex1b_dataset(
        project_root / input_dir,
        use_dask=use_dask,
        dask_blocksize=dask_blocksize,
        min_date=min_date,
        max_date=max_date,
    )
    df = normalize_dataset(df_raw, target_col=target_col, min_date=min_date, max_date=max_date)

    train_df, val_df, test_df = split_model_propinas(
        df,
        train_frac=train_frac,
        val_frac=val_frac,
    )

    feature_cols, feature_meta = select_feature_columns(df, feature_scope=feature_scope)

    summary_table = Table(title="Dataset propinas", header_style="bold cyan")
    summary_table.add_column("Campo", style="bold white")
    summary_table.add_column("Valor", justify="right")
    summary_table.add_row("input_rows", f"{len(df_raw):,}")
    summary_table.add_row("rows_after_normalization", f"{len(df):,}")
    summary_table.add_row("target", target_col)
    summary_table.add_row("feature_scope", feature_scope)
    summary_table.add_row("feature_columns", f"{len(feature_cols):,}")
    summary_table.add_row("train_rows", f"{len(train_df):,}")
    summary_table.add_row("val_rows", f"{len(val_df):,}")
    summary_table.add_row("test_rows", f"{len(test_df):,}")
    console.print(summary_table)

    all_models = build_models(random_state=random_state)
    models = select_models(all_models=all_models, requested_model_names=model_names)
    if not models:
        raise ValueError("No hay modelos seleccionados para entrenar.")

    reports: dict[str, Any] = {}
    for model_name, model in models.items():
        reports[model_name] = evaluate_and_persist_model(
            model_name=model_name,
            model=model,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_cols=feature_cols,
            out_dir=out_base,
            target_col=target_col,
            save_model_flag=save_model_flag,
        )

    best_model_name = min(reports.keys(), key=lambda name: reports[name]["metrics"]["val"]["mae"])

    dataset_profile = {
        "input_dir": str((project_root / input_dir).resolve()),
        "n_source_rows": int(len(df_raw)),
        "n_rows_after_normalization": int(len(df)),
        "target_col": target_col,
        "n_feature_columns": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "feature_selection": feature_meta,
        "split": {
            "train_frac": train_frac,
            "val_frac": val_frac,
            "test_frac": float(1.0 - train_frac - val_frac),
            "train_start": str(train_df["timestamp_hour"].min()),
            "train_end": str(train_df["timestamp_hour"].max()),
            "val_start": str(val_df["timestamp_hour"].min()),
            "val_end": str(val_df["timestamp_hour"].max()),
            "test_start": str(test_df["timestamp_hour"].min()),
            "test_end": str(test_df["timestamp_hour"].max()),
        },
        "target_distribution": {
            "full": target_distribution_payload(df, target_col),
            "train": target_distribution_payload(train_df, target_col),
            "val": target_distribution_payload(val_df, target_col),
            "test": target_distribution_payload(test_df, target_col),
        },
    }
    save_json(dataset_profile, out_base / "dataset_profile.json")

    final_summary = {
        "task": "predict_tip_amount",
        "problem_type": "regression",
        "target_definition": target_col,
        "random_state": random_state,
        "feature_scope": feature_scope,
        "selected_models": list(models.keys()),
        "best_model_by_val_mae": best_model_name,
        "outputs_dir": str(out_base),
        "models": reports,
    }
    save_json(final_summary, out_base / "training_summary.json")

    leaderboard = Table(title="Leaderboard", header_style="bold green")
    leaderboard.add_column("Modelo", style="bold white")
    leaderboard.add_column("Split")
    leaderboard.add_column("MAE", justify="right")
    leaderboard.add_column("RMSE", justify="right")
    leaderboard.add_column("MedAE", justify="right")
    leaderboard.add_column("R2", justify="right")
    for model_name, report in reports.items():
        for split_name in ["val", "test"]:
            m = report["metrics"][split_name]
            leaderboard.add_row(
                model_name,
                split_name,
                f"{m['mae']:.4f}",
                f"{m['rmse']:.4f}",
                f"{m['medae']:.4f}",
                f"{m['r2']:.4f}",
            )
    console.print(leaderboard)

    print_done(f"MODELOS DE PROPINAS GUARDADOS EN {out_base}")
    return final_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena modelos para predecir la propina de un viaje (EX1b)."
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Directorio o parquet de entrada con el dataset EX1(b).",
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
        "--no-save-model",
        action="store_true",
        help="Si se activa, no serializa los modelos entrenados.",
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
        choices=SUPPORTED_MODEL_NAMES,
        default=None,
        help="Modelos a entrenar. Si no se indica, ejecuta todos los disponibles del entorno.",
    )
    parser.add_argument(
        "--dask-blocksize",
        default=None,
        help="Tamaño de bloque para Dask (ej: '32MB', '64MB', '128MB'). Si no se especifica, se auto-detecta.",
    )
    parser.add_argument(
        "--no-dask",
        action="store_true",
        help="Desactiva Dask y carga todo en memoria (solo para datasets pequeños).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Número de jobs paralelos para sklearn. Si no se especifica, se auto-detecta (n_cpu - 1).",
    )
    args = parser.parse_args()

    run_training(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        target_col=args.target_col,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        random_state=args.random_state,
        save_model_flag=not args.no_save_model,
        min_date=args.min_date,
        max_date=args.max_date,
        feature_scope=args.feature_scope,
        model_names=args.models,
        dask_blocksize=args.dask_blocksize,
        use_dask=not args.no_dask,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()