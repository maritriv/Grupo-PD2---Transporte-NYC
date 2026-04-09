from __future__ import annotations

"""
Entrena modelos para predecir la zona de maxima demanda por franja horaria.

Formulacion elegida
-------------------
La capa 3 EX1(a) deja un panel por (timestamp_hour, pu_location_id) donde
`target_n_trips` representa la demanda observada de viajes/pickups en la zona de
pickup `pu_location_id`.

Para convertirlo en un problema de clasificacion multiclase:

1. Cada muestra pasa a ser una unica hora (`timestamp_hour`).
2. El target es la zona con mayor `target_n_trips` en esa hora.
3. En caso de empate, se elige la zona con menor `pu_location_id` para que la
   etiqueta sea determinista.
4. Las features se construyen con:
   - variables globales de calendario, meteo y eventos de esa hora
   - variables historicas por zona (`lag_*`, `rolling_*`) pivotadas a formato ancho

Asi evitamos fuga de informacion: no usamos `target_n_trips` actual como feature,
solo señales historicas disponibles antes de la hora a predecir.

Ejemplo:
    uv run -m src.ml.models_ej1.model_a_demanda
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rich.table import Table
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, top_k_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.dataset.split_dataset import split_model_demanda

from config.pipeline_runner import console, print_done, print_stage
from src.ml.dataset.modules.io import read_partitioned_parquet_dir

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


DEFAULT_INPUT_DIR = "data/aggregated/ex1a/df_demand_zone_hour_day"
DEFAULT_OUTPUT_DIR = "outputs/ml/max_demand_zone"
RANDOM_STATE = 42
SUPPORTED_MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]

GLOBAL_FEATURE_COLS = [
    "month",
    "hour",
    "hour_block_3h",
    "day_of_week",
    "is_weekend",
    "temp_c",
    "precip_mm",
    "city_n_events",
    "city_has_event",
]
OPTIONAL_GLOBAL_DEFAULTS = {
    "temp_c": np.nan,
    "precip_mm": np.nan,
    "city_n_events": 0.0,
    "city_has_event": 0,
}
ZONE_HISTORY_COLS = [
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_3h",
    "rolling_mean_24h",
]
CORE_REQUIRED_COLS = [
    "timestamp_hour",
    "date",
    "pu_location_id",
    "target_n_trips",
    "month",
    "hour",
    "hour_block_3h",
    "day_of_week",
    "is_weekend",
]

NON_FEATURE_COLS = [
    "timestamp_hour",
    "date",
    "target_zone_id",
    "winning_trips",
    "n_tied_winners",
]


def ensure_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Faltan columnas requeridas: {missing}")


def load_ex1a_dataset(path: str | Path) -> pd.DataFrame:
    base = Path(path).resolve()
    if not base.exists():
        raise FileNotFoundError(
            f"No existe el input: {base}\n"
            "Genera antes EX1(a) con: uv run -m src.procesamiento.capa3.pipelines.run_demand_zone"
        )

    if base.is_file():
        if base.suffix != ".parquet":
            raise ValueError(f"El input debe ser un parquet o un directorio particionado: {base}")
        return pd.read_parquet(base)

    return read_partitioned_parquet_dir(base)


def normalize_panel(
    df: pd.DataFrame,
    min_date: str | None = None,
    max_date: str | None = None,
) -> pd.DataFrame:
    df = df.copy()

    for col, default_value in OPTIONAL_GLOBAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default_value

    ensure_columns(df, CORE_REQUIRED_COLS + ZONE_HISTORY_COLS + GLOBAL_FEATURE_COLS, "EX1A")

    df["timestamp_hour"] = pd.to_datetime(df["timestamp_hour"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")

    int_like_cols = [
        "pu_location_id",
        "target_n_trips",
        "month",
        "hour",
        "hour_block_3h",
        "day_of_week",
        "is_weekend",
        "city_has_event",
    ]
    float_like_cols = [
        "temp_c",
        "precip_mm",
        "city_n_events",
        *ZONE_HISTORY_COLS,
    ]

    for col in int_like_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in float_like_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp_hour", "date", "pu_location_id", "target_n_trips"])

    if min_date is not None:
        df = df[df["date"] >= pd.to_datetime(min_date)]
    if max_date is not None:
        df = df[df["date"] <= pd.to_datetime(max_date)]

    df = df.drop_duplicates().sort_values(["timestamp_hour", "pu_location_id"]).reset_index(drop=True)

    duplicate_keys = df.duplicated(subset=["timestamp_hour", "pu_location_id"], keep=False)
    if duplicate_keys.any():
        dup_count = int(duplicate_keys.sum())
        raise ValueError(
            "El dataset de entrada no tiene una fila unica por (timestamp_hour, pu_location_id). "
            f"Se detectaron {dup_count} filas conflictivas."
        )

    return df


def build_multiclass_dataset(df_panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if df_panel.empty:
        raise ValueError("El dataset EX1(a) esta vacio tras la normalizacion.")

    winners_sorted = df_panel.sort_values(
        ["timestamp_hour", "target_n_trips", "pu_location_id"],
        ascending=[True, False, True],
    )
    winners = (
        winners_sorted.groupby("timestamp_hour", as_index=False)
        .first()[["timestamp_hour", "date", "pu_location_id", "target_n_trips"]]
        .rename(
            columns={
                "pu_location_id": "target_zone_id",
                "target_n_trips": "winning_trips",
            }
        )
    )

    max_trips = (
        df_panel.groupby("timestamp_hour", as_index=False)["target_n_trips"]
        .max()
        .rename(columns={"target_n_trips": "_max_trips"})
    )
    ties = (
        df_panel.merge(max_trips, on="timestamp_hour", how="left")
        .loc[lambda x: x["target_n_trips"] == x["_max_trips"]]
        .groupby("timestamp_hour", as_index=False)
        .size()
        .rename(columns={"size": "n_tied_winners"})
    )
    winners = winners.merge(ties, on="timestamp_hour", how="left")
    winners["n_tied_winners"] = winners["n_tied_winners"].fillna(1).astype("Int64")

    globals_df = (
        df_panel.sort_values(["timestamp_hour", "pu_location_id"])
        .groupby("timestamp_hour", as_index=False)
        .first()[["timestamp_hour", "date", *GLOBAL_FEATURE_COLS]]
        .set_index("timestamp_hour")
    )

    wide_parts: list[pd.DataFrame] = []
    zones = sorted(int(z) for z in df_panel["pu_location_id"].dropna().unique().tolist())
    for feature in ZONE_HISTORY_COLS:
        wide = df_panel.pivot(index="timestamp_hour", columns="pu_location_id", values=feature)
        wide = wide.reindex(columns=zones)
        wide.columns = [f"zone_{int(zone_id)}__{feature}" for zone_id in wide.columns]
        wide_parts.append(wide)

    if not wide_parts:
        raise ValueError("No se pudieron construir features historicas por zona.")

    df_wide = pd.concat(wide_parts, axis=1)
    dataset = globals_df.join(df_wide, how="inner").reset_index()
    dataset = dataset.merge(winners, on=["timestamp_hour", "date"], how="inner")
    dataset = dataset.sort_values("timestamp_hour").reset_index(drop=True)

    feature_cols = [c for c in dataset.columns if c not in NON_FEATURE_COLS]
    metadata = {
        "n_source_rows": int(len(df_panel)),
        "n_samples": int(len(dataset)),
        "n_zones_available": int(len(zones)),
        "n_classes_observed": int(dataset["target_zone_id"].nunique()),
        "n_feature_columns": int(len(feature_cols)),
        "global_feature_columns": GLOBAL_FEATURE_COLS,
        "zone_history_columns": ZONE_HISTORY_COLS,
        "feature_columns": feature_cols,
        "zones_available": zones,
        "tie_hours": int((dataset["n_tied_winners"] > 1).sum()),
        "max_tie_size": int(dataset["n_tied_winners"].max()),
    }
    return dataset, metadata


def select_feature_columns(
    df_samples: pd.DataFrame,
    train_df: pd.DataFrame,
    feature_scope: str = "train_winner_zones",
) -> tuple[list[str], dict[str, Any]]:
    all_feature_cols = [c for c in df_samples.columns if c not in NON_FEATURE_COLS]

    if feature_scope == "all":
        return all_feature_cols, {
            "feature_scope": feature_scope,
            "n_all_feature_columns": int(len(all_feature_cols)),
            "n_selected_feature_columns": int(len(all_feature_cols)),
            "selected_candidate_zones": [],
        }

    if feature_scope != "train_winner_zones":
        raise ValueError("feature_scope no soportado. Usa 'all' o 'train_winner_zones'.")

    train_candidate_zones = sorted(int(v) for v in train_df["target_zone_id"].dropna().unique().tolist())
    selected = [
        c
        for c in all_feature_cols
        if (c in GLOBAL_FEATURE_COLS)
        or any(c.startswith(f"zone_{zone_id}__") for zone_id in train_candidate_zones)
    ]

    return selected, {
        "feature_scope": feature_scope,
        "n_all_feature_columns": int(len(all_feature_cols)),
        "n_selected_feature_columns": int(len(selected)),
        "selected_candidate_zones": train_candidate_zones,
    }


def build_models(random_state: int, train_class_count: int) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

    if XGBClassifier is not None:
        models["xgboost"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        objective="multi:softprob",
                        num_class=train_class_count,
                        eval_metric="mlogloss",
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=random_state,
                        n_jobs=-1,
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
        if model_name == "xgboost" and XGBClassifier is None:
            raise ImportError(
                "Se ha pedido 'xgboost' pero la libreria xgboost no esta instalada en el entorno."
            )
        if model_name not in all_models:
            raise ValueError(
                f"Modelo no disponible: {model_name}. Disponibles ahora: {sorted(all_models)}"
            )
        selected[model_name] = all_models[model_name]

    return selected


def encode_with_global_classes(y: pd.Series, class_labels: list[int]) -> np.ndarray:
    mapping = {label: idx for idx, label in enumerate(class_labels)}
    return np.asarray([mapping[int(v)] for v in y], dtype=int)


def encode_with_train_local_classes(
    y_global: np.ndarray,
    train_class_indices: list[int],
) -> np.ndarray:
    mapping = {global_idx: local_idx for local_idx, global_idx in enumerate(train_class_indices)}
    return np.asarray([mapping[int(v)] for v in y_global], dtype=int)


def full_probability_matrix(model: Pipeline, x: pd.DataFrame, n_classes: int) -> np.ndarray:
    proba_partial = model.predict_proba(x)
    classes_seen = np.asarray(model.classes_, dtype=int)

    proba_full = np.zeros((len(x), n_classes), dtype=float)
    proba_full[:, classes_seen] = proba_partial
    return proba_full


def prepare_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    x = df[feature_cols].copy()
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    return x.astype(float)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    top_k_values: list[int],
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    labels = np.arange(y_proba.shape[1], dtype=int)
    for k in top_k_values:
        k_eff = min(int(k), int(y_proba.shape[1]))
        metrics[f"top_{int(k)}_accuracy"] = float(
            top_k_accuracy_score(y_true, y_proba, k=k_eff, labels=labels)
        )

    return metrics


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def save_confusion_matrix_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[int],
    path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_labels), dtype=int))
    df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    df_cm.index.name = "true_zone_id"
    df_cm.to_csv(path)


def extract_feature_importance(
    fitted_model: Pipeline,
    feature_cols: list[str],
) -> pd.DataFrame:
    estimator = fitted_model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        importance = np.mean(np.abs(coef), axis=0)
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
    class_labels: list[int],
    out_dir: Path,
    save_model_flag: bool,
    top_k_values: list[int],
) -> dict[str, Any]:
    x_train = prepare_feature_matrix(train_df, feature_cols)
    x_val = prepare_feature_matrix(val_df, feature_cols)
    x_test = prepare_feature_matrix(test_df, feature_cols)

    y_train_global = encode_with_global_classes(train_df["target_zone_id"], class_labels)
    y_val_global = encode_with_global_classes(val_df["target_zone_id"], class_labels)
    y_test_global = encode_with_global_classes(test_df["target_zone_id"], class_labels)

    fit_metadata: dict[str, Any] = {
        "label_encoding": "global",
        "train_seen_class_indices": sorted(np.unique(y_train_global).tolist()),
    }

    if model_name == "xgboost":
        train_seen_class_indices = fit_metadata["train_seen_class_indices"]
        y_train_fit = encode_with_train_local_classes(y_train_global, train_seen_class_indices)
        fit_metadata["label_encoding"] = "train_local"
        model.fit(x_train, y_train_fit)
    else:
        model.fit(x_train, y_train_global)

    split_payloads = {
        "train": (x_train, y_train_global),
        "val": (x_val, y_val_global),
        "test": (x_test, y_test_global),
    }
    split_metrics: dict[str, dict[str, float]] = {}

    for split_name, (x_split, y_split) in split_payloads.items():
        if fit_metadata["label_encoding"] == "train_local":
            proba_local = model.predict_proba(x_split)
            proba = np.zeros((len(x_split), len(class_labels)), dtype=float)
            proba[:, fit_metadata["train_seen_class_indices"]] = proba_local
        else:
            proba = full_probability_matrix(model, x_split, n_classes=len(class_labels))
        pred = np.argmax(proba, axis=1)
        split_metrics[split_name] = compute_metrics(
            y_true=y_split,
            y_pred=pred,
            y_proba=proba,
            top_k_values=top_k_values,
        )

        if split_name == "test":
            save_confusion_matrix_csv(
                y_true=y_split,
                y_pred=pred,
                class_labels=class_labels,
                path=out_dir / f"{model_name}_confusion_matrix_test.csv",
            )

    importance_df = extract_feature_importance(model, feature_cols)
    importance_fp = out_dir / f"{model_name}_feature_importance.csv"
    importance_df.to_csv(importance_fp, index=False)

    artifact_paths: dict[str, str] = {
        "feature_importance_csv": str(importance_fp),
        "confusion_matrix_test_csv": str(out_dir / f"{model_name}_confusion_matrix_test.csv"),
    }

    if save_model_flag:
        model_fp = out_dir / f"{model_name}.pkl"
        with model_fp.open("wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "feature_columns": feature_cols,
                    "class_labels": class_labels,
                    "fit_metadata": fit_metadata,
                    "model_name": model_name,
                },
                f,
            )
        artifact_paths["model_pickle"] = str(model_fp)

    report = {
        "model_name": model_name,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "n_features": int(len(feature_cols)),
        "n_classes": int(len(class_labels)),
        "fit_metadata": fit_metadata,
        "metrics": split_metrics,
        "artifacts": artifact_paths,
    }
    save_json(report, out_dir / f"{model_name}_report.json")
    return report


def target_distribution_payload(df: pd.DataFrame) -> dict[str, int]:
    counts = df["target_zone_id"].astype("Int64").value_counts().sort_index()
    return {str(int(zone_id)): int(count) for zone_id, count in counts.items()}


def run_training(
    input_dir: str = DEFAULT_INPUT_DIR,
    out_dir: str = DEFAULT_OUTPUT_DIR,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_state: int = RANDOM_STATE,
    save_model_flag: bool = True,
    min_date: str | None = None,
    max_date: str | None = None,
    top_k_values: list[int] | None = None,
    feature_scope: str = "train_winner_zones",
    model_names: list[str] | None = None,
) -> dict[str, Any]:
    if top_k_values is None:
        top_k_values = [3, 5]

    print_stage("ML MAX DEMAND ZONE", "Clasificacion multiclase por hora", color="cyan")

    project_root = Path(__file__).resolve().parents[3]
    out_base = (project_root / out_dir).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    df_raw = load_ex1a_dataset(project_root / input_dir)
    df_panel = normalize_panel(df_raw, min_date=min_date, max_date=max_date)
    df_samples, dataset_meta = build_multiclass_dataset(df_panel)

    train_df, val_df, test_df = split_model_demanda(
        df_samples,
        train_frac=train_frac,
        val_frac=val_frac,
    )

    class_labels = sorted(int(v) for v in df_samples["target_zone_id"].dropna().unique().tolist())
    if len(class_labels) < 2:
        raise ValueError("El target solo tiene una clase. No se puede entrenar clasificacion multiclase.")

    feature_cols, feature_meta = select_feature_columns(
        df_samples=df_samples,
        train_df=train_df,
        feature_scope=feature_scope,
    )

    summary_table = Table(title="Dataset max demand zone", header_style="bold cyan")
    summary_table.add_column("Campo", style="bold white")
    summary_table.add_column("Valor", justify="right")
    summary_table.add_row("input_rows", f"{len(df_panel):,}")
    summary_table.add_row("samples_hourly", f"{len(df_samples):,}")
    summary_table.add_row("feature_scope", feature_scope)
    summary_table.add_row("feature_columns", f"{len(feature_cols):,}")
    summary_table.add_row("classes_observed", f"{len(class_labels):,}")
    summary_table.add_row("tie_hours", f"{dataset_meta['tie_hours']:,}")
    summary_table.add_row("train_rows", f"{len(train_df):,}")
    summary_table.add_row("val_rows", f"{len(val_df):,}")
    summary_table.add_row("test_rows", f"{len(test_df):,}")
    console.print(summary_table)

    train_class_count = int(train_df["target_zone_id"].nunique())
    all_models = build_models(
        random_state=random_state,
        train_class_count=train_class_count,
    )
    models = select_models(
        all_models=all_models,
        requested_model_names=model_names,
    )

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
            class_labels=class_labels,
            out_dir=out_base,
            save_model_flag=save_model_flag,
            top_k_values=top_k_values,
        )

    dataset_profile = {
        "input_dir": str((project_root / input_dir).resolve()),
        "n_source_rows": int(len(df_raw)),
        "n_panel_rows_after_normalization": int(len(df_panel)),
        "n_hourly_samples": int(len(df_samples)),
        "n_feature_columns": int(len(feature_cols)),
        "n_classes_observed": int(len(class_labels)),
        "feature_columns": feature_cols,
        "class_labels": class_labels,
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
            "full": target_distribution_payload(df_samples),
            "train": target_distribution_payload(train_df),
            "val": target_distribution_payload(val_df),
            "test": target_distribution_payload(test_df),
        },
        "dataset_meta": dataset_meta,
    }
    save_json(dataset_profile, out_base / "dataset_profile.json")

    final_summary = {
        "task": "predict_max_demand_zone",
        "problem_type": "multiclass_classification",
        "target_definition": "Zona pu_location_id con mayor target_n_trips por timestamp_hour",
        "top_k_values": top_k_values,
        "random_state": random_state,
        "feature_scope": feature_scope,
        "selected_models": list(models.keys()),
        "outputs_dir": str(out_base),
        "models": reports,
    }
    save_json(final_summary, out_base / "training_summary.json")

    leaderboard = Table(title="Leaderboard", header_style="bold green")
    leaderboard.add_column("Modelo", style="bold white")
    leaderboard.add_column("Split")
    leaderboard.add_column("Accuracy", justify="right")
    leaderboard.add_column("F1 macro", justify="right")
    leaderboard.add_column("Top-3", justify="right")
    leaderboard.add_column("Top-5", justify="right")
    for model_name, report in reports.items():
        for split_name in ["val", "test"]:
            m = report["metrics"][split_name]
            leaderboard.add_row(
                model_name,
                split_name,
                f"{m['accuracy']:.4f}",
                f"{m['f1_macro']:.4f}",
                f"{m.get('top_3_accuracy', float('nan')):.4f}",
                f"{m.get('top_5_accuracy', float('nan')):.4f}",
            )
    console.print(leaderboard)

    print_done(f"MODELOS DE MAXIMA DEMANDA GUARDADOS EN {out_base}")
    return final_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena un baseline para predecir la zona de maxima demanda por hora."
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Directorio o parquet de entrada con el dataset EX1(a).",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directorio de salida para metricas, artefactos y modelos.",
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
        "--top-k",
        nargs="+",
        type=int,
        default=[3, 5],
        help="Valores de k para top-k accuracy.",
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Si se activa, no serializa los modelos entrenados.",
    )
    parser.add_argument(
        "--feature-scope",
        choices=["train_winner_zones", "all"],
        default="train_winner_zones",
        help=(
            "Seleccion de features por zona. "
            "'train_winner_zones' reduce ruido usando solo zonas candidatas en train."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODEL_NAMES,
        default=None,
        help=(
            "Modelos a entrenar. "
            "Si no se indica, ejecuta todos los disponibles del entorno."
        ),
    )
    args = parser.parse_args()

    run_training(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        random_state=args.random_state,
        save_model_flag=not args.no_save_model,
        min_date=args.min_date,
        max_date=args.max_date,
        top_k_values=args.top_k,
        feature_scope=args.feature_scope,
        model_names=args.models,
    )


if __name__ == "__main__":
    main()
