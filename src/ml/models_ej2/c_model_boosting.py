from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from xgboost import XGBClassifier, XGBRegressor

from config.pipeline_runner import console, print_done, print_stage

DEFAULT_TARGET_REG = "target_stress_t24"
DEFAULT_TARGET_CLF = "target_is_stress_t24"

DEFAULT_DATASET_DIR = "data/aggregated/ex_stress/df_stress_zone_hour_day"
DEFAULT_OUTPUTS_DIR = "outputs/ml/ej2/boosting"

RANDOM_STATE = 42

TARGET_COLS = {
    "target_stress_t1",
    "target_stress_t3",
    "target_stress_t24",
    "target_is_stress_t1",
    "target_is_stress_t3",
    "target_is_stress_t24",
}

DROP_FEATURE_COLS = {
    "date",
    "datetime",
    "pickup_datetime",
    "dropoff_datetime",
    "timestamp",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path.resolve()
    return (project_root() / path).resolve()


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def read_partitioned_parquet(dataset_dir: str | Path) -> pd.DataFrame:
    base = resolve_project_path(dataset_dir)
    if not base.exists():
        raise FileNotFoundError(f"No existe el dataset de EX2: {base}")

    files = sorted(base.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No se encontraron parquets en: {base}")

    console.print(f"[cyan]Leyendo parquets[/cyan] -> {len(files):,} archivos")
    return pd.concat([pd.read_parquet(fp) for fp in files], ignore_index=True)


def add_time_sort_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "datetime" in out.columns:
        out["_sort_time"] = pd.to_datetime(out["datetime"], errors="coerce")
    elif "timestamp" in out.columns:
        out["_sort_time"] = pd.to_datetime(out["timestamp"], errors="coerce")
    elif "date" in out.columns and "hour" in out.columns:
        out["_sort_time"] = pd.to_datetime(out["date"], errors="coerce") + pd.to_timedelta(
            pd.to_numeric(out["hour"], errors="coerce").fillna(0),
            unit="h",
        )
    elif {"year", "month", "day", "hour"}.issubset(out.columns):
        out["_sort_time"] = pd.to_datetime(
            dict(
                year=pd.to_numeric(out["year"], errors="coerce"),
                month=pd.to_numeric(out["month"], errors="coerce"),
                day=pd.to_numeric(out["day"], errors="coerce"),
            ),
            errors="coerce",
        ) + pd.to_timedelta(pd.to_numeric(out["hour"], errors="coerce").fillna(0), unit="h")
    else:
        out["_sort_time"] = pd.RangeIndex(len(out))

    return out


def prepare_targets(
    df: pd.DataFrame,
    target_reg: str,
    target_clf: str,
) -> pd.DataFrame:
    if target_reg not in df.columns:
        raise ValueError(f"Falta target de regresión: {target_reg}")
    if target_clf not in df.columns:
        raise ValueError(f"Falta target de clasificación: {target_clf}")

    out = df.copy()
    out[target_reg] = pd.to_numeric(out[target_reg], errors="coerce")
    out[target_clf] = pd.to_numeric(out[target_clf], errors="coerce")

    out = out.dropna(subset=[target_reg, target_clf])
    out[target_clf] = out[target_clf].astype(int)

    return out.reset_index(drop=True)


def split_train_val_test(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> dict[str, pd.DataFrame]:
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("val_size y test_size deben estar entre 0 y 1 y sumar menos de 1.")

    df = add_time_sort_column(df)
    df = df.sort_values("_sort_time").reset_index(drop=True)

    n = len(df)
    n_test = int(n * test_size)
    n_val = int(n * val_size)

    train = df.iloc[: n - n_val - n_test].copy()
    val = df.iloc[n - n_val - n_test : n - n_test].copy()
    test = df.iloc[n - n_test :].copy()

    return {"train": train, "val": val, "test": test}


def split_xy(
    df: pd.DataFrame,
    target_reg: str,
    target_clf: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    y_reg = df[target_reg].copy()
    y_clf = df[target_clf].copy().astype(int)

    drop_cols = set(TARGET_COLS) | set(DROP_FEATURE_COLS) | {"_sort_time"}
    x = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()

    return x, y_reg, y_clf


def sanitize_feature_names(*dfs: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    cleaned = []

    for df in dfs:
        out = df.copy()
        out.columns = (
            out.columns.astype(str)
            .str.replace("[", "(", regex=False)
            .str.replace("]", ")", regex=False)
            .str.replace("<", "_lt_", regex=False)
            .str.replace(">", "_gt_", regex=False)
        )
        cleaned.append(out)

    return tuple(cleaned)


def preprocess_features(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for df in [x_train, x_val, x_test]:
        bool_cols = df.select_dtypes(include=["bool"]).columns
        for col in bool_cols:
            df[col] = df[col].astype(int)

    cat_cols = x_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    x_train = pd.get_dummies(x_train, columns=cat_cols, dummy_na=True)
    x_val = pd.get_dummies(x_val, columns=[c for c in cat_cols if c in x_val.columns], dummy_na=True)
    x_test = pd.get_dummies(x_test, columns=[c for c in cat_cols if c in x_test.columns], dummy_na=True)

    x_val = x_val.reindex(columns=x_train.columns, fill_value=0)
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

    for df in [x_train, x_val, x_test]:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    all_null_cols = [c for c in x_train.columns if x_train[c].isna().all()]
    if all_null_cols:
        x_train = x_train.drop(columns=all_null_cols)
        x_val = x_val.drop(columns=all_null_cols, errors="ignore")
        x_test = x_test.drop(columns=all_null_cols, errors="ignore")

    medians = x_train.median(numeric_only=True)

    x_train = x_train.fillna(medians).fillna(0)
    x_val = x_val.fillna(medians).fillna(0)
    x_test = x_test.fillna(medians).fillna(0)

    x_train, x_val, x_test = sanitize_feature_names(x_train, x_val, x_test)

    return x_train, x_val, x_test


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_classification(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_json(data: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_feature_importance(model, feature_names: list[str], out_path: Path) -> None:
    importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(30)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(out_path, index=False)


def run_xgboost_models(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    outputs_dir: str = DEFAULT_OUTPUTS_DIR,
    target_reg: str = DEFAULT_TARGET_REG,
    target_clf: str = DEFAULT_TARGET_CLF,
    val_size: float = 0.15,
    test_size: float = 0.15,
    fit_all_data: bool = False,
    sample_frac: float | None = None,
) -> dict[str, Any]:
    print_stage(
        "ML BOOSTING",
        f"XGBoost regresión + clasificación | EX2 estrés urbano | target={target_reg}",
        color="magenta",
    )

    dataset_path = resolve_project_path(dataset_dir)
    outputs_base = resolve_project_path(outputs_dir)
    outputs_base.mkdir(parents=True, exist_ok=True)

    raw_df = read_partitioned_parquet(dataset_path)
    df = prepare_targets(raw_df, target_reg=target_reg, target_clf=target_clf)

    if sample_frac is not None and sample_frac < 1.0:
        console.print(f"[cyan]Sampleando dataset[/cyan] -> {sample_frac:.1%} del total")
        df = df.sample(frac=sample_frac, random_state=RANDOM_STATE).reset_index(drop=True)

    splits = split_train_val_test(df, val_size=val_size, test_size=test_size)

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    x_train, y_train_reg, y_train_clf = split_xy(train_df, target_reg, target_clf)
    x_val, y_val_reg, y_val_clf = split_xy(val_df, target_reg, target_clf)
    x_test, y_test_reg, y_test_clf = split_xy(test_df, target_reg, target_clf)

    x_train, x_val, x_test = preprocess_features(x_train, x_val, x_test)

    console.print(
        f"[cyan]Dataset cargado[/cyan] -> {dataset_path} | "
        f"filas={len(df):,} | train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,}"
    )
    console.print(f"[cyan]Target regresión[/cyan] -> {target_reg}")
    console.print(f"[cyan]Target clasificación[/cyan] -> {target_clf}")
    console.print(f"[cyan]Features finales[/cyan] -> {x_train.shape[1]:,}")
    console.print(
        f"[cyan]Clase positiva train[/cyan] -> "
        f"{int((y_train_clf == 1).sum()):,}/{len(y_train_clf):,}"
    )

    reg_model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=4,  # Limitar a 4 cores para reducir memoria
    )

    reg_model.fit(
        x_train,
        y_train_reg,
        eval_set=[(x_val, y_val_reg)],
        verbose=False,
    )

    val_pred_reg = reg_model.predict(x_val)
    test_pred_reg = reg_model.predict(x_test)

    reg_val_metrics = evaluate_regression(y_val_reg, val_pred_reg)
    reg_test_metrics = evaluate_regression(y_test_reg, test_pred_reg)

    pos_count = int((y_train_clf == 1).sum())
    neg_count = int((y_train_clf == 0).sum())
    scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0

    clf_model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=4,  # Limitar a 4 cores para reducir memoria
    )

    clf_model.fit(
        x_train,
        y_train_clf,
        eval_set=[(x_val, y_val_clf)],
        verbose=False,
    )

    val_pred_clf = clf_model.predict(x_val)
    test_pred_clf = clf_model.predict(x_test)

    clf_val_metrics = evaluate_classification(y_val_clf, val_pred_clf)
    clf_test_metrics = evaluate_classification(y_test_clf, test_pred_clf)

    final_training_data = "train"

    if fit_all_data:
        console.print(
            "[cyan]Reentrenando XGBoost final con train+val+test[/cyan] "
            "(modelo para despliegue)"
        )

        x_all = pd.concat([x_train, x_val, x_test], ignore_index=True)
        y_reg_all = pd.concat([y_train_reg, y_val_reg, y_test_reg], ignore_index=True)
        y_clf_all = pd.concat([y_train_clf, y_val_clf, y_test_clf], ignore_index=True)

        reg_model.fit(x_all, y_reg_all, verbose=False)
        clf_model.fit(x_all, y_clf_all, verbose=False)

        final_training_data = "train_val_test"

    reg_model_fp = outputs_base / f"xgboost_regressor_{target_reg}.joblib"
    clf_model_fp = outputs_base / f"xgboost_classifier_{target_clf}.joblib"
    reg_importance_fp = outputs_base / "xgboost_regression_feature_importance.csv"
    clf_importance_fp = outputs_base / "xgboost_classification_feature_importance.csv"
    feature_cols_fp = outputs_base / "xgboost_feature_columns.json"
    report_fp = outputs_base / "xgboost_report.json"

    joblib.dump(reg_model, reg_model_fp)
    joblib.dump(clf_model, clf_model_fp)

    save_feature_importance(reg_model, list(x_train.columns), reg_importance_fp)
    save_feature_importance(clf_model, list(x_train.columns), clf_importance_fp)
    save_json({"feature_columns": list(x_train.columns)}, feature_cols_fp)

    report = {
        "model": "xgboost",
        "task": "ej2_stress_urban",
        "dataset_dir": str(dataset_path),
        "target": target_reg,
        "target_regression": target_reg,
        "target_classification": target_clf,
        "fit_all_data": bool(fit_all_data),
        "final_training_data": final_training_data,
        "sample_frac": sample_frac,
        "rmse_test": reg_test_metrics["rmse"],
        "mae_test": reg_test_metrics["mae"],
        "r2_test": reg_test_metrics["r2"],
        "accuracy_test": clf_test_metrics["accuracy"],
        "f1_test": clf_test_metrics["f1"],
        "n_rows": int(len(df)),
        "n_features": int(x_train.shape[1]),
        "splits": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
            "val_size": float(val_size),
            "test_size": float(test_size),
        },
        "regression": {
            "val": reg_val_metrics,
            "test": reg_test_metrics,
        },
        "classification": {
            "val": clf_val_metrics,
            "test": clf_test_metrics,
            "scale_pos_weight": float(scale_pos_weight),
            "positive_train": int(pos_count),
            "negative_train": int(neg_count),
        },
        "artifacts": {
            "regression_model": str(reg_model_fp),
            "classification_model": str(clf_model_fp),
            "feature_columns": str(feature_cols_fp),
            "regression_feature_importance_csv": str(reg_importance_fp),
            "classification_feature_importance_csv": str(clf_importance_fp),
        },
    }

    save_json(report, report_fp)

    table = Table(title="XGBoost EX2 - Estrés urbano", header_style="bold magenta")
    table.add_column("Bloque", style="bold white")
    table.add_column("Métrica")
    table.add_column("Valor", justify="right")

    table.add_row("Regresión val", "MAE", f"{reg_val_metrics['mae']:.4f}")
    table.add_row("Regresión val", "RMSE", f"{reg_val_metrics['rmse']:.4f}")
    table.add_row("Regresión val", "R²", f"{reg_val_metrics['r2']:.4f}")
    table.add_row("Regresión test", "MAE", f"{reg_test_metrics['mae']:.4f}")
    table.add_row("Regresión test", "RMSE", f"{reg_test_metrics['rmse']:.4f}")
    table.add_row("Regresión test", "R²", f"{reg_test_metrics['r2']:.4f}")

    table.add_row("Clasificación val", "Accuracy", f"{clf_val_metrics['accuracy']:.4f}")
    table.add_row("Clasificación val", "Precision", f"{clf_val_metrics['precision']:.4f}")
    table.add_row("Clasificación val", "Recall", f"{clf_val_metrics['recall']:.4f}")
    table.add_row("Clasificación val", "F1", f"{clf_val_metrics['f1']:.4f}")
    table.add_row("Clasificación test", "Accuracy", f"{clf_test_metrics['accuracy']:.4f}")
    table.add_row("Clasificación test", "Precision", f"{clf_test_metrics['precision']:.4f}")
    table.add_row("Clasificación test", "Recall", f"{clf_test_metrics['recall']:.4f}")
    table.add_row("Clasificación test", "F1", f"{clf_test_metrics['f1']:.4f}")

    console.print(table)
    console.print(f"[green]Reporte guardado[/green] -> {report_fp}")
    console.print(f"[green]Modelo regresión[/green] -> {reg_model_fp}")
    console.print(f"[green]Modelo clasificación[/green] -> {clf_model_fp}")
    console.print(f"[green]Importancias regresión[/green] -> {reg_importance_fp}")
    console.print(f"[green]Importancias clasificación[/green] -> {clf_importance_fp}")

    if fit_all_data:
        console.print("[green]Modelo final reentrenado con train+val+test[/green]")

    print_done("XGBOOST EX2 COMPLETADO")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena XGBoost para EX2 sobre el dataset actual de estrés urbano."
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help="Dataset particionado de EX2.",
    )
    parser.add_argument(
        "--outputs-dir",
        default=DEFAULT_OUTPUTS_DIR,
        help="Directorio de salida para modelos, reportes e importancias.",
    )
    parser.add_argument(
        "--target-reg",
        default=DEFAULT_TARGET_REG,
        help="Columna target de regresión.",
    )
    parser.add_argument(
        "--target-clf",
        default=DEFAULT_TARGET_CLF,
        help="Columna target de clasificación.",
    )
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument(
        "--fit-all-data",
        action="store_true",
        help="Reentrena el modelo final con train+val+test para despliegue.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Fracción del dataset a usar (0.0-1.0). Útil para reducir memoria.",
    )

    args = parser.parse_args()

    run_xgboost_models(
        dataset_dir=args.dataset_dir,
        outputs_dir=args.outputs_dir,
        target_reg=args.target_reg,
        target_clf=args.target_clf,
        val_size=args.val_size,
        test_size=args.test_size,
        fit_all_data=args.fit_all_data,
        sample_frac=args.sample_frac,
    )


if __name__ == "__main__":
    main()