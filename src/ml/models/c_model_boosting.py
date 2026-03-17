from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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

TARGET_REG = "stress_score"
TARGET_CLF = "is_stress"
VALID_MODES = {"operational", "predictive"}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_processed_splits(
    splits_dir: str = "data/ml/splits_processed",
    prefix: str = "completo",
    mode: str = "operational",
) -> dict[str, pd.DataFrame]:
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


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if TARGET_REG not in df.columns or TARGET_CLF not in df.columns:
        raise ValueError(f"Faltan targets '{TARGET_REG}' y/o '{TARGET_CLF}' en el split.")

    x = df.drop(columns=[TARGET_REG, TARGET_CLF]).copy()
    y_reg = df[TARGET_REG].copy()
    y_clf = df[TARGET_CLF].copy()
    return x, y_reg, y_clf


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


# -----------------------------------------------------------------------------
# Main training
# -----------------------------------------------------------------------------
def run_xgboost_models(
    splits_dir: str = "data/ml/splits_processed",
    prefix: str = "completo",
    mode: str = "operational",
    outputs_dir: str = "outputs/ml",
) -> dict[str, Any]:
    if mode not in VALID_MODES:
        raise ValueError(f"Modo no soportado: {mode}. Usa uno de {sorted(VALID_MODES)}")

    print_stage("ML BOOSTING", f"XGBoost regresión + clasificación | mode={mode}", color="magenta")

    project_root = Path(__file__).resolve().parents[3]
    outputs_base = (project_root / outputs_dir).resolve()
    outputs_base.mkdir(parents=True, exist_ok=True)

    splits = load_processed_splits(
        splits_dir=splits_dir,
        prefix=prefix,
        mode=mode,
    )

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    x_train, y_train_reg, y_train_clf = split_xy(train_df)
    x_val, y_val_reg, y_val_clf = split_xy(val_df)
    x_test, y_test_reg, y_test_clf = split_xy(test_df)

    console.print(
        f"[cyan]Splits cargados[/cyan] -> "
        f"mode={mode} | train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,}"
    )
    console.print(f"[cyan]Features[/cyan] -> {x_train.shape[1]:,}")

    # -------------------------------------------------------------------------
    # REGRESIÓN
    # -------------------------------------------------------------------------
    reg_model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
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

    # -------------------------------------------------------------------------
    # CLASIFICACIÓN
    # -------------------------------------------------------------------------
    pos_count = int((y_train_clf == 1).sum())
    neg_count = int((y_train_clf == 0).sum())
    scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0

    clf_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
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

    # -------------------------------------------------------------------------
    # Importancias
    # -------------------------------------------------------------------------
    reg_importance = (
        pd.DataFrame(
            {
                "feature": x_train.columns,
                "importance": reg_model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(20)
    )

    clf_importance = (
        pd.DataFrame(
            {
                "feature": x_train.columns,
                "importance": clf_model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(20)
    )

    reg_importance_fp = outputs_base / f"xgboost_regression_feature_importance_{prefix}_{mode}.csv"
    clf_importance_fp = outputs_base / f"xgboost_classification_feature_importance_{prefix}_{mode}.csv"

    reg_importance.to_csv(reg_importance_fp, index=False)
    clf_importance.to_csv(clf_importance_fp, index=False)

    # -------------------------------------------------------------------------
    # Reporte
    # -------------------------------------------------------------------------
    report = {
        "model": "xgboost",
        "prefix": prefix,
        "mode": mode,
        "splits_dir": str((project_root / splits_dir).resolve()),
        "n_features": int(x_train.shape[1]),
        "regression": {
            "val": reg_val_metrics,
            "test": reg_test_metrics,
        },
        "classification": {
            "val": clf_val_metrics,
            "test": clf_test_metrics,
            "scale_pos_weight": float(scale_pos_weight),
        },
        "artifacts": {
            "regression_feature_importance_csv": str(reg_importance_fp),
            "classification_feature_importance_csv": str(clf_importance_fp),
        },
    }

    report_fp = outputs_base / f"xgboost_report_{prefix}_{mode}.json"
    save_json(report, report_fp)

    # -------------------------------------------------------------------------
    # Mostrar resumen
    # -------------------------------------------------------------------------
    table = Table(title=f"XGBoost ({prefix} | {mode})", header_style="bold magenta")
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
    console.print(f"[green]Importancias regresión[/green] -> {reg_importance_fp}")
    console.print(f"[green]Importancias clasificación[/green] -> {clf_importance_fp}")

    print_done(f"XGBOOST COMPLETADO ({mode})")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Entrena XGBoost sobre splits procesados.")

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
        help="Directorio de salida para reportes e importancias.",
    )

    args = p.parse_args()

    modes = (
        ["operational", "predictive"]
        if args.mode == "both"
        else [args.mode]
    )

    for mode in modes:
        run_xgboost_models(
            splits_dir=args.splits_dir,
            prefix=args.prefix,
            mode=mode,
            outputs_dir=args.outputs_dir,
        )


if __name__ == "__main__":
    main()