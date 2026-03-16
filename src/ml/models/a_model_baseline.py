from __future__ import annotations

"""
Script: a_model_baseline.py

Modelo baseline para predecir stress_score (regresión) e is_stress (clasificación).

Implementa 3 estrategias naive como referencia mínima contra la que comparar
cualquier modelo posterior (Random Forest, Boosting, etc.):

    1. Global Mean   — predice siempre la media del train
    2. Global Median — predice siempre la mediana del train
    3. Group Mean    — predice la media por (service_type, hour, is_weekend)

Métricas que reporta:
    Regresión:     MAE, RMSE, R²
    Clasificación: Accuracy, Precision, Recall, F1

-------------------------------------------------------
EJECUCIÓN
-------------------------------------------------------

1. Modo normal (usa splits por defecto)

    uv run -m src.ml.models.a_model_baseline

    Lee:
        data/ml/splits/completo_train.parquet
        data/ml/splits/completo_val.parquet
        data/ml/splits/completo_test.parquet

    Genera:
        outputs/ml/baseline_report.json


2. Usar splits de un rango concreto

    uv run -m src.ml.models.a_model_baseline \\
        --prefix rango_2024-01-01__2024-01-14


3. Evaluar solo sobre test (omitir val)

    uv run -m src.ml.models.a_model_baseline --skip-val

"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Targets y features del grupo-mean
# ──────────────────────────────────────────────
TARGET_REG = "stress_score"
TARGET_CLF = "is_stress"

# Columnas que definen los grupos del baseline más elaborado.
# Son features "gruesas" que capturan patrones básicos sin sobreajustar.
GROUP_COLS = ["service_type", "hour", "is_weekend"]


# ──────────────────────────────────────────────
# Métricas
# ──────────────────────────────────────────────
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, R² sobre arrays numpy."""
    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy, Precision, Recall, F1 para binario (0/1)."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ──────────────────────────────────────────────
# Estrategias baseline
# ──────────────────────────────────────────────
class BaselineGlobalMean:
    """Predice siempre la media de stress_score del train."""

    name = "global_mean"

    def fit(self, train: pd.DataFrame) -> None:
        self.mean_ = float(train[TARGET_REG].mean())
        self.threshold_ = float(train[TARGET_REG].quantile(0.90))

    def predict_regression(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.mean_)

    def predict_classification(self, df: pd.DataFrame) -> np.ndarray:
        scores = self.predict_regression(df)
        return (scores >= self.threshold_).astype(int)


class BaselineGlobalMedian:
    """Predice siempre la mediana de stress_score del train."""

    name = "global_median"

    def fit(self, train: pd.DataFrame) -> None:
        self.median_ = float(train[TARGET_REG].median())
        self.threshold_ = float(train[TARGET_REG].quantile(0.90))

    def predict_regression(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.median_)

    def predict_classification(self, df: pd.DataFrame) -> np.ndarray:
        scores = self.predict_regression(df)
        return (scores >= self.threshold_).astype(int)


class BaselineGroupMean:
    """
    Predice la media de stress_score por grupo (service_type, hour, is_weekend).
    Si un grupo del test no aparecía en train, cae al fallback global.

    Es el baseline más "informativo": captura que la tensión depende del tipo
    de servicio, la hora del día y si es fin de semana.
    """

    name = "group_mean"

    def fit(self, train: pd.DataFrame) -> None:
        self.global_mean_ = float(train[TARGET_REG].mean())
        self.threshold_ = float(train[TARGET_REG].quantile(0.90))

        self.group_means_ = (
            train.groupby(GROUP_COLS, as_index=False)[TARGET_REG]
            .mean()
            .rename(columns={TARGET_REG: "_pred"})
        )

    def predict_regression(self, df: pd.DataFrame) -> np.ndarray:
        merged = df[GROUP_COLS].merge(self.group_means_, on=GROUP_COLS, how="left")
        preds = merged["_pred"].fillna(self.global_mean_).to_numpy(dtype=float)
        return preds

    def predict_classification(self, df: pd.DataFrame) -> np.ndarray:
        scores = self.predict_regression(df)
        return (scores >= self.threshold_).astype(int)


# ──────────────────────────────────────────────
# Evaluación
# ──────────────────────────────────────────────
def evaluate_baseline(
    model,
    df: pd.DataFrame,
    split_name: str,
) -> dict[str, Any]:
    """Evalúa un baseline sobre un split y devuelve un dict con métricas."""
    y_true_reg = df[TARGET_REG].to_numpy(dtype=float)
    y_true_clf = df[TARGET_CLF].to_numpy(dtype=int)

    y_pred_reg = model.predict_regression(df)
    y_pred_clf = model.predict_classification(df)

    return {
        "split": split_name,
        "n_rows": len(df),
        "regression": regression_metrics(y_true_reg, y_pred_reg),
        "classification": classification_metrics(y_true_clf, y_pred_clf),
    }


# ──────────────────────────────────────────────
# Ejecución principal
# ──────────────────────────────────────────────
def run_baselines(
    splits_dir: str = "data/ml/splits",
    prefix: str = "completo",
    out_dir: str = "outputs/ml",
    skip_val: bool = False,
) -> dict[str, Any]:
    """
    Carga splits, entrena los 3 baselines sobre train,
    evalúa sobre val y test, y guarda un JSON con los resultados.
    """
    project_root = Path(__file__).resolve().parents[3]
    base = project_root / splits_dir

    train_fp = base / f"{prefix}_train.parquet"
    val_fp = base / f"{prefix}_val.parquet"
    test_fp = base / f"{prefix}_test.parquet"

    if not train_fp.exists():
        raise FileNotFoundError(
            f"No se encontró el split de train: {train_fp}\n"
            f"Ejecuta primero:  uv run -m src.ml.dataset.b_split_dataset --prefix {prefix}"
        )

    print(f"📂 Cargando splits (prefix={prefix})...")
    train = pd.read_parquet(train_fp)

    eval_splits: list[tuple[str, pd.DataFrame]] = []
    if not skip_val and val_fp.exists():
        eval_splits.append(("val", pd.read_parquet(val_fp)))
    if test_fp.exists():
        eval_splits.append(("test", pd.read_parquet(test_fp)))

    if not eval_splits:
        raise FileNotFoundError(
            f"No se encontraron splits de evaluación en {base}.\n"
            f"Se necesita al menos {prefix}_test.parquet."
        )

    # Verificar columnas necesarias
    for col in [TARGET_REG, TARGET_CLF] + GROUP_COLS:
        if col not in train.columns:
            raise ValueError(f"Falta columna '{col}' en el dataset. Revisa a_build_dataset.py.")

    print(f"   train : {len(train):>10,} filas")
    for name, split_df in eval_splits:
        print(f"   {name:5s} : {len(split_df):>10,} filas")

    # Instanciar y entrenar baselines
    baselines = [BaselineGlobalMean(), BaselineGlobalMedian(), BaselineGroupMean()]
    for bl in baselines:
        bl.fit(train)

    # Evaluar
    full_report: dict[str, Any] = {
        "prefix": prefix,
        "train_rows": len(train),
        "target_reg": TARGET_REG,
        "target_clf": TARGET_CLF,
        "group_cols": GROUP_COLS,
        "train_stats": {
            "mean": round(float(train[TARGET_REG].mean()), 4),
            "median": round(float(train[TARGET_REG].median()), 4),
            "std": round(float(train[TARGET_REG].std()), 4),
            "p90_threshold": round(float(train[TARGET_REG].quantile(0.90)), 4),
            "is_stress_rate": round(float(train[TARGET_CLF].mean()), 4),
        },
        "baselines": {},
    }

    for bl in baselines:
        print(f"\n{'─'*50}")
        print(f"🔹 Baseline: {bl.name}")
        print(f"{'─'*50}")

        bl_results: dict[str, Any] = {}

        for split_name, split_df in eval_splits:
            metrics = evaluate_baseline(bl, split_df, split_name)
            bl_results[split_name] = metrics

            reg = metrics["regression"]
            clf = metrics["classification"]
            print(f"  [{split_name}] Regresión  → MAE={reg['mae']:.4f}  RMSE={reg['rmse']:.4f}  R²={reg['r2']:.4f}")
            print(f"  [{split_name}] Clasif.    → Acc={clf['accuracy']:.4f}  P={clf['precision']:.4f}  R={clf['recall']:.4f}  F1={clf['f1']:.4f}")

        full_report["baselines"][bl.name] = bl_results

    # Guardar reporte
    out_path = project_root / out_dir / "baseline_report.json"
    os.makedirs(out_path.parent, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Reporte guardado -> {out_path}")
    return full_report


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="Evalúa modelos baseline (mean, median, group_mean) "
        "sobre los splits generados por dataset/b_split_dataset.py."
    )
    p.add_argument(
        "--splits-dir",
        default="data/ml/splits",
        help="Carpeta con los splits (default: data/ml/splits)",
    )
    p.add_argument(
        "--prefix",
        default="completo",
        help="Prefijo de los splits (default: completo)",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/ml",
        help="Carpeta de salida del reporte (default: outputs/ml)",
    )
    p.add_argument(
        "--skip-val",
        action="store_true",
        help="Omitir evaluación sobre validación (solo test)",
    )
    args = p.parse_args()

    run_baselines(
        splits_dir=args.splits_dir,
        prefix=args.prefix,
        out_dir=args.out_dir,
        skip_val=args.skip_val,
    )


if __name__ == "__main__":
    main()
