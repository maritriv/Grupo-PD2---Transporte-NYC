from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.table import Table

from config.pipeline_runner import console, print_done, print_stage
from src.ml.models_ej2.common.io import ensure_project_dir, resolve_project_path, save_json

DEFAULT_DATASET_DIR = "data/aggregated/ex_stress/df_stress_zone_hour_day"
DEFAULT_OUTPUTS_DIR = "outputs/ml/ej2/baseline"

DEFAULT_TARGET_REG = "target_stress_t24"
DEFAULT_TIME_COL = "timestamp_hour"
DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_VAL_FRAC = 0.15
DEFAULT_THRESHOLD_QUANTILE = 0.90

BASELINE_ROLE_MAP = {
    "persistence_current_stress": "primary_naive",
    "zone_hour_weekend_mean": "strong_reference",
    "global_mean": "sanity_floor",
}


def infer_classification_target(target_col: str) -> str:
    if not target_col.startswith("target_stress_t"):
        raise ValueError(
            "target_col debe tener formato 'target_stress_tH', por ejemplo 'target_stress_t1'."
        )
    return target_col.replace("target_stress_", "target_is_stress_", 1)


def read_partitioned_parquet(
    dataset_dir: str | Path,
    required_columns: list[str],
    optional_columns: list[str],
) -> tuple[pd.DataFrame, Path]:
    dataset_path = resolve_project_path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"No existe el dataset de entrada: {dataset_path}\n"
            "Generalo primero con: uv run -m src.procesamiento.capa3.ejercicios.ex2_stress"
        )

    files = sorted(dataset_path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No se encontraron parquets en: {dataset_path}")

    schema_df = pd.read_parquet(files[0])
    available_cols = set(schema_df.columns)

    missing_required = [col for col in required_columns if col not in available_cols]
    if missing_required:
        available_target_cols = sorted(col for col in available_cols if col.startswith("target_"))
        available_target_msg = (
            "\nTargets disponibles en este dataset: " + ", ".join(available_target_cols)
            if available_target_cols
            else ""
        )
        raise ValueError(
            "Faltan columnas necesarias en el dataset de EX2: "
            + ", ".join(sorted(missing_required))
            + available_target_msg
        )

    selected_cols = required_columns + [col for col in optional_columns if col in available_cols]

    console.print(f"[cyan]Leyendo parquets[/cyan] -> {len(files):,} archivos")
    parts = [pd.read_parquet(fp, columns=selected_cols) for fp in files]
    return pd.concat(parts, ignore_index=True), dataset_path


def prepare_dataset(
    df: pd.DataFrame,
    *,
    time_col: str,
    target_reg_col: str,
    target_clf_col: str,
) -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out[target_reg_col] = pd.to_numeric(out[target_reg_col], errors="coerce")
    out[target_clf_col] = pd.to_numeric(out[target_clf_col], errors="coerce")

    numeric_cols = ["stress_score", "hour", "is_weekend", "pu_location_id"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "borough" in out.columns:
        out["borough"] = out["borough"].astype("string")

    out = out.dropna(subset=[time_col, target_reg_col, target_clf_col]).copy()
    out[target_clf_col] = out[target_clf_col].astype(int)
    out = out.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    return out


def temporal_split(
    df: pd.DataFrame,
    *,
    time_col: str,
    train_frac: float,
    val_frac: float,
    gap_steps: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, dict[str, str | None]]:
    if train_frac <= 0 or train_frac >= 1:
        raise ValueError("train_frac debe estar entre 0 y 1.")
    if val_frac < 0 or val_frac >= 1:
        raise ValueError("val_frac debe estar entre 0 y 1 (puede ser 0).")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac debe ser menor que 1.")
    if gap_steps < 0:
        raise ValueError("gap_steps no puede ser negativo.")

    unique_ts = df[time_col].drop_duplicates().sort_values().reset_index(drop=True)
    n_steps = len(unique_ts)
    if n_steps < 3:
        raise ValueError(
            "Muy pocos timestamps unicos para hacer split temporal fiable. "
            f"Timestamps disponibles: {n_steps}"
        )

    n_train = int(n_steps * train_frac)
    n_val = int(n_steps * val_frac)
    has_val = val_frac > 0
    gap_slots = gap_steps * (2 if has_val else 1)
    n_test = n_steps - n_train - n_val - gap_slots

    if n_train <= 0:
        raise ValueError("El bloque train queda vacio. Aumenta train_frac.")
    if has_val and n_val <= 0:
        raise ValueError(
            "val_frac > 0 pero el bloque de validacion queda vacio. "
            "Aumenta val_frac o desactivalo con val_frac=0."
        )
    if n_test <= 0:
        raise ValueError(
            "El bloque test queda vacio tras aplicar fracciones y gap_steps. "
            "Reduce gap_steps o ajusta train_frac/val_frac."
        )

    train_end_idx = n_train - 1
    train_end_ts = unique_ts.iloc[train_end_idx]
    train_df = df[df[time_col] <= train_end_ts].copy()

    if has_val:
        val_start_idx = train_end_idx + 1 + gap_steps
        val_end_idx = val_start_idx + n_val - 1
        test_start_idx = val_end_idx + 1 + gap_steps

        val_start_ts = unique_ts.iloc[val_start_idx]
        val_end_ts = unique_ts.iloc[val_end_idx]
        test_start_ts = unique_ts.iloc[test_start_idx]

        val_df = df[(df[time_col] >= val_start_ts) & (df[time_col] <= val_end_ts)].copy()
        test_df = df[df[time_col] >= test_start_ts].copy()
    else:
        val_start_ts = None
        val_end_ts = None
        test_start_idx = train_end_idx + 1 + gap_steps
        test_start_ts = unique_ts.iloc[test_start_idx]

        val_df = None
        test_df = df[df[time_col] >= test_start_ts].copy()

    if train_df.empty:
        raise ValueError("El bloque train quedo vacio.")
    if test_df.empty:
        raise ValueError("El bloque test quedo vacio.")
    if has_val and val_df is not None and val_df.empty:
        raise ValueError("El bloque val quedo vacio.")

    bounds = {
        "train_end": str(train_end_ts),
        "val_start": None if val_start_ts is None else str(val_start_ts),
        "val_end": None if val_end_ts is None else str(val_end_ts),
        "test_start": str(test_start_ts),
    }
    return train_df, val_df, test_df, bounds


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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


class RegressionBaseline:
    name = "baseline"
    required_columns: tuple[str, ...] = ()

    def __init__(self, *, target_reg_col: str, threshold_quantile: float) -> None:
        self.target_reg_col = target_reg_col
        self.threshold_quantile = threshold_quantile

    def fit(self, train: pd.DataFrame) -> None:
        target = pd.to_numeric(train[self.target_reg_col], errors="coerce")
        self.global_mean_ = float(target.mean())
        self.threshold_ = float(target.quantile(self.threshold_quantile))
        self._fit(train)

    def _fit(self, train: pd.DataFrame) -> None:
        return None

    def predict_regression(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_classification(self, df: pd.DataFrame) -> np.ndarray:
        scores = self.predict_regression(df)
        return (scores >= self.threshold_).astype(int)


class GlobalMeanBaseline(RegressionBaseline):
    name = "global_mean"

    def predict_regression(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.global_mean_, dtype=float)


class HourWeekendMeanBaseline(RegressionBaseline):
    name = "hour_weekend_mean"
    required_columns = ("hour", "is_weekend")
    group_cols = ["hour", "is_weekend"]

    def _fit(self, train: pd.DataFrame) -> None:
        self.group_means_ = (
            train.groupby(self.group_cols, as_index=False, dropna=False)[self.target_reg_col]
            .mean()
            .rename(columns={self.target_reg_col: "_pred"})
        )

    def predict_regression(self, df: pd.DataFrame) -> np.ndarray:
        merged = df[self.group_cols].merge(self.group_means_, on=self.group_cols, how="left")
        return merged["_pred"].fillna(self.global_mean_).to_numpy(dtype=float)


class ZoneHourWeekendMeanBaseline(RegressionBaseline):
    name = "zone_hour_weekend_mean"
    required_columns = ("pu_location_id", "hour", "is_weekend")
    zone_group_cols = ["pu_location_id", "hour", "is_weekend"]
    borough_group_cols = ["borough", "hour", "is_weekend"]
    fallback_group_cols = ["hour", "is_weekend"]

    def _fit(self, train: pd.DataFrame) -> None:
        self.zone_means_ = (
            train.groupby(self.zone_group_cols, as_index=False, dropna=False)[self.target_reg_col]
            .mean()
            .rename(columns={self.target_reg_col: "_pred"})
        )
        self.fallback_means_ = (
            train.groupby(self.fallback_group_cols, as_index=False, dropna=False)[self.target_reg_col]
            .mean()
            .rename(columns={self.target_reg_col: "_pred"})
        )
        self.use_borough_fallback_ = "borough" in train.columns
        if self.use_borough_fallback_:
            self.borough_means_ = (
                train.groupby(self.borough_group_cols, as_index=False, dropna=False)[self.target_reg_col]
                .mean()
                .rename(columns={self.target_reg_col: "_pred"})
            )

    def predict_regression(self, df: pd.DataFrame) -> np.ndarray:
        zone_pred = (
            df[self.zone_group_cols]
            .merge(self.zone_means_, on=self.zone_group_cols, how="left")["_pred"]
            .astype(float)
        )
        fallback_pred = (
            df[self.fallback_group_cols]
            .merge(self.fallback_means_, on=self.fallback_group_cols, how="left")["_pred"]
            .astype(float)
        )

        preds = zone_pred
        if self.use_borough_fallback_ and "borough" in df.columns:
            borough_pred = (
                df[self.borough_group_cols]
                .merge(self.borough_means_, on=self.borough_group_cols, how="left")["_pred"]
                .astype(float)
            )
            preds = preds.fillna(borough_pred)

        preds = preds.fillna(fallback_pred).fillna(self.global_mean_)
        return preds.to_numpy(dtype=float)


class PersistenceBaseline(RegressionBaseline):
    name = "persistence_current_stress"
    required_columns = ("stress_score",)

    def predict_regression(self, df: pd.DataFrame) -> np.ndarray:
        current = pd.to_numeric(df["stress_score"], errors="coerce").fillna(self.global_mean_)
        return current.to_numpy(dtype=float)


def build_baselines(
    available_columns: set[str],
    *,
    target_reg_col: str,
    threshold_quantile: float,
) -> list[RegressionBaseline]:
    baseline_classes = [
        PersistenceBaseline,
        ZoneHourWeekendMeanBaseline,
        GlobalMeanBaseline,
    ]

    baselines: list[RegressionBaseline] = []
    for baseline_cls in baseline_classes:
        if all(col in available_columns for col in baseline_cls.required_columns):
            baselines.append(
                baseline_cls(
                    target_reg_col=target_reg_col,
                    threshold_quantile=threshold_quantile,
                )
            )
    return baselines


def evaluate_baseline(
    model: RegressionBaseline,
    df: pd.DataFrame,
    *,
    split_name: str,
    target_reg_col: str,
    target_clf_col: str,
) -> dict[str, Any]:
    y_true_reg = df[target_reg_col].to_numpy(dtype=float)
    y_true_clf = df[target_clf_col].to_numpy(dtype=int)

    y_pred_reg = model.predict_regression(df)
    y_pred_clf = model.predict_classification(df)

    return {
        "split": split_name,
        "n_rows": int(len(df)),
        "regression": regression_metrics(y_true_reg, y_pred_reg),
        "classification": classification_metrics(y_true_clf, y_pred_clf),
    }


def print_summary(report: dict[str, Any]) -> None:
    table = Table(title=f"Baseline EX2 - {report['target_regression']}", header_style="bold cyan")
    table.add_column("Baseline", style="bold white")
    table.add_column("Split")
    table.add_column("MAE", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_column("R2", justify="right")
    table.add_column("Acc", justify="right")
    table.add_column("Prec", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")

    for baseline_name in report["baseline_order"]:
        split_results = report["baselines"][baseline_name]
        for split_name, metrics in split_results.items():
            reg = metrics["regression"]
            clf = metrics["classification"]
            table.add_row(
                baseline_name,
                split_name,
                f"{reg['mae']:.4f}",
                f"{reg['rmse']:.4f}",
                f"{reg['r2']:.4f}",
                f"{clf['accuracy']:.4f}",
                f"{clf['precision']:.4f}",
                f"{clf['recall']:.4f}",
                f"{clf['f1']:.4f}",
            )

    console.print(table)
    console.print(
        "[dim]Roles recomendados:[/dim] "
        "naive principal=persistence_current_stress | "
        "referencia fuerte=zone_hour_weekend_mean | "
        "piso de control=global_mean"
    )


def run_baselines(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    outputs_dir: str = DEFAULT_OUTPUTS_DIR,
    target_col: str = DEFAULT_TARGET_REG,
    target_clf_col: str | None = None,
    time_col: str = DEFAULT_TIME_COL,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    gap_steps: int = 0,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
) -> dict[str, Any]:
    print_stage("ML BASELINE (EJ2)", "Referencias naive sobre el dataset actual de estres", color="yellow")

    resolved_target_clf_col = target_clf_col or infer_classification_target(target_col)
    required_columns = [time_col, target_col, resolved_target_clf_col]
    optional_columns = [
        "stress_score",
        "hour",
        "is_weekend",
        "pu_location_id",
        "borough",
    ]

    raw_df, dataset_path = read_partitioned_parquet(
        dataset_dir=dataset_dir,
        required_columns=required_columns,
        optional_columns=optional_columns,
    )
    df = prepare_dataset(
        raw_df,
        time_col=time_col,
        target_reg_col=target_col,
        target_clf_col=resolved_target_clf_col,
    )

    train_df, val_df, test_df, bounds = temporal_split(
        df,
        time_col=time_col,
        train_frac=train_frac,
        val_frac=val_frac,
        gap_steps=gap_steps,
    )

    console.print(
        f"[cyan]Dataset cargado[/cyan] -> {dataset_path} | "
        f"filas={len(df):,} | train={len(train_df):,} | "
        f"val={0 if val_df is None else len(val_df):,} | test={len(test_df):,}"
    )

    baselines = build_baselines(
        set(df.columns),
        target_reg_col=target_col,
        threshold_quantile=threshold_quantile,
    )
    if not baselines:
        raise ValueError("No se pudo construir ningun baseline con las columnas disponibles.")

    eval_splits: list[tuple[str, pd.DataFrame]] = [("train", train_df)]
    if val_df is not None:
        eval_splits.append(("val", val_df))
    eval_splits.append(("test", test_df))

    report: dict[str, Any] = {
        "model": "baseline",
        "task": "ej2_stress_urban",
        "dataset_dir": str(dataset_path),
        "target_regression": target_col,
        "target_classification": resolved_target_clf_col,
        "time_col": time_col,
        "baseline_order": [baseline.name for baseline in baselines],
        "baseline_roles": {
            baseline.name: BASELINE_ROLE_MAP.get(baseline.name, "baseline")
            for baseline in baselines
        },
        "splits": {
            "train": int(len(train_df)),
            "val": 0 if val_df is None else int(len(val_df)),
            "test": int(len(test_df)),
            "train_frac": float(train_frac),
            "val_frac": float(val_frac),
            "gap_steps": int(gap_steps),
            "bounds": bounds,
        },
        "train_stats": {
            "target_mean": round(float(train_df[target_col].mean()), 4),
            "target_median": round(float(train_df[target_col].median()), 4),
            "target_std": round(float(train_df[target_col].std()), 4),
            "threshold_quantile": float(threshold_quantile),
            "target_threshold_value": round(float(train_df[target_col].quantile(threshold_quantile)), 4),
            "positive_rate": round(float(train_df[resolved_target_clf_col].mean()), 4),
        },
        "baselines": {},
    }

    for baseline in baselines:
        baseline.fit(train_df)
        results_by_split: dict[str, Any] = {}
        for split_name, split_df in eval_splits:
            results_by_split[split_name] = evaluate_baseline(
                baseline,
                split_df,
                split_name=split_name,
                target_reg_col=target_col,
                target_clf_col=resolved_target_clf_col,
            )
        report["baselines"][baseline.name] = results_by_split

    outputs_path = ensure_project_dir(outputs_dir)
    report_path = outputs_path / f"baseline_report_{target_col}.json"
    save_json(report, report_path)

    print_summary(report)
    console.print(f"[green]Reporte guardado[/green] -> {report_path}")
    print_done("BASELINE EX2 COMPLETADO")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evalua baselines para EX2 sobre el dataset actual de estres urbano."
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help="Dataset particionado de EX2.",
    )
    parser.add_argument(
        "--outputs-dir",
        default=DEFAULT_OUTPUTS_DIR,
        help="Directorio de salida para reportes.",
    )
    parser.add_argument(
        "--target-col",
        default=DEFAULT_TARGET_REG,
        choices=["target_stress_t1", "target_stress_t3", "target_stress_t24"],
        help="Target continuo a predecir.",
    )
    parser.add_argument(
        "--target-clf-col",
        default=None,
        help="Target binario. Si no se pasa, se infiere desde --target-col.",
    )
    parser.add_argument(
        "--time-col",
        default=DEFAULT_TIME_COL,
        help="Columna temporal usada para el split.",
    )
    parser.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC)
    parser.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    parser.add_argument("--gap-steps", type=int, default=0)
    parser.add_argument("--threshold-quantile", type=float, default=DEFAULT_THRESHOLD_QUANTILE)

    args = parser.parse_args()

    run_baselines(
        dataset_dir=args.dataset_dir,
        outputs_dir=args.outputs_dir,
        target_col=args.target_col,
        target_clf_col=args.target_clf_col,
        time_col=args.time_col,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        gap_steps=args.gap_steps,
        threshold_quantile=args.threshold_quantile,
    )


if __name__ == "__main__":
    main()
