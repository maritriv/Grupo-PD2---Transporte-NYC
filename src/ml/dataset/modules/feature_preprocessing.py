from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from rich.table import Table

from config.pipeline_runner import console, print_done, print_stage
from src.ml.dataset.modules.model_feature_sets import (
    FeatureMode,
    TARGET_CLF,
    TARGET_REG,
    get_forced_drop_columns,
)


def _split_xy(df: pd.DataFrame, target_reg: str, target_clf: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    for c in [target_reg, target_clf]:
        if c not in df.columns:
            raise ValueError(f"Falta target '{c}' en dataset.")
    y = df[[target_reg, target_clf]].copy()
    x = df.drop(columns=[target_reg, target_clf]).copy()
    return x, y


def _align_columns(base_cols: list[str], df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(columns=base_cols, fill_value=0)


def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _is_binary_series(s: pd.Series) -> bool:
    vals = s.dropna().unique()
    if len(vals) == 0:
        return True
    return set(vals).issubset({0, 1})


def _top_nulls_table(null_ratio: pd.Series, limit: int = 10) -> Table:
    table = Table(title="Top columnas por nulos (train)", header_style="bold yellow")
    table.add_column("Columna", style="bold white")
    table.add_column("Null ratio", justify="right")
    for col, ratio in null_ratio.sort_values(ascending=False).head(limit).items():
        table.add_row(str(col), f"{float(ratio):.4f}")
    return table


def _correlation_pairs(
    df: pd.DataFrame,
    threshold: float,
) -> tuple[list[str], list[dict[str, float | str]]]:
    corr_num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not corr_num_cols:
        return [], []

    corr = df[corr_num_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    corr_drop: list[str] = []
    corr_pairs: list[dict[str, float | str]] = []

    for col in upper.columns:
        high_corr = upper[col][upper[col] > threshold]
        if not high_corr.empty:
            corr_drop.append(col)
            for other_col, corr_value in high_corr.sort_values(ascending=False).items():
                corr_pairs.append(
                    {
                        "feature_kept": str(other_col),
                        "feature_dropped": str(col),
                        "corr_abs": float(corr_value),
                    }
                )

    corr_drop = sorted(set(corr_drop))
    corr_pairs = sorted(corr_pairs, key=lambda x: float(x["corr_abs"]), reverse=True)
    return corr_drop, corr_pairs


def _correlation_table(corr_pairs: list[dict[str, float | str]], limit: int = 15) -> Table:
    table = Table(title="Top pares con alta correlación", header_style="bold cyan")
    table.add_column("Feature kept", style="bold white")
    table.add_column("Feature dropped", style="bold white")
    table.add_column("|corr|", justify="right")

    for row in corr_pairs[:limit]:
        table.add_row(
            str(row["feature_kept"]),
            str(row["feature_dropped"]),
            f"{float(row['corr_abs']):.4f}",
        )
    return table


def _outlier_clip_table(
    clip_bounds: Dict[str, Tuple[float, float]],
    limit: int = 12,
) -> Table:
    table = Table(title="Clip de outliers (train)", header_style="bold magenta")
    table.add_column("Columna", style="bold white")
    table.add_column("Low", justify="right")
    table.add_column("High", justify="right")

    shown = 0
    for col, (low, high) in clip_bounds.items():
        table.add_row(str(col), f"{low:.4f}", f"{high:.4f}")
        shown += 1
        if shown >= limit:
            break

    return table


def feature_preprocessing(
    splits_dir: str = "data/ml/splits",
    prefix: str = "completo",
    out_dir: str = "data/ml/splits_processed",
    mode: FeatureMode = "operational",
    null_threshold: float = 0.70,
    corr_threshold: float = 0.90,
    onehot_max_levels: int = 20,
    outlier_low_q: float = 0.01,
    outlier_high_q: float = 0.99,
) -> Dict[str, str]:
    print_stage("ML FEATURE PREPROCESSING", f"Prefix: {prefix} | Mode: {mode}", color="magenta")
    project_root = Path(__file__).resolve().parents[4]
    in_base = (project_root / splits_dir).resolve()
    out_base = (project_root / out_dir).resolve()
    os.makedirs(out_base, exist_ok=True)

    train_fp = in_base / f"{prefix}_train.parquet"
    val_fp = in_base / f"{prefix}_val.parquet"
    test_fp = in_base / f"{prefix}_test.parquet"

    if not train_fp.exists() or not val_fp.exists() or not test_fp.exists():
        raise FileNotFoundError(
            "No se encontraron los splits raw esperados. "
            f"Buscados: {train_fp}, {val_fp}, {test_fp}"
        )

    train_df = pd.read_parquet(train_fp)
    val_df = pd.read_parquet(val_fp)
    test_df = pd.read_parquet(test_fp)

    x_train, y_train = _split_xy(train_df, TARGET_REG, TARGET_CLF)
    x_val, y_val = _split_xy(val_df, TARGET_REG, TARGET_CLF)
    x_test, y_test = _split_xy(test_df, TARGET_REG, TARGET_CLF)

    metadata: Dict[str, object] = {"mode": mode}

    # 1) Drop columnas forzadas según modo
    forced_drop = [c for c in get_forced_drop_columns(mode) if c in x_train.columns]
    if forced_drop:
        x_train = x_train.drop(columns=forced_drop)
        x_val = x_val.drop(columns=[c for c in forced_drop if c in x_val.columns])
        x_test = x_test.drop(columns=[c for c in forced_drop if c in x_test.columns])
    metadata["forced_drop_columns"] = forced_drop

    # 2) Drop por nulos (fit solo train)
    null_ratio = x_train.isna().mean()
    high_null_cols = sorted(null_ratio[null_ratio > null_threshold].index.tolist())
    if high_null_cols:
        x_train = x_train.drop(columns=high_null_cols)
        x_val = x_val.drop(columns=[c for c in high_null_cols if c in x_val.columns])
        x_test = x_test.drop(columns=[c for c in high_null_cols if c in x_test.columns])

    metadata["null_threshold"] = null_threshold
    metadata["high_null_columns_dropped"] = high_null_cols
    metadata["top_null_ratios_train"] = {
        str(k): float(v) for k, v in null_ratio.sort_values(ascending=False).head(20).items()
    }

    console.print(_top_nulls_table(null_ratio))

    # 3) Categóricas
    cat_cols = x_train.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    low_card_cols = []
    high_card_cols = []

    for c in cat_cols:
        n_levels = x_train[c].astype("string").nunique(dropna=True)
        if n_levels <= onehot_max_levels:
            low_card_cols.append(c)
        else:
            high_card_cols.append(c)

    high_card_maps: Dict[str, Dict[str, float]] = {}
    for c in high_card_cols:
        tr = x_train[c].astype("string").fillna("__MISSING__")
        freq_map = tr.value_counts(normalize=True).to_dict()
        high_card_maps[c] = {str(k): float(v) for k, v in freq_map.items()}
        new_c = f"{c}__freq"
        x_train[new_c] = tr.map(freq_map).fillna(0.0)
        x_val[new_c] = x_val[c].astype("string").fillna("__MISSING__").map(freq_map).fillna(0.0)
        x_test[new_c] = x_test[c].astype("string").fillna("__MISSING__").map(freq_map).fillna(0.0)

    if high_card_cols:
        x_train = x_train.drop(columns=high_card_cols)
        x_val = x_val.drop(columns=[c for c in high_card_cols if c in x_val.columns])
        x_test = x_test.drop(columns=[c for c in high_card_cols if c in x_test.columns])

    if low_card_cols:
        tr_cat = x_train[low_card_cols].astype("string").fillna("__MISSING__")
        va_cat = x_val[low_card_cols].astype("string").fillna("__MISSING__")
        te_cat = x_test[low_card_cols].astype("string").fillna("__MISSING__")

        tr_dum = pd.get_dummies(tr_cat, prefix=low_card_cols, prefix_sep="__")
        va_dum = pd.get_dummies(va_cat, prefix=low_card_cols, prefix_sep="__")
        te_dum = pd.get_dummies(te_cat, prefix=low_card_cols, prefix_sep="__")

        va_dum = _align_columns(tr_dum.columns.tolist(), va_dum)
        te_dum = _align_columns(tr_dum.columns.tolist(), te_dum)

        x_train = x_train.drop(columns=low_card_cols).join(tr_dum)
        x_val = x_val.drop(columns=low_card_cols).join(va_dum)
        x_test = x_test.drop(columns=low_card_cols).join(te_dum)

    metadata["onehot_max_levels"] = onehot_max_levels
    metadata["categorical_low_card_onehot"] = low_card_cols
    metadata["categorical_high_card_freq"] = high_card_cols

    # 4) Numéricas -> imputación
    num_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    x_train = _safe_numeric(x_train, num_cols)
    x_val = _safe_numeric(x_val, num_cols)
    x_test = _safe_numeric(x_test, num_cols)

    medians = x_train[num_cols].median(numeric_only=True)
    x_train[num_cols] = x_train[num_cols].fillna(medians)
    x_val[num_cols] = x_val[num_cols].fillna(medians)
    x_test[num_cols] = x_test[num_cols].fillna(medians)

    metadata["imputation_medians"] = {str(k): float(v) for k, v in medians.items()}

    # 5) Outliers
    clip_bounds: Dict[str, Tuple[float, float]] = {}
    for c in num_cols:
        low = float(x_train[c].quantile(outlier_low_q))
        high = float(x_train[c].quantile(outlier_high_q))
        if low > high:
            low, high = high, low
        clip_bounds[c] = (low, high)
        x_train[c] = x_train[c].clip(lower=low, upper=high)
        x_val[c] = x_val[c].clip(lower=low, upper=high)
        x_test[c] = x_test[c].clip(lower=low, upper=high)

    metadata["outlier_low_q"] = outlier_low_q
    metadata["outlier_high_q"] = outlier_high_q
    metadata["clip_bounds"] = {k: [float(v[0]), float(v[1])] for k, v in clip_bounds.items()}

    console.print(_outlier_clip_table(clip_bounds))

    # 6) Correlación
    corr_drop, corr_pairs = _correlation_pairs(x_train, corr_threshold)
    if corr_drop:
        x_train = x_train.drop(columns=corr_drop)
        x_val = x_val.drop(columns=[c for c in corr_drop if c in x_val.columns])
        x_test = x_test.drop(columns=[c for c in corr_drop if c in x_test.columns])

    metadata["corr_threshold"] = corr_threshold
    metadata["correlation_drop_columns"] = corr_drop
    metadata["correlation_pairs_above_threshold"] = corr_pairs
    metadata["n_correlation_drop_columns"] = len(corr_drop)

    if corr_pairs:
        console.print(_correlation_table(corr_pairs))

    # 7) Escalado
    scale_cols = []
    for c in x_train.select_dtypes(include=[np.number]).columns:
        if not _is_binary_series(x_train[c]):
            scale_cols.append(c)

    means = x_train[scale_cols].mean()
    stds = x_train[scale_cols].std().replace(0, 1.0)

    x_train[scale_cols] = (x_train[scale_cols] - means) / stds
    x_val[scale_cols] = (x_val[scale_cols] - means) / stds
    x_test[scale_cols] = (x_test[scale_cols] - means) / stds

    metadata["scale_columns"] = scale_cols

    # Alinear columnas
    final_cols = x_train.columns.tolist()
    x_val = _align_columns(final_cols, x_val)
    x_test = _align_columns(final_cols, x_test)

    train_out = pd.concat([x_train, y_train], axis=1)
    val_out = pd.concat([x_val, y_val], axis=1)
    test_out = pd.concat([x_test, y_test], axis=1)

    # Salidas con sufijo por modo
    out_train_fp = out_base / f"{prefix}_{mode}_train.parquet"
    out_val_fp = out_base / f"{prefix}_{mode}_val.parquet"
    out_test_fp = out_base / f"{prefix}_{mode}_test.parquet"

    train_out.to_parquet(out_train_fp, index=False)
    val_out.to_parquet(out_val_fp, index=False)
    test_out.to_parquet(out_test_fp, index=False)

    metadata.update(
        {
            "prefix": prefix,
            "feature_count": len(final_cols),
            "rows": {
                "train": len(train_out),
                "val": len(val_out),
                "test": len(test_out),
            },
            "files": {
                "train": str(out_train_fp),
                "val": str(out_val_fp),
                "test": str(out_test_fp),
            },
        }
    )

    meta_fp = out_base / f"{prefix}_{mode}_feature_preprocessing_meta.json"
    with meta_fp.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    out = {
        "train": str(out_train_fp),
        "val": str(out_val_fp),
        "test": str(out_test_fp),
        "meta": str(meta_fp),
    }

    table = Table(title=f"Feature preprocessing ({prefix} | {mode})", header_style="bold magenta")
    table.add_column("Metrica", style="bold white")
    table.add_column("Valor")
    table.add_row("Rows train", f"{len(train_out):,}")
    table.add_row("Rows val", f"{len(val_out):,}")
    table.add_row("Rows test", f"{len(test_out):,}")
    table.add_row("Features finales", f"{len(final_cols):,}")
    table.add_row("Cols drop forzadas", f"{len(forced_drop):,}")
    table.add_row("Cols drop nulos", f"{len(high_null_cols):,}")
    table.add_row("Cols drop correlación", f"{len(corr_drop):,}")
    table.add_row("Train", out["train"])
    table.add_row("Val", out["val"])
    table.add_row("Test", out["test"])
    table.add_row("Meta", out["meta"])
    console.print(table)

    print_done(f"FEATURE PREPROCESSING COMPLETADO ({mode})")
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Preprocesa splits ML sin leakage (fit solo en train): nulos, outliers, OHE, correlación y escalado."
    )
    p.add_argument("--splits-dir", default="data/ml/splits", help="Carpeta de splits raw.")
    p.add_argument("--prefix", default="completo", help="Prefijo de splits.")
    p.add_argument("--out-dir", default="data/ml/splits_processed", help="Salida de splits procesados.")
    p.add_argument("--mode", choices=["operational", "predictive"], default="operational", help="Modo de features.")
    p.add_argument("--null-threshold", type=float, default=0.70, help="Eliminar columnas con ratio de nulos > umbral.")
    p.add_argument("--corr-threshold", type=float, default=0.90, help="Umbral de correlación absoluta para eliminar colinealidad.")
    p.add_argument("--onehot-max-levels", type=int, default=20, help="Máximo niveles para OHE (si supera, usa freq encoding).")
    p.add_argument("--outlier-low-q", type=float, default=0.01, help="Percentil inferior para clipping.")
    p.add_argument("--outlier-high-q", type=float, default=0.99, help="Percentil superior para clipping.")
    args = p.parse_args()

    feature_preprocessing(
        splits_dir=args.splits_dir,
        prefix=args.prefix,
        out_dir=args.out_dir,
        mode=args.mode,
        null_threshold=args.null_threshold,
        corr_threshold=args.corr_threshold,
        onehot_max_levels=args.onehot_max_levels,
        outlier_low_q=args.outlier_low_q,
        outlier_high_q=args.outlier_high_q,
    )


if __name__ == "__main__":
    main()