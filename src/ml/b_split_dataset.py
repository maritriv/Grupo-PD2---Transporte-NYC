from __future__ import annotations

"""
Script: split_dataset.py

Este script genera los splits de entrenamiento para los modelos de ML.

Lee un dataset parquet generado por `build_dataset.py` y crea:
    - train
    - validation
    - test

El split es TEMPORAL (ordenado por date + hour).

-------------------------------------------------------
EJECUCIÓN
-------------------------------------------------------

1. Generar splits del dataset completo (modo normal)

uv run -m src.ml.b_split_dataset

Lee:
    data/ml/dataset_completo.parquet

Genera:
    data/ml/splits/completo_train.parquet
    data/ml/splits/completo_val.parquet
    data/ml/splits/completo_test.parquet


-------------------------------------------------------

2. Generar splits para un dataset de rango

uv run -m src.ml.b_split_dataset \
--input data/ml/dataset_rango_2024-01-01__2024-01-14.parquet \
--prefix rango_2024-01-01__2024-01-14

Genera:
    data/ml/splits/rango_2024-01-01__2024-01-14_train.parquet
    data/ml/splits/rango_2024-01-01__2024-01-14_val.parquet
    data/ml/splits/rango_2024-01-01__2024-01-14_test.parquet


-------------------------------------------------------

3. Cambiar proporciones del split

uv run -m src.ml.b_split_dataset \
--train-frac 0.8 \
--val-frac 0.1

En este caso:
    train = 80%
    val = 10%
    test = 10%

"""

import argparse
import os
from pathlib import Path
import pandas as pd


# -----------------------
# Split temporal
# -----------------------
def split_timewise(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if "date" not in df.columns or "hour" not in df.columns:
        raise ValueError("El dataset debe tener columnas 'date' y 'hour'.")

    tmp = df.copy()

    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp = tmp.dropna(subset=["date"])

    tmp["hour"] = pd.to_numeric(tmp["hour"], errors="coerce").fillna(0).astype(int)

    tmp["ts"] = tmp["date"] + pd.to_timedelta(tmp["hour"], unit="h")

    tmp = tmp.sort_values("ts").drop(columns=["ts"])

    n = len(tmp)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = tmp.iloc[:n_train].copy()
    val = tmp.iloc[n_train:n_train + n_val].copy()
    test = tmp.iloc[n_train + n_val:].copy()

    return train, val, test


# -----------------------
# Crear archivos
# -----------------------
def make_split_outputs(
    input_path: str,
    out_dir: str = "data/ml/splits",
    prefix: str | None = None,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> None:

    in_fp = Path(input_path).resolve()

    if not in_fp.exists():
        raise FileNotFoundError(f"No existe el dataset: {in_fp}")

    df = pd.read_parquet(in_fp)

    if prefix is None:
        prefix = in_fp.stem

    train, val, test = split_timewise(
        df,
        train_frac=train_frac,
        val_frac=val_frac,
    )

    out_base = Path(out_dir).resolve()
    os.makedirs(out_base, exist_ok=True)

    train_fp = out_base / f"{prefix}_train.parquet"
    val_fp = out_base / f"{prefix}_val.parquet"
    test_fp = out_base / f"{prefix}_test.parquet"

    train.to_parquet(train_fp, index=False)
    val.to_parquet(val_fp, index=False)
    test.to_parquet(test_fp, index=False)

    print("\n✅ Splits creados")
    print(f"train -> {train_fp} ({len(train):,} filas)")
    print(f"val   -> {val_fp} ({len(val):,} filas)")
    print(f"test  -> {test_fp} ({len(test):,} filas)")


# -----------------------
# Main
# -----------------------
def main() -> None:

    p = argparse.ArgumentParser(
        description="Crea splits temporales train/val/test."
    )

    p.add_argument(
        "--input",
        default="data/ml/dataset_completo.parquet",
        help="Dataset de entrada (default: dataset_completo.parquet)",
    )

    p.add_argument(
        "--out-dir",
        default="data/ml/splits",
        help="Directorio de salida (default: data/ml/splits)",
    )

    p.add_argument(
        "--prefix",
        default="completo",
        help="Prefijo de los archivos generados (default: completo)",
    )

    p.add_argument(
        "--train-frac",
        type=float,
        default=0.70,
        help="Fracción de entrenamiento (default: 0.70)",
    )

    p.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fracción de validación (default: 0.15)",
    )

    args = p.parse_args()

    make_split_outputs(
        input_path=args.input,
        out_dir=args.out_dir,
        prefix=args.prefix,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )


if __name__ == "__main__":
    main()