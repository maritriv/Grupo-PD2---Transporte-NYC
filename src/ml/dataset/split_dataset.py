from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def split_model_demanda(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_frac <= 0 or train_frac >= 1:
        raise ValueError("train_frac debe estar entre 0 y 1.")
    if val_frac <= 0 or val_frac >= 1:
        raise ValueError("val_frac debe estar entre 0 y 1.")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac debe ser menor que 1.")

    if "timestamp_hour" not in df.columns:
        raise ValueError("El dataset debe incluir la columna 'timestamp_hour'.")

    out = df.copy()
    out["timestamp_hour"] = pd.to_datetime(out["timestamp_hour"], errors="coerce")
    out = out.dropna(subset=["timestamp_hour"])
    out = out.sort_values("timestamp_hour").reset_index(drop=True)

    n = len(out)
    if n < 10:
        raise ValueError(
            "Hay muy pocas muestras horarias para entrenar de forma fiable. "
            f"Muestras disponibles: {n}"
        )

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = out.iloc[:n_train].copy()
    val = out.iloc[n_train:n_train + n_val].copy()
    test = out.iloc[n_train + n_val:].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError("El split temporal ha dejado un bloque vacio. Ajusta train_frac/val_frac.")

    return train, val, test

