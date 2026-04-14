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


def split_model_stress(
    df: pd.DataFrame,
    target_col: str = "target_stress_t1",
    time_col: str = "timestamp_hour",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    gap_steps: int = 0,
    drop_cols: list[str] | tuple[str, ...] | None = None,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame | None,
    pd.Series | None,
    pd.DataFrame,
    pd.Series,
]:
    """
    Split temporal + separacion X/y listo para entrenar modelos.

    - Usa `time_col` solo para hacer el corte temporal.
    - Despues elimina de X: `time_col`, `target_col` y `drop_cols`.
    - Devuelve: X_train, y_train, X_val, y_val, X_test, y_test.
      Si `val_frac == 0`, X_val e y_val seran `None`.
    """
    if train_frac <= 0 or train_frac >= 1:
        raise ValueError("train_frac debe estar entre 0 y 1.")
    if val_frac < 0 or val_frac >= 1:
        raise ValueError("val_frac debe estar entre 0 y 1 (puede ser 0).")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac debe ser menor que 1.")
    if gap_steps < 0:
        raise ValueError("gap_steps no puede ser negativo.")

    if time_col not in df.columns:
        raise ValueError(f"El dataset debe incluir la columna temporal '{time_col}'.")
    if target_col not in df.columns:
        raise ValueError(f"El dataset debe incluir la columna target '{target_col}'.")

    base_drop_cols = list(drop_cols) if drop_cols is not None else [
        "date",
        "target_is_stress_t1",
        "is_stress_now",
    ]
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col]).copy()
    out = out.sort_values(time_col, kind="mergesort").reset_index(drop=True)

    unique_ts = (
        out[time_col]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
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
    train_df = out[out[time_col] <= train_end_ts].copy()

    if has_val:
        val_start_idx = train_end_idx + 1 + gap_steps
        val_end_idx = val_start_idx + n_val - 1
        test_start_idx = val_end_idx + 1 + gap_steps

        val_start_ts = unique_ts.iloc[val_start_idx]
        val_end_ts = unique_ts.iloc[val_end_idx]
        test_start_ts = unique_ts.iloc[test_start_idx]

        val_df = out[(out[time_col] >= val_start_ts) & (out[time_col] <= val_end_ts)].copy()
        test_df = out[out[time_col] >= test_start_ts].copy()
    else:
        test_start_idx = train_end_idx + 1 + gap_steps
        test_start_ts = unique_ts.iloc[test_start_idx]
        val_df = None
        test_df = out[out[time_col] >= test_start_ts].copy()

    if train_df.empty or test_df.empty or (has_val and val_df is not None and val_df.empty):
        raise ValueError(
            "El split temporal genero un bloque vacio. "
            "Revisa train_frac/val_frac/gap_steps."
        )

    cols_to_remove = set(base_drop_cols + [time_col, target_col])

    def _to_xy(split_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        y = split_df[target_col].copy()
        x = split_df.drop(columns=[c for c in cols_to_remove if c in split_df.columns]).copy()
        return x, y

    x_train, y_train = _to_xy(train_df)
    x_test, y_test = _to_xy(test_df)

    if val_df is None:
        x_val = None
        y_val = None
    else:
        x_val, y_val = _to_xy(val_df)

    return x_train, y_train, x_val, y_val, x_test, y_test



def split_model_propinas(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    time_col: str = "timestamp_hour",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split temporal por bloques de timestamps unicos.

    A diferencia de un corte por filas, esta version garantiza que todas las
    muestras con el mismo `timestamp_hour` quedan en el mismo bloque
    (train/val/test), algo especialmente importante en datasets a nivel viaje.
    """
    if train_frac <= 0 or train_frac >= 1:
        raise ValueError("train_frac debe estar entre 0 y 1.")
    if val_frac <= 0 or val_frac >= 1:
        raise ValueError("val_frac debe estar entre 0 y 1.")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac debe ser menor que 1.")

    if time_col not in df.columns:
        raise ValueError(f"El dataset debe incluir la columna temporal '{time_col}'.")

    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col]).copy()
    out = out.sort_values(time_col, kind="mergesort").reset_index(drop=True)

    unique_ts = out[time_col].drop_duplicates().sort_values().reset_index(drop=True)
    n_steps = len(unique_ts)
    if n_steps < 10:
        raise ValueError(
            "Hay muy pocas muestras temporales para entrenar de forma fiable. "
            f"Timestamps disponibles: {n_steps}"
        )

    n_train = int(n_steps * train_frac)
    n_val = int(n_steps * val_frac)
    n_test = n_steps - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError("El split temporal ha dejado un bloque vacio. Ajusta train_frac/val_frac.")

    train_end_ts = unique_ts.iloc[n_train - 1]
    val_start_ts = unique_ts.iloc[n_train]
    val_end_ts = unique_ts.iloc[n_train + n_val - 1]
    test_start_ts = unique_ts.iloc[n_train + n_val]

    train = out[out[time_col] <= train_end_ts].copy()
    val = out[(out[time_col] >= val_start_ts) & (out[time_col] <= val_end_ts)].copy()
    test = out[out[time_col] >= test_start_ts].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError("El split temporal ha dejado un bloque vacio. Ajusta train_frac/val_frac.")

    return train, val, test