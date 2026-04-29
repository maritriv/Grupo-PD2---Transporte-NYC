from __future__ import annotations

import pandas as pd

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F


def split_model_stress_spark(
    df: SparkDataFrame,
    target_col: str = "target_stress_t1",
    time_col: str = "timestamp_hour",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    gap_steps: int = 0,
    extra_drop_cols: list[str] | tuple[str, ...] | None = None,
    label_col: str = "label",
    return_bounds: bool = False,
) -> tuple[SparkDataFrame, SparkDataFrame, SparkDataFrame] | tuple[
    SparkDataFrame, SparkDataFrame, SparkDataFrame, dict[str, str | None]
]:
    """
    Split temporal en Spark replicando la logica de split_model_stress (pandas).

    - Usa `time_col` solo para hacer el corte temporal.
    - Respeta `gap_steps` entre train/val y val/test.
    - Elimina de salida: `time_col`, columnas base y `extra_drop_cols`.
    - Mantiene `target_col` en salida para facilitar analisis/experimentacion.
    - Añade `label_col` a partir de `target_col` casteado a double.
    - Devuelve:
        * con val_frac > 0: train_df, val_df, test_df
        * con val_frac == 0: train_df, val_df_vacio, test_df
      Si `return_bounds=True`, agrega un dict con limites temporales.
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

    # Columnas comunes a eliminar en cualquier configuración.
    base_drop_cols_all = [
        "date",
        "is_stress_now",
        "target_stress_t1",
        "target_stress_t3",
        "target_stress_t24",
        "target_is_stress_t1",
        "target_is_stress_t3",
        "target_is_stress_t24",
    ]
    # Conservar la target elegida y eliminar el resto de targets.
    base_drop_cols = [c for c in base_drop_cols_all if c != target_col]
    extra = list(extra_drop_cols) if extra_drop_cols is not None else []
    drop_cols = list(dict.fromkeys(base_drop_cols + extra + [time_col]))

    # 1) Limpiar y normalizar la columna temporal como hace pandas
    out = (
        df.withColumn(time_col, F.to_timestamp(F.col(time_col))).filter(F.col(time_col).isNotNull())
    )

    # 2) Obtener timestamps unicos ordenados
    ts_pd = out.select(time_col).distinct().orderBy(time_col).toPandas()

    if ts_pd.empty:
        raise ValueError("No hay timestamps disponibles para el split.")

    unique_ts = ts_pd[time_col].reset_index(drop=True)
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

    # 3) Calcular bounds exactamente igual que en pandas
    train_end_idx = n_train - 1
    train_end_ts = pd.Timestamp(unique_ts.iloc[train_end_idx])

    if has_val:
        val_start_idx = train_end_idx + 1 + gap_steps
        val_end_idx = val_start_idx + n_val - 1
        test_start_idx = val_end_idx + 1 + gap_steps

        val_start_ts = pd.Timestamp(unique_ts.iloc[val_start_idx])
        val_end_ts = pd.Timestamp(unique_ts.iloc[val_end_idx])
        test_start_ts = pd.Timestamp(unique_ts.iloc[test_start_idx])

        train_df = out.filter(F.col(time_col) <= F.lit(train_end_ts))
        val_df = out.filter((F.col(time_col) >= F.lit(val_start_ts)) & (F.col(time_col) <= F.lit(val_end_ts)))
        test_df = out.filter(F.col(time_col) >= F.lit(test_start_ts))
    else:
        test_start_idx = train_end_idx + 1 + gap_steps
        test_start_ts = pd.Timestamp(unique_ts.iloc[test_start_idx])

        val_start_ts = None
        val_end_ts = None

        train_df = out.filter(F.col(time_col) <= F.lit(train_end_ts))
        val_df = out.limit(0)
        test_df = out.filter(F.col(time_col) >= F.lit(test_start_ts))

    # 4) Validar que no haya bloques vacios, igual que en pandas
    if train_df.limit(1).count() == 0:
        raise ValueError("El bloque train quedo vacio.")
    if test_df.limit(1).count() == 0:
        raise ValueError("El bloque test quedo vacio.")
    if has_val and val_df.limit(1).count() == 0:
        raise ValueError("El bloque val quedo vacio.")

    # 5) Preparar salida: primero crear label y luego dropear columnas
    def _prep(split_df: SparkDataFrame) -> SparkDataFrame:
        out_df = split_df.withColumn(label_col, F.col(target_col).cast("double"))
        cols_to_drop = [c for c in drop_cols if c in out_df.columns]
        if cols_to_drop:
            out_df = out_df.drop(*cols_to_drop)
        return out_df

    train_df = _prep(train_df)
    val_df = _prep(val_df)
    test_df = _prep(test_df)

    if return_bounds:
        bounds = {
            "train_end": str(train_end_ts),
            "val_start": None if val_start_ts is None else str(val_start_ts),
            "val_end": None if val_end_ts is None else str(val_end_ts),
            "test_start": str(test_start_ts),
        }
        return train_df, val_df, test_df, bounds

    return train_df, val_df, test_df
