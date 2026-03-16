from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from src.pipeline_runner import console
from src.ml.dataset.modules.display import print_step_status
from src.ml.dataset.modules.io import read_partitioned_parquet_dir


def add_business_metrics(
    df: pd.DataFrame,
    project_root: Path,
    variability_dir: str,
) -> pd.DataFrame:
    """
    Añade:
    - price_variability
    - price_variability_abs
    - price_variability_rel
    - biz_score_iqr
    - biz_score
    - biz_score_zsum
    - biz_score_zproduct
    - stress_score
    - is_stress
    """
    var_base = project_root / variability_dir
    has_iqr = False

    if var_base.exists():
        try:
            var_df = read_partitioned_parquet_dir(var_base)
            if "price_variability" in var_df.columns:
                var_df["pu_location_id"] = pd.to_numeric(var_df["pu_location_id"], errors="coerce").astype("Int64")
                var_df["hour"] = pd.to_numeric(var_df["hour"], errors="coerce").astype("Int64")
                var_df["service_type"] = var_df["service_type"].astype(str)

                iqr_map = var_df[
                    ["pu_location_id", "hour", "service_type", "price_variability"]
                ].drop_duplicates()

                df = df.merge(iqr_map, on=["pu_location_id", "hour", "service_type"], how="left")
                has_iqr = True
                print_step_status(
                    "IQR join",
                    f"{df['price_variability'].notna().sum():,}/{len(df):,} filas con IQR de capa3",
                )
        except Exception as exc:
            console.print(f"[bold yellow]WARNING[/bold yellow] No se pudo leer df_variability: {exc}")

    if has_iqr:
        df["price_variability"] = df["price_variability"].fillna(df["std_price"].fillna(0))
        variability = df["price_variability"].astype(float)
    else:
        variability = df["std_price"].fillna(0).astype(float)
        df["price_variability"] = variability
        console.print("[bold yellow]WARNING[/bold yellow] IQR no disponible: usando std_price como proxy de variabilidad")

    df["price_variability_abs"] = variability

    avg_price_safe = df["avg_price"].replace(0, pd.NA)
    df["price_variability_rel"] = (variability / avg_price_safe).fillna(0).astype(float)

    log_volume = df["num_trips"].fillna(0).astype(float).apply(lambda x: math.log1p(x))

    df["biz_score_iqr"] = variability
    df["biz_score"] = variability * log_volume

    v_mean, v_std = float(variability.mean()), float(variability.std())
    lv_mean, lv_std = float(log_volume.mean()), float(log_volume.std())

    z_var = (variability - v_mean) / v_std if v_std > 0 else 0.0
    z_lv = (log_volume - lv_mean) / lv_std if lv_std > 0 else 0.0

    df["biz_score_zsum"] = z_var + z_lv
    df["biz_score_zproduct"] = z_var * z_lv

    df["stress_score"] = df["biz_score"]
    thr = df["stress_score"].quantile(0.90)
    df["is_stress"] = (df["stress_score"] >= thr).astype(int)

    print_step_status(
        "Métricas",
        f"calculadas | threshold p90={thr:.4f} | is_stress=1: {df['is_stress'].sum():,} filas",
    )
    return df


def reorder_dataset_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred_order = [
        "pu_location_id",
        "borough",
        "hour",
        "date",
        "service_type",
        "num_trips",
        "avg_price",
        "std_price",
        "price_variability",
        "price_variability_abs",
        "price_variability_rel",
        "rain_mm_sum",
        "temp_c_mean",
        "wind_kmh_mean",
        "precip_mm_sum",
        "snowfall_mm_sum",
        "weather_code_mode",
        "event_count_city",
        "event_count_borough",
        "lag_1h_trips",
        "lag_24h_trips",
        "roll_3h_trips",
        "day_of_week",
        "month",
        "is_weekend",
        "is_peak_hour",
        "biz_score_iqr",
        "biz_score",
        "biz_score_zsum",
        "biz_score_zproduct",
        "stress_score",
        "is_stress",
    ]
    existing_cols = [c for c in preferred_order if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in existing_cols]
    return df[existing_cols + remaining_cols]