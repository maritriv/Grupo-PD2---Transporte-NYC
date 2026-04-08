from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.pipeline_runner import console
from src.ml.dataset.modules.display import print_step_status
from src.ml.dataset.modules.io import read_partitioned_parquet_dir


def join_restaurants(
    df: pd.DataFrame,
    project_root: Path,
    city_day_dir: str = "data/aggregated/restaurants/df_city_day",
    borough_day_dir: str = "data/aggregated/restaurants/df_borough_day",
    borough_hour_day_dir: str = "data/aggregated/restaurants/df_borough_hour_day",
) -> pd.DataFrame:
    out = df.copy()

    city_fp = project_root / city_day_dir
    if city_fp.exists():
        city = read_partitioned_parquet_dir(city_fp)
        city["date"] = pd.to_datetime(city["date"], errors="coerce").dt.date
        out = out.merge(city, on=["date"], how="left")

    borough_fp = project_root / borough_day_dir
    if borough_fp.exists() and "borough" in out.columns:
        borough = read_partitioned_parquet_dir(borough_fp)
        borough["date"] = pd.to_datetime(borough["date"], errors="coerce").dt.date
        borough["borough"] = borough["borough"].astype(str).str.strip()
        out = out.merge(borough, on=["borough", "date"], how="left")

    borough_hour_fp = project_root / borough_hour_day_dir
    if borough_hour_fp.exists() and {"borough", "hour"}.issubset(out.columns):
        bh = read_partitioned_parquet_dir(borough_hour_fp)
        bh["date"] = pd.to_datetime(bh["date"], errors="coerce").dt.date
        bh["hour"] = pd.to_numeric(bh["hour"], errors="coerce").astype("Int64")
        bh["borough"] = bh["borough"].astype(str).str.strip()
        out = out.merge(bh, on=["borough", "date", "hour"], how="left")

    fill_zero_cols = [c for c in out.columns if c.startswith("restaurant_") and ("count" in c or "inspections" in c or "unique_places" in c)]
    for c in fill_zero_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    print_step_status("Restaurants", f"unido | {len(out):,} filas | {len(out.columns)} columnas")
    return out


def join_rent(
    df: pd.DataFrame,
    project_root: Path,
    city_day_dir: str = "data/aggregated/rent/df_city_day",
    borough_day_dir: str = "data/aggregated/rent/df_borough_day",
    zone_day_dir: str = "data/aggregated/rent/df_zone_day",
) -> pd.DataFrame:
    out = df.copy()

    city_fp = project_root / city_day_dir
    if city_fp.exists():
        city = read_partitioned_parquet_dir(city_fp)
        city["date"] = pd.to_datetime(city["date"], errors="coerce").dt.date
        out = out.merge(city, on=["date"], how="left")

    borough_fp = project_root / borough_day_dir
    if borough_fp.exists() and "borough" in out.columns:
        borough = read_partitioned_parquet_dir(borough_fp)
        borough["date"] = pd.to_datetime(borough["date"], errors="coerce").dt.date
        borough["borough"] = borough["borough"].astype(str).str.strip()
        out = out.merge(borough, on=["borough", "date"], how="left")

    zone_fp = project_root / zone_day_dir
    if zone_fp.exists() and "pu_location_id" in out.columns:
        zone = read_partitioned_parquet_dir(zone_fp)
        zone["date"] = pd.to_datetime(zone["date"], errors="coerce").dt.date
        zone["zone_id"] = pd.to_numeric(zone["zone_id"], errors="coerce").astype("Int64")
        out["pu_location_id"] = pd.to_numeric(out["pu_location_id"], errors="coerce").astype("Int64")
        out = out.merge(zone, left_on=["pu_location_id", "date"], right_on=["zone_id", "date"], how="left")
        if "zone_id" in out.columns:
            out = out.drop(columns=["zone_id"])

    fill_zero_cols = [c for c in out.columns if c.startswith("rent_") and "count" in c]
    for c in fill_zero_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    print_step_status("Rent", f"unido | {len(out):,} filas | {len(out.columns)} columnas")
    return out
