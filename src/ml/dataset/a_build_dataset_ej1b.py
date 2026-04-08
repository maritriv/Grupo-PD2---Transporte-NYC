from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from config.pipeline_runner import print_done, print_stage
from src.ml.dataset.modules.display import print_build_summary, print_rich_table_preview, print_step_status
from src.ml.dataset.modules.enrichments_ej2 import join_events, join_meteo
from src.ml.dataset.modules.enrichments_ej1 import join_rent, join_restaurants
from src.ml.dataset.modules.io import ensure_cols, read_partitioned_parquet_dir, safe_date_for_filename
from src.ml.dataset.modules.zone_lookup import load_zone_lookup

MIN_ALLOWED_DATE = pd.Timestamp("2023-01-01").date()
MAX_ALLOWED_DATE = pd.Timestamp("2025-12-31").date()


def _add_boroughs(df: pd.DataFrame, zones: pd.DataFrame | None) -> pd.DataFrame:
    if zones is None:
        df = df.copy()
        df["borough"] = pd.NA
        df["borough_dropoff"] = pd.NA
        return df
    zones_pick = zones.rename(columns={"borough": "borough"})
    zones_drop = zones.rename(columns={"pu_location_id": "do_location_id", "borough": "borough_dropoff"})
    out = df.merge(zones_pick, on="pu_location_id", how="left")
    out = out.merge(zones_drop[["do_location_id", "borough_dropoff"]], on="do_location_id", how="left")
    return out


def build_dataset_ex1b(
    tip_base_dir: str = "data/aggregated/ex1b/df_tip_trip_level",
    meteo_path: str = "data/external/meteo/aggregated/df_hour_day/data.parquet",
    events_dir: str = "data/aggregated/events/df_borough_hour_day",
    zone_lookup_path: str = "data/external/taxi_zone_lookup.csv",
    out_path: str = "data/ml/dataset_ex1b_tip_completo.parquet",
    date_from: str | None = None,
    date_to: str | None = None,
    sample_frac: float | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    print_stage("ML DATASET BUILD EX1(b)", "Construyendo dataset de propina a nivel viaje", color="yellow")
    project_root = Path(__file__).resolve().parents[3]

    df = read_partitioned_parquet_dir(project_root / tip_base_dir)
    ensure_cols(df, ["date", "hour", "pu_location_id", "do_location_id", "target_tip_amount", "target_tip_pct"], "EX1B")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype("Int64")
    df["pu_location_id"] = pd.to_numeric(df["pu_location_id"], errors="coerce").astype("Int64")
    df["do_location_id"] = pd.to_numeric(df["do_location_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["date", "hour", "pu_location_id", "target_tip_amount"])
    df = df[(df["date"] >= MIN_ALLOWED_DATE) & (df["date"] <= MAX_ALLOWED_DATE)]

    if date_from is not None:
        df = df[df["date"] >= pd.to_datetime(date_from).date()]
    if date_to is not None:
        df = df[df["date"] <= pd.to_datetime(date_to).date()]
    if sample_frac is not None and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=random_state)
        print_step_status("Muestreo", f"frac={sample_frac}")

    zones = load_zone_lookup(project_root, zone_lookup_path)
    df = _add_boroughs(df, zones)
    df = join_meteo(df, project_root, meteo_path)
    df = join_events(df, project_root, events_dir)
    df = join_restaurants(df, project_root)
    df = join_rent(df, project_root)

    dt = pd.to_datetime(df["date"])
    df["day_of_week"] = dt.dt.dayofweek.astype(int)
    df["month"] = dt.dt.month.astype(int)
    df["day"] = dt.dt.day.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    df["is_night_hour"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)

    out_fp = (project_root / out_path).resolve()
    os.makedirs(out_fp.parent, exist_ok=True)
    df.to_parquet(out_fp, index=False)

    print_rich_table_preview(df, "Preview dataset EX1(b)", cols=[c for c in ["pu_location_id", "do_location_id", "borough", "hour", "date", "fare_amount", "target_tip_amount", "target_tip_pct", "restaurant_inspections_borough", "rent_price_mean_borough"] if c in df.columns], n=10, max_col_width=16)
    print_build_summary(df, out_fp)
    print_done("DATASET EX1(b) GUARDADO CORRECTAMENTE")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Construye dataset EX1(b) para prediccion de propina.")
    p.add_argument("--from", dest="date_from", default=None)
    p.add_argument("--to", dest="date_to", default=None)
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--out-dir", default="data/ml")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    if args.date_from and args.date_to:
        out_name = f"dataset_ex1b_tip_rango_{safe_date_for_filename(args.date_from)}__{safe_date_for_filename(args.date_to)}.parquet"
    else:
        out_name = "dataset_ex1b_tip_completo.parquet"

    build_dataset_ex1b(out_path=str(out_dir / out_name), date_from=args.date_from, date_to=args.date_to, sample_frac=args.sample_frac)


if __name__ == "__main__":
    main()
