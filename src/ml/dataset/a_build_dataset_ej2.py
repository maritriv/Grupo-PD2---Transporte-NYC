from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from config.pipeline_runner import print_done, print_stage
from src.ml.dataset.modules.business_metrics import add_business_metrics, reorder_dataset_columns
from src.ml.dataset.modules.display import print_build_summary, print_rich_table_preview, print_step_status
from src.ml.dataset.modules.enrichments_ej2 import join_events, join_meteo
from src.ml.dataset.modules.feature_engineering import add_temporal_features
from src.ml.dataset.modules.io import ensure_cols, read_partitioned_parquet_dir, safe_date_for_filename
from src.ml.dataset.modules.zone_lookup import add_borough_to_tlc, load_zone_lookup

MIN_ALLOWED_DATE = pd.Timestamp("2023-01-01").date()
MAX_ALLOWED_DATE = pd.Timestamp("2025-12-31").date()


def build_dataset_ej2(
    tlc_base_dir: str = "data/aggregated/df_zone_hour_day_service",
    meteo_path: str = "data/external/meteo/aggregated/df_hour_day/data.parquet",
    events_dir: str = "data/aggregated/events/df_borough_hour_day",
    variability_dir: str = "data/aggregated/df_variability",
    zone_lookup_path: str = "data/external/taxi_zone_lookup.csv",
    out_path: str = "data/ml/dataset_completo.parquet",
    date_from: str | None = None,
    date_to: str | None = None,
    sample_frac: float | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    print_stage("ML DATASET BUILD EJ2", "Construyendo dataset principal del ejercicio 2", color="yellow")
    project_root = Path(__file__).resolve().parents[3]

    df = read_partitioned_parquet_dir(project_root / tlc_base_dir)
    ensure_cols(df, ["date", "hour", "service_type", "pu_location_id", "num_trips", "avg_price", "std_price"], "EJ2")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype("Int64")
    df["pu_location_id"] = pd.to_numeric(df["pu_location_id"], errors="coerce").astype("Int64")
    df["service_type"] = df["service_type"].astype(str).str.strip()

    for c in ["num_trips", "avg_price", "std_price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date", "hour", "service_type", "pu_location_id", "num_trips", "avg_price", "std_price"])
    df = df[(df["date"] >= MIN_ALLOWED_DATE) & (df["date"] <= MAX_ALLOWED_DATE)]
    df = df[(df["hour"] >= 0) & (df["hour"] <= 23)]
    df = df[df["num_trips"] >= 0]

    if date_from is not None:
        df = df[df["date"] >= pd.to_datetime(date_from).date()]
    if date_to is not None:
        df = df[df["date"] <= pd.to_datetime(date_to).date()]
    if sample_frac is not None and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=random_state)
        print_step_status("Muestreo", f"frac={sample_frac}")

    zones = load_zone_lookup(project_root, zone_lookup_path)
    df = add_borough_to_tlc(df, zones)

    df = join_meteo(df, project_root, meteo_path)
    df = join_events(df, project_root, events_dir)
    df = add_temporal_features(df)
    df = add_business_metrics(df, project_root, variability_dir)
    df = reorder_dataset_columns(df)

    out_fp = (project_root / out_path).resolve()
    os.makedirs(out_fp.parent, exist_ok=True)
    df.to_parquet(out_fp, index=False)

    print_rich_table_preview(
        df,
        "Preview dataset EJ2",
        cols=[
            c for c in [
                "pu_location_id", "borough", "service_type", "hour", "date",
                "num_trips", "avg_price", "std_price", "event_count_city",
                "temp_c_mean", "lag_1h_trips", "lag_24h_trips", "stress_score", "is_stress",
            ] if c in df.columns
        ],
        n=10,
        max_col_width=16,
    )
    print_build_summary(df, out_fp)
    print_done("DATASET EJ2 GUARDADO CORRECTAMENTE")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Construye el dataset principal del ejercicio 2.")
    p.add_argument("--from", dest="date_from", default=None)
    p.add_argument("--to", dest="date_to", default=None)
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--out-dir", default="data/ml")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    if args.date_from and args.date_to:
        out_name = f"dataset_rango_{safe_date_for_filename(args.date_from)}__{safe_date_for_filename(args.date_to)}.parquet"
    else:
        out_name = "dataset_completo.parquet"

    build_dataset_ej2(
        out_path=str(out_dir / out_name),
        date_from=args.date_from,
        date_to=args.date_to,
        sample_frac=args.sample_frac,
    )


if __name__ == "__main__":
    main()
