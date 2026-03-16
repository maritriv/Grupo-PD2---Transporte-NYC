from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from src.pipeline_runner import print_done, print_stage
from src.ml.dataset.modules.business_metrics import add_business_metrics, reorder_dataset_columns
from src.ml.dataset.modules.display import print_build_summary, print_rich_table_preview, print_step_status
from src.ml.dataset.modules.enrichments import join_events, join_meteo
from src.ml.dataset.modules.feature_engineering import add_temporal_features
from src.ml.dataset.modules.io import ensure_cols, read_partitioned_parquet_dir, safe_date_for_filename
from src.ml.dataset.modules.zone_lookup import add_borough_to_tlc, load_zone_lookup

MIN_ALLOWED_DATE = pd.Timestamp("2023-01-01").date()
MAX_ALLOWED_DATE = pd.Timestamp("2025-12-31").date()

# Para generar dataset completo
# uv run -m src.ml.dataset.a_build_dataset
# Genera: data/ml/dataset_completo.parquet
#
# Para generar un dataset más pequeño para entrenar los modelos más rápidamente (rango por fechas)
# uv run -m src.ml.dataset.a_build_dataset --from 2024-01-01 --to 2024-01-14
# Genera: data/ml/dataset_rango_2024-01-01__2024-01-14.parquet
#
# Para generar un dataset más pequeño para entrenar los modelos más rápidamente (muestreo aleatorio)
# uv run -m src.ml.dataset.a_build_dataset --sample-frac 0.01
# Genera: data/ml/dataset_completo.parquet pero con 1% de filas (si no pasas --from/--to)


def build_dataset(
    tlc_dir: str = "data/aggregated/df_zone_hour_day_service",
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
    """
    Genera dataset final de modelado:
    - Base: capa 3 TLC (zona-hora-fecha-servicio)
    - Enriquecimiento: borough, meteo, eventos
    - Features temporales
    - Métricas de negocio y target operativo
    """
    print_stage("ML DATASET BUILD", "Construyendo dataset final", color="yellow")
    project_root = Path(__file__).resolve().parents[3]

    # --- 1) TLC base
    tlc_base = project_root / tlc_dir
    tlc = read_partitioned_parquet_dir(tlc_base)

    ensure_cols(
        tlc,
        ["pu_location_id", "hour", "date", "service_type", "num_trips", "avg_price", "std_price"],
        "TLC",
    )

    tlc["date"] = pd.to_datetime(tlc["date"], errors="coerce").dt.date
    tlc["hour"] = pd.to_numeric(tlc["hour"], errors="coerce").astype("Int64")
    tlc["pu_location_id"] = pd.to_numeric(tlc["pu_location_id"], errors="coerce").astype("Int64")
    tlc["num_trips"] = pd.to_numeric(tlc["num_trips"], errors="coerce")
    tlc["avg_price"] = pd.to_numeric(tlc["avg_price"], errors="coerce")
    tlc["std_price"] = pd.to_numeric(tlc["std_price"], errors="coerce")
    tlc["service_type"] = tlc["service_type"].astype(str)

    tlc = tlc.dropna(subset=["date", "hour", "pu_location_id", "service_type", "num_trips", "std_price"])
    tlc = tlc[(tlc["date"] >= MIN_ALLOWED_DATE) & (tlc["date"] <= MAX_ALLOWED_DATE)]

    if date_from is not None:
        tlc = tlc[tlc["date"] >= pd.to_datetime(date_from).date()]
    if date_to is not None:
        tlc = tlc[tlc["date"] <= pd.to_datetime(date_to).date()]

    if sample_frac is not None and 0 < sample_frac < 1:
        tlc = tlc.sample(frac=sample_frac, random_state=random_state)
        print_step_status("Muestreo", f"frac={sample_frac}")

    print_step_status("TLC base", f"{len(tlc):,} filas | {len(tlc.columns)} columnas")

    # --- 2) Borough
    zones = load_zone_lookup(project_root, zone_lookup_path)
    tlc = add_borough_to_tlc(tlc, zones)

    # --- 3) Meteo
    df = join_meteo(tlc, project_root, meteo_path)

    # --- 4) Eventos
    df = join_events(df, project_root, events_dir)

    # --- 5) Features temporales
    df = add_temporal_features(df)

    # --- 6) Métricas de negocio
    df = add_business_metrics(df, project_root, variability_dir)

    # --- 7) Orden final
    df = reorder_dataset_columns(df)

    print_rich_table_preview(
        df,
        "Preview dataset final",
        cols=[
            "pu_location_id",
            "borough",
            "hour",
            "date",
            "service_type",
            "num_trips",
            "avg_price",
            "price_variability_rel",
            "event_count_city",
            "event_count_borough",
            "stress_score",
            "is_stress",
        ],
        n=10,
        max_col_width=16,
    )

    # --- 8) Guardado
    out_fp = (project_root / out_path).resolve()
    os.makedirs(out_fp.parent, exist_ok=True)
    df.to_parquet(out_fp, index=False)

    print_build_summary(df, out_fp)
    print_done("DATASET GUARDADO CORRECTAMENTE")
    return df


def main() -> None:
    p = argparse.ArgumentParser(
        description="Construye dataset de modelado (TLC + Meteo + Eventos). "
        "Por defecto genera el dataset completo."
    )
    p.add_argument(
        "--from",
        dest="date_from",
        default=None,
        help="YYYY-MM-DD (inclusive). Si se indica junto con --to, genera dataset por rango.",
    )
    p.add_argument(
        "--to",
        dest="date_to",
        default=None,
        help="YYYY-MM-DD (inclusive). Si se indica junto con --from, genera dataset por rango.",
    )
    p.add_argument("--sample-frac", type=float, default=None, help="Opcional: muestreo aleatorio (0<frac<1).")
    p.add_argument("--out-dir", default="data/ml", help="Carpeta de salida (default: data/ml).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)

    if args.date_from and args.date_to:
        f = safe_date_for_filename(args.date_from)
        t = safe_date_for_filename(args.date_to)
        out_name = f"dataset_rango_{f}__{t}.parquet"
    else:
        out_name = "dataset_completo.parquet"

    build_dataset(
        out_path=str(out_dir / out_name),
        date_from=args.date_from,
        date_to=args.date_to,
        sample_frac=args.sample_frac,
    )


if __name__ == "__main__":
    main()