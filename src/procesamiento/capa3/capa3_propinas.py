from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    from config.settings import obtener_ruta  # type: ignore
except Exception:
    def obtener_ruta(p: str) -> Path:
        return Path(p)

from src.procesamiento.capa3.common.io import cleanup_dataset_output, iter_month_partitions, write_partitioned_dataset

console = Console()

TIP_COLS = [
    "service_type", "pickup_datetime", "dropoff_datetime", "date", "year", "month", "hour", "day_of_week",
    "is_weekend", "week_of_year", "pu_location_id", "do_location_id", "trip_distance", "trip_duration_min",
    "total_amount_std", "fare_amount", "tip_amount", "tip_pct", "passenger_count", "payment_type", "RatecodeID",
    "is_valid_for_tip", "is_valid_for_distance", "is_valid_for_duration", "is_valid_for_price",
]


def _load_tlc_standardized(base: Path) -> pd.DataFrame:
    parts = []
    for _y, _m, files in iter_month_partitions(base):
        for fp in files:
            parts.append(pd.read_parquet(fp, columns=[c for c in TIP_COLS if c]))
    if not parts:
        raise FileNotFoundError(f"No se encontraron particiones TLC en {base}")
    return pd.concat(parts, ignore_index=True)


def build_tip_trip_level(df: pd.DataFrame, date_from: str | None = None, date_to: str | None = None) -> pd.DataFrame:
    out = df.copy()
    out["pickup_datetime"] = pd.to_datetime(out["pickup_datetime"], errors="coerce")
    out["dropoff_datetime"] = pd.to_datetime(out["dropoff_datetime"], errors="coerce")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date

    num_cols = [
        "year", "month", "hour", "day_of_week", "is_weekend", "week_of_year", "pu_location_id", "do_location_id",
        "trip_distance", "trip_duration_min", "total_amount_std", "fare_amount", "tip_amount", "tip_pct",
        "passenger_count", "payment_type", "RatecodeID", "is_valid_for_tip", "is_valid_for_distance",
        "is_valid_for_duration", "is_valid_for_price",
    ]
    for c in num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["pickup_datetime", "date", "pu_location_id", "fare_amount", "tip_amount"])
    out = out[out.get("is_valid_for_tip", 1).fillna(0).astype(int) == 1]
    out = out[out["fare_amount"] > 0]

    if date_from is not None:
        out = out[out["date"] >= pd.to_datetime(date_from).date()]
    if date_to is not None:
        out = out[out["date"] <= pd.to_datetime(date_to).date()]

    out["target_tip_amount"] = pd.to_numeric(out["tip_amount"], errors="coerce")
    out["target_tip_pct"] = pd.to_numeric(out["tip_pct"], errors="coerce")
    out["has_tip"] = (out["target_tip_amount"] > 0).astype(int)

    final_cols = [
        "service_type", "pickup_datetime", "dropoff_datetime", "date", "year", "month", "hour", "day_of_week",
        "is_weekend", "week_of_year", "pu_location_id", "do_location_id", "trip_distance", "trip_duration_min",
        "total_amount_std", "fare_amount", "passenger_count", "payment_type", "RatecodeID",
        "target_tip_amount", "target_tip_pct", "has_tip",
    ]
    final_cols = [c for c in final_cols if c in out.columns]
    return out[final_cols].drop_duplicates()


def main():
    console.print(Panel.fit("[bold cyan]CAPA 3 EX1(b) - TLC TRIP LEVEL PARA PROPINA[/bold cyan]"))
    p = argparse.ArgumentParser(description="Construye base a nivel viaje para prediccion de propina.")
    p.add_argument("--in-dir", default=str(obtener_ruta("data/standarized")))
    p.add_argument("--out-dir", default=str(obtener_ruta("data/aggregated/ex1b")))
    p.add_argument("--from", dest="date_from", default=None)
    p.add_argument("--to", dest="date_to", default=None)
    p.add_argument("--mode", choices=["overwrite", "append"], default="overwrite")
    args = p.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_base = Path(args.out_dir).resolve()
    if args.mode == "overwrite":
        cleanup_dataset_output(out_base, "df_tip_trip_level", label="EX1(b)")

    df = _load_tlc_standardized(in_dir)
    out = build_tip_trip_level(df, date_from=args.date_from, date_to=args.date_to)
    write_partitioned_dataset(out, out_base / "df_tip_trip_level", ["year", "month"])

    summary = Table(show_header=True, header_style="bold magenta", title="Resumen EX1(b) Trip Level")
    summary.add_column("Metrica", style="bold white")
    summary.add_column("Valor", justify="right")
    summary.add_row("Rows input", f"{len(df):,}")
    summary.add_row("Rows output", f"{len(out):,}")
    summary.add_row("Output", str(out_base / 'df_tip_trip_level'))
    console.print(summary)
    console.print("[bold green]OK[/bold green] EX1(b) trip level completado")


if __name__ == "__main__":
    main()
