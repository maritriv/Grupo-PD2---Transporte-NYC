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

from src.procesamiento.capa3.common.io import cleanup_dataset_output, write_partitioned_dataset

console = Console()


def _load_layer2_rent(base: Path) -> pd.DataFrame:
    files = sorted(Path(base).glob('rent_*.parquet'))
    if not files:
        raise FileNotFoundError(f"No encuentro rent_*.parquet en {base}")
    return pd.concat([pd.read_parquet(fp) for fp in files], ignore_index=True)


def build_layer3_rent(df2: pd.DataFrame):
    df = df2.copy()
    df["date"] = pd.to_datetime(df["source_snapshot_date"], errors="coerce").dt.date
    df["borough"] = df["borough"].astype(str).str.strip().str.title()
    if "zone_id" in df.columns:
        df["zone_id"] = pd.to_numeric(df["zone_id"], errors="coerce").astype("Int64")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["availability_365"] = pd.to_numeric(df.get("availability_365"), errors="coerce")
    df["minimum_nights"] = pd.to_numeric(df.get("minimum_nights"), errors="coerce")
    df["accommodates"] = pd.to_numeric(df.get("accommodates"), errors="coerce")
    df = df.dropna(subset=["date", "borough", "price"])

    df_city_day = df.groupby(["date"], as_index=False).agg(
        rent_listing_count_city=("id", "count"),
        rent_price_mean_city=("price", "mean"),
        rent_price_median_city=("price", "median"),
        rent_availability_mean_city=("availability_365", "mean"),
        rent_min_nights_mean_city=("minimum_nights", "mean"),
    )

    df_borough_day = df.groupby(["borough", "date"], as_index=False).agg(
        rent_listing_count_borough=("id", "count"),
        rent_price_mean_borough=("price", "mean"),
        rent_price_median_borough=("price", "median"),
        rent_availability_mean_borough=("availability_365", "mean"),
        rent_accommodates_mean_borough=("accommodates", "mean"),
    )

    if "zone_id" in df.columns and df["zone_id"].notna().any():
        df_zone_day = (
            df.dropna(subset=["zone_id"])
            .groupby(["zone_id", "date"], as_index=False)
            .agg(
                rent_listing_count_zone=("id", "count"),
                rent_price_mean_zone=("price", "mean"),
                rent_price_median_zone=("price", "median"),
                rent_availability_mean_zone=("availability_365", "mean"),
            )
        )
        df_zone_day["zone_id"] = pd.to_numeric(df_zone_day["zone_id"], errors="coerce").astype("Int64")
    else:
        df_zone_day = pd.DataFrame(columns=["zone_id", "date"])

    return df_city_day, df_borough_day, df_zone_day


def save_layer3_rent(df_city_day, df_borough_day, df_zone_day, out_base: Path, mode: str = "overwrite"):
    out_base = Path(out_base).resolve()
    if mode == "overwrite":
        cleanup_dataset_output(out_base, "rent/df_city_day", label="rent city_day")
        cleanup_dataset_output(out_base, "rent/df_borough_day", label="rent borough_day")
        cleanup_dataset_output(out_base, "rent/df_zone_day", label="rent zone_day")
    write_partitioned_dataset(df_city_day, out_base / "rent" / "df_city_day", ["date"])
    write_partitioned_dataset(df_borough_day, out_base / "rent" / "df_borough_day", ["borough", "date"])
    if not df_zone_day.empty:
        write_partitioned_dataset(df_zone_day, out_base / "rent" / "df_zone_day", ["zone_id", "date"])


def main():
    console.print(Panel.fit("[bold cyan]CAPA 3 - RENT: AGREGADOS CITY/BOROUGH/ZONE[/bold cyan]"))
    p = argparse.ArgumentParser(description="Capa 3 Rent: agregados para ML.")
    p.add_argument("--in-dir", default=str(obtener_ruta("data/external/rent/standarized")))
    p.add_argument("--out-dir", default=str(obtener_ruta("data/aggregated")))
    p.add_argument("--mode", choices=["overwrite", "append"], default="overwrite")
    args = p.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_base = Path(args.out_dir).resolve()
    cfg = Table(show_header=True, header_style="bold white", title="Configuracion Capa3 Rent")
    cfg.add_column("Campo", style="bold cyan")
    cfg.add_column("Valor")
    cfg.add_row("in_dir", str(in_dir))
    cfg.add_row("out_dir", str(out_base))
    cfg.add_row("mode", args.mode)
    console.print(cfg)

    df2 = _load_layer2_rent(in_dir)
    city, borough, zone = build_layer3_rent(df2)
    save_layer3_rent(city, borough, zone, out_base, mode=args.mode)

    summary = Table(show_header=True, header_style="bold magenta", title="Resumen Capa3 Rent")
    summary.add_column("Dataset")
    summary.add_column("Rows", justify="right")
    summary.add_row("df_city_day", f"{len(city):,}")
    summary.add_row("df_borough_day", f"{len(borough):,}")
    summary.add_row("df_zone_day", f"{len(zone):,}")
    console.print(summary)
    console.print("[bold green]OK[/bold green] Capa 3 RENT completada")


if __name__ == "__main__":
    main()
