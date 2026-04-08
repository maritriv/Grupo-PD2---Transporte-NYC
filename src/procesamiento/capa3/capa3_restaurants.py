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


def _load_layer2_restaurants(base: Path) -> pd.DataFrame:
    files = sorted(Path(base).glob('restaurants_*.parquet'))
    if not files:
        raise FileNotFoundError(f"No encuentro restaurants_*.parquet en {base}")
    return pd.concat([pd.read_parquet(fp) for fp in files], ignore_index=True)


def build_layer3_restaurants(df2: pd.DataFrame):
    df = df2.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["hour"] = pd.to_numeric(df.get("hour"), errors="coerce").fillna(12).astype("Int64")
    df["boro"] = df["boro"].astype(str).str.strip().str.title()
    df["score"] = pd.to_numeric(df.get("score"), errors="coerce")
    df["critical_flag_norm"] = df.get("critical_flag", pd.Series(pd.NA, index=df.index)).astype("string").str.lower()
    df["is_critical"] = df["critical_flag_norm"].isin(["critical", "y", "yes"]).astype(int)
    df["is_good_grade"] = df.get("grade", pd.Series(pd.NA, index=df.index)).astype("string").str.upper().isin(["A", "B"]).astype(int)
    df = df.dropna(subset=["date", "boro", "camis"])
    df = df[(df["hour"] >= 0) & (df["hour"] <= 23)]

    df_city_day = df.groupby(["date"], as_index=False).agg(
        restaurant_inspections_city=("camis", "count"),
        restaurant_unique_places_city=("camis", "nunique"),
        restaurant_score_mean_city=("score", "mean"),
        restaurant_critical_count_city=("is_critical", "sum"),
        restaurant_good_grade_count_city=("is_good_grade", "sum"),
    )
    df_borough_day = df.groupby(["boro", "date"], as_index=False).agg(
        restaurant_inspections_borough=("camis", "count"),
        restaurant_unique_places_borough=("camis", "nunique"),
        restaurant_score_mean_borough=("score", "mean"),
        restaurant_critical_count_borough=("is_critical", "sum"),
        restaurant_good_grade_count_borough=("is_good_grade", "sum"),
    ).rename(columns={"boro": "borough"})
    df_borough_hour_day = df.groupby(["boro", "date", "hour"], as_index=False).agg(
        restaurant_inspections_borough_hour=("camis", "count"),
        restaurant_score_mean_borough_hour=("score", "mean"),
        restaurant_critical_count_borough_hour=("is_critical", "sum"),
    ).rename(columns={"boro": "borough"})

    return df_city_day, df_borough_day, df_borough_hour_day


def save_layer3_restaurants(df_city_day, df_borough_day, df_borough_hour_day, out_base: Path, mode: str = "overwrite"):
    out_base = Path(out_base).resolve()
    if mode == "overwrite":
        cleanup_dataset_output(out_base, "restaurants/df_city_day", label="restaurants city_day")
        cleanup_dataset_output(out_base, "restaurants/df_borough_day", label="restaurants borough_day")
        cleanup_dataset_output(out_base, "restaurants/df_borough_hour_day", label="restaurants borough_hour_day")
    write_partitioned_dataset(df_city_day, out_base / "restaurants" / "df_city_day", ["date"])
    write_partitioned_dataset(df_borough_day, out_base / "restaurants" / "df_borough_day", ["borough", "date"])
    write_partitioned_dataset(df_borough_hour_day, out_base / "restaurants" / "df_borough_hour_day", ["borough", "date"])


def main():
    console.print(Panel.fit("[bold cyan]CAPA 3 - RESTAURANTS: AGREGADOS CITY/BOROUGH[/bold cyan]"))
    p = argparse.ArgumentParser(description="Capa 3 Restaurants: agregados para ML.")
    p.add_argument("--in-dir", default=str(obtener_ruta("data/external/restaurants/standarized")))
    p.add_argument("--out-dir", default=str(obtener_ruta("data/aggregated")))
    p.add_argument("--mode", choices=["overwrite", "append"], default="overwrite")
    args = p.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_base = Path(args.out_dir).resolve()
    cfg = Table(show_header=True, header_style="bold white", title="Configuracion Capa3 Restaurants")
    cfg.add_column("Campo", style="bold cyan")
    cfg.add_column("Valor")
    cfg.add_row("in_dir", str(in_dir))
    cfg.add_row("out_dir", str(out_base))
    cfg.add_row("mode", args.mode)
    console.print(cfg)

    df2 = _load_layer2_restaurants(in_dir)
    city, borough, borough_hour = build_layer3_restaurants(df2)
    save_layer3_restaurants(city, borough, borough_hour, out_base, mode=args.mode)

    summary = Table(show_header=True, header_style="bold magenta", title="Resumen Capa3 Restaurants")
    summary.add_column("Dataset")
    summary.add_column("Rows", justify="right")
    summary.add_row("df_city_day", f"{len(city):,}")
    summary.add_row("df_borough_day", f"{len(borough):,}")
    summary.add_row("df_borough_hour_day", f"{len(borough_hour):,}")
    console.print(summary)
    console.print("[bold green]OK[/bold green] Capa 3 RESTAURANTS completada")


if __name__ == "__main__":
    main()
