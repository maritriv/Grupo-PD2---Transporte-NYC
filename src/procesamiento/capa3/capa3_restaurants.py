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

from src.procesamiento.capa3.common.io import cleanup_dataset_output

console = Console()
OUTPUT_ZONE_COLUMNS = [
    "pu_location_id",
    "n_restaurants_zone",
    "n_cuisines_zone",
    "mean_score_zone",
    "share_grade_A_zone",
]


def _load_layer2_restaurants(base: Path) -> pd.DataFrame:
    files = sorted(Path(base).glob('restaurants_*.parquet'))
    if not files:
        raise FileNotFoundError(f"No encuentro restaurants_*.parquet en {base}")
    return pd.concat([pd.read_parquet(fp) for fp in files], ignore_index=True)


def build_layer3_restaurants(df2: pd.DataFrame) -> pd.DataFrame:
    df = df2.copy()
    if "pu_location_id" not in df.columns:
        raise ValueError(
            "No existe columna 'pu_location_id' en capa2 restaurants. "
            "Ejecuta capa2_restaurants con mapeo a taxi zones."
        )

    df["pu_location_id"] = pd.to_numeric(df["pu_location_id"], errors="coerce").astype("Int64")
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    else:
        ts = pd.to_datetime(df.get("inspection_date"), errors="coerce")
        df["year"] = ts.dt.year.astype("Int64")

    df["camis"] = pd.to_numeric(df.get("camis"), errors="coerce").astype("Int64")
    df["score"] = pd.to_numeric(df.get("score"), errors="coerce")
    df["cuisine_description"] = (
        df.get("cuisine_description", pd.Series(pd.NA, index=df.index))
        .astype("string")
        .str.strip()
    )
    grade = (
        df.get("grade", pd.Series(pd.NA, index=df.index))
        .astype("string")
        .str.strip()
        .str.upper()
    )
    df["grade"] = grade
    # Regla de negocio: A -> 1, resto (incluido nulo) -> 0.
    df["is_grade_a"] = grade.eq("A").fillna(False).astype("int")
    df["inspection_date"] = pd.to_datetime(df.get("inspection_date"), errors="coerce")

    # Dejamos una fila por restaurante dentro de cada año (la inspección más reciente).
    df = df.dropna(subset=["year", "pu_location_id", "camis", "inspection_date"])
    df = (
        df.sort_values(["year", "inspection_date", "camis"], ascending=[True, True, True])
        .drop_duplicates(subset=["year", "camis"], keep="last")
        .reset_index(drop=True)
    )

    agg = df.groupby(["year", "pu_location_id"], as_index=False).agg(
        n_restaurants_zone=("camis", "nunique"),
        n_cuisines_zone=("cuisine_description", "nunique"),
        mean_score_zone=("score", "mean"),
        share_grade_A_zone=("is_grade_a", "mean"),
    )

    agg["year"] = pd.to_numeric(agg["year"], errors="coerce").astype("Int64")
    agg["pu_location_id"] = pd.to_numeric(agg["pu_location_id"], errors="coerce").astype("Int64")
    agg["n_restaurants_zone"] = pd.to_numeric(agg["n_restaurants_zone"], errors="coerce").astype("Int64")
    agg["n_cuisines_zone"] = pd.to_numeric(agg["n_cuisines_zone"], errors="coerce").astype("Int64")
    out_cols = ["year"] + OUTPUT_ZONE_COLUMNS
    return agg[out_cols].sort_values(["year", "pu_location_id"]).reset_index(drop=True)


def save_layer3_restaurants(df_location_static: pd.DataFrame, out_base: Path, mode: str = "overwrite"):
    out_base = Path(out_base).resolve()
    if mode == "overwrite":
        cleanup_dataset_output(out_base, "df_location_static", label="restaurants location_static")

    out_dir = (out_base / "df_location_static").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if df_location_static.empty:
        return

    for year, g in df_location_static.groupby("year", dropna=False):
        year_dir = (out_dir / f"year={int(year)}").resolve()
        year_dir.mkdir(parents=True, exist_ok=True)
        g_out = g[OUTPUT_ZONE_COLUMNS].copy()
        g_out.to_parquet(year_dir / "part_00000.parquet", index=False, engine="pyarrow")


def main():
    console.print(Panel.fit("[bold cyan]CAPA 3 - RESTAURANTS: AGREGADO POR PU_LOCATION_ID[/bold cyan]"))
    p = argparse.ArgumentParser(description="Capa 3 Restaurants: agregado estatico por pu_location_id.")
    p.add_argument("--in-dir", default=str(obtener_ruta("data/external/restaurants/standarized")))
    p.add_argument("--out-dir", default=str(obtener_ruta("data/external/restaurants/aggregated")))
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
    location_static = build_layer3_restaurants(df2)
    save_layer3_restaurants(location_static, out_base, mode=args.mode)

    summary = Table(show_header=True, header_style="bold magenta", title="Resumen Capa3 Restaurants")
    summary.add_column("Dataset")
    summary.add_column("Rows", justify="right")
    summary.add_row("df_location_static", f"{len(location_static):,}")
    summary.add_row("salida", str((out_base / "df_location_static").resolve()))
    console.print(summary)
    console.print("[bold green]OK[/bold green] Capa 3 RESTAURANTS completada")


if __name__ == "__main__":
    main()
