# src/procesamiento/capa3/aggregates/rent.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import obtener_ruta

from src.procesamiento.capa3.common.io import cleanup_dataset_output

console = Console()
OUTPUT_ZONE_COLUMNS = [
    "pu_location_id",
    "rent_price_zone",
    "n_tracts_zone",
    "rent_price_moe_zone",
    "rent_rel_error_zone",
]


def _load_layer2_rent(base: Path) -> pd.DataFrame:
    files = sorted(Path(base).glob('rent_*.parquet'))
    if not files:
        raise FileNotFoundError(f"No encuentro rent_*.parquet en {base}")
    return pd.concat([pd.read_parquet(fp) for fp in files], ignore_index=True)


def _weighted_price_mean(group: pd.DataFrame) -> float:
    """
    Media ponderada de price usando 1/(price_moe^2) cuando hay MOE válido.
    Si no hay pesos válidos, cae a media simple.
    """
    price = pd.to_numeric(group.get("price"), errors="coerce")
    price = price[price.notna() & (price > 0)]
    if price.empty:
        return float("nan")

    moe = pd.to_numeric(group.get("price_moe"), errors="coerce")
    if moe is None:
        return float(price.mean())

    moe = moe.loc[price.index]
    valid_w = moe.notna() & (moe > 0)
    if valid_w.any():
        w = 1.0 / (moe.loc[valid_w] ** 2)
        return float((price.loc[valid_w] * w).sum() / w.sum())

    return float(price.mean())


def _build_location_static(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agregado estatico por zona taxi (location_id), listo para merge con taxis.
    """
    zone_col = "taxi_zone_id" if "taxi_zone_id" in df.columns else "pu_location_id" if "pu_location_id" in df.columns else None
    if zone_col is None or df[zone_col].isna().all():
        return pd.DataFrame(
            columns=[
                "pu_location_id",
                "rent_price_zone",
                "rent_price_moe_zone",
                "n_tracts_zone",
                "rent_rel_error_zone",
            ]
        )

    z = df.copy()
    if "year" in z.columns:
        z["year"] = pd.to_numeric(z["year"], errors="coerce").astype("Int64")
    else:
        z["year"] = pd.to_datetime(z.get("source_snapshot_date"), errors="coerce").dt.year.astype("Int64")
    z["location_id"] = pd.to_numeric(z[zone_col], errors="coerce").astype("Int64")
    z["price"] = pd.to_numeric(z["price"], errors="coerce")
    z["price_moe"] = pd.to_numeric(z.get("price_moe"), errors="coerce")

    tract_col = "source_zone_id" if "source_zone_id" in z.columns else "zone_id" if "zone_id" in z.columns else "id"
    z["tract_id"] = z[tract_col].astype("string").str.strip()
    z["rent_rel_error"] = pd.NA
    rel_ok = z["price"].notna() & (z["price"] > 0) & z["price_moe"].notna() & (z["price_moe"] >= 0)
    z.loc[rel_ok, "rent_rel_error"] = z.loc[rel_ok, "price_moe"] / z.loc[rel_ok, "price"]
    z["rent_rel_error"] = pd.to_numeric(z["rent_rel_error"], errors="coerce")

    # Calidad de nulos por zona (price o price_moe faltantes).
    z = z.dropna(subset=["year", "location_id"])
    z["has_null_quality"] = z[["price", "price_moe"]].isna().any(axis=1).astype("int")
    quality = z.groupby(["year", "location_id"], as_index=False).agg(
        n_tracts_zone=("tract_id", "nunique"),
        rent_null_pct_zone=("has_null_quality", "mean"),
    )
    quality["rent_null_pct_zone"] = quality["rent_null_pct_zone"] * 100.0

    # Agregados de renta usando solo price válido.
    z_valid_price = z[z["price"].notna() & (z["price"] > 0)].copy()
    base = z_valid_price.groupby(["year", "location_id"], as_index=False).agg(
        rent_price_moe_zone=("price_moe", "mean"),
        rent_rel_error_zone=("rent_rel_error", "mean"),
    )
    weighted = (
        z_valid_price.groupby(["year", "location_id"])
        .apply(_weighted_price_mean)
        .rename("rent_price_zone")
        .reset_index()
    )
    out = quality.merge(base, on=["year", "location_id"], how="left").merge(weighted, on=["year", "location_id"], how="left")

    meta_cols = [c for c in ["taxi_borough", "taxi_zone_name", "taxi_service_zone"] if c in z.columns]
    if meta_cols:
        meta = z[["location_id"] + meta_cols].drop_duplicates(subset=["location_id"], keep="first")
        out = out.merge(meta, on="location_id", how="left")
    else:
        out["taxi_borough"] = pd.NA
        out["taxi_zone_name"] = pd.NA
        out["taxi_service_zone"] = pd.NA

    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["location_id"] = pd.to_numeric(out["location_id"], errors="coerce").astype("Int64")
    out["pu_location_id"] = out["location_id"].astype("Int64")
    out["n_tracts_zone"] = pd.to_numeric(out["n_tracts_zone"], errors="coerce").astype("Int64")

    cols = [
        "year",
        "rent_price_zone",
        "rent_price_moe_zone",
        "n_tracts_zone",
        "rent_rel_error_zone",
        "pu_location_id",
    ]
    out = out[cols].sort_values(["year", "pu_location_id"]).reset_index(drop=True)
    return out


def build_layer3_rent(df2: pd.DataFrame):
    df = df2.copy()
    if "taxi_zone_id" in df.columns:
        df["taxi_zone_id"] = pd.to_numeric(df["taxi_zone_id"], errors="coerce").astype("Int64")
    if "zone_id" in df.columns:
        df["zone_id"] = pd.to_numeric(df["zone_id"], errors="coerce").astype("Int64")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df_location_static = _build_location_static(df)
    return df_location_static


def save_layer3_rent(df_location_static, out_base: Path, mode: str = "overwrite"):
    out_base = Path(out_base).resolve()
    if mode == "overwrite":
        cleanup_dataset_output(out_base, "df_location_static", label="rent location_static")

    static_dir = (out_base / "df_location_static").resolve()
    static_dir.mkdir(parents=True, exist_ok=True)
    if not df_location_static.empty:
        for year, g in df_location_static.groupby("year", dropna=False):
            year_dir = (static_dir / f"year={int(year)}").resolve()
            year_dir.mkdir(parents=True, exist_ok=True)
            g_out = g[OUTPUT_ZONE_COLUMNS].copy()
            g_out.to_parquet(year_dir / "part_00000.parquet", index=False, engine="pyarrow")


def main():
    console.print(Panel.fit("[bold cyan]CAPA 3 - RENT: AGREGADO ESTATICO POR LOCATION_ID[/bold cyan]"))
    p = argparse.ArgumentParser(description="Capa 3 Rent: agregado estatico por location_id.")
    p.add_argument("--in-dir", default=str(obtener_ruta("data/external/rent/standarized")))
    p.add_argument("--out-dir", default=str(obtener_ruta("data/external/rent/aggregated")))
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
    location_static = build_layer3_rent(df2)
    save_layer3_rent(location_static, out_base, mode=args.mode)

    summary = Table(show_header=True, header_style="bold magenta", title="Resumen Capa3 Rent")
    summary.add_column("Dataset")
    summary.add_column("Rows", justify="right")
    summary.add_row("df_location_static", f"{len(location_static):,}")
    summary.add_row("salida", str((out_base / "df_location_static").resolve()))
    console.print(summary)
    console.print("[bold green]OK[/bold green] Capa 3 RENT completada")


if __name__ == "__main__":
    main()
