from __future__ import annotations

import argparse
from datetime import datetime
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


console = Console()
MIN_YEAR = 2023
MAX_YEAR = 2025


def read_raw_rent(in_dir: Path) -> pd.DataFrame:
    in_dir = Path(in_dir).resolve()
    if not in_dir.exists():
        raise FileNotFoundError(f"No existe el directorio RAW: {in_dir}")

    files = sorted(in_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No hay .parquet en {in_dir}")

    console.print(f"[cyan]Lectura RAW[/cyan] {len(files)} snapshots rent")
    return pd.concat([pd.read_parquet(fp) for fp in files], ignore_index=True)


def filter_by_range(df: pd.DataFrame, date_from: str | None, date_to: str | None) -> pd.DataFrame:
    if "source_snapshot_date" not in df.columns or (date_from is None and date_to is None):
        return df

    out = df.copy()
    out["_date_ts"] = pd.to_datetime(out["source_snapshot_date"], errors="coerce")
    if date_from is not None:
        out = out[out["_date_ts"] >= pd.to_datetime(date_from)]
    if date_to is not None:
        out = out[out["_date_ts"] <= pd.to_datetime(date_to)]
    return out.drop(columns=["_date_ts"])


def build_layer2_rent(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["source_snapshot_date"] = pd.to_datetime(df2["source_snapshot_date"], errors="coerce").dt.date
    for c in ["latitude", "longitude", "price", "price_moe"]:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
    for c in ["id", "accommodates", "minimum_nights", "availability_365"]:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce").astype("Int64")

    for c in ["zone_id", "zone_name", "borough", "neighborhood", "room_type", "property_type"]:
        if c in df2.columns:
            df2[c] = df2[c].astype("string").str.strip()

    df2 = df2.dropna(subset=["source_snapshot_date", "borough", "price"])
    df2 = df2[df2["price"] > 0]

    dt = pd.to_datetime(df2["source_snapshot_date"])
    df2["year"] = dt.dt.year.astype("int")
    df2["month"] = dt.dt.month.astype("int")
    df2 = df2[(df2["year"] >= MIN_YEAR) & (df2["year"] <= MAX_YEAR)]

    cols = [
        "id", "zone_id", "zone_name", "source_snapshot_date", "year", "month", "borough", "neighborhood",
        "latitude", "longitude", "room_type", "property_type", "accommodates", "minimum_nights",
        "availability_365", "price", "price_moe",
    ]
    cols = [c for c in cols if c in df2.columns]
    return df2[cols].drop_duplicates()


def save_layer2_rent(df: pd.DataFrame, out_dir: Path) -> int:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files_written = 0
    for (y, m), g in df.groupby(["year", "month"], dropna=False):
        file_name = f"rent_{int(y)}-{int(m):02d}.parquet"
        g.to_parquet(out_dir / file_name, index=False, engine="pyarrow")
        files_written += 1
    return files_written


def main():
    console.print(Panel.fit("[bold cyan]CAPA 2 - RENT: TIPADO + HIGIENE + PARTICIONADO[/bold cyan]"))

    p = argparse.ArgumentParser(description="Capa 2 Rent: tipado + higiene + particionado.")
    p.add_argument("--from", dest="date_from", default=None, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--to", dest="date_to", default=None, help="YYYY-MM-DD (inclusive)")
    args = p.parse_args()

    if args.date_from is not None:
        datetime.strptime(args.date_from, "%Y-%m-%d")
    if args.date_to is not None:
        datetime.strptime(args.date_to, "%Y-%m-%d")

    project_root = Path(__file__).resolve().parents[3]
    validated_dir = (project_root / "data" / "external" / "rent" / "validated").resolve()
    raw_dir = (project_root / "data" / "external" / "rent" / "raw").resolve()
    in_dir = validated_dir if validated_dir.exists() else raw_dir
    out_dir = (project_root / "data" / "external" / "rent" / "standarized").resolve()

    cfg = Table(show_header=True, header_style="bold white", title="Configuracion Capa2 Rent")
    cfg.add_column("Campo", style="bold cyan")
    cfg.add_column("Valor")
    cfg.add_row("in_dir", str(in_dir))
    cfg.add_row("out_dir", str(out_dir))
    cfg.add_row("filtro", f"{args.date_from or '...'} -> {args.date_to or '...'}")
    console.print(cfg)

    with console.status("[cyan]Procesando rent...[/cyan]"):
        df_raw = read_raw_rent(in_dir)
        df_raw = filter_by_range(df_raw, args.date_from, args.date_to)
        df_l2 = build_layer2_rent(df_raw)
        files_written = save_layer2_rent(df_l2, out_dir)

    summary = Table(show_header=True, header_style="bold magenta", title="Resumen Capa2 Rent")
    summary.add_column("Metrica", style="bold white")
    summary.add_column("Valor", justify="right")
    summary.add_row("Rows raw", f"{len(df_raw):,}")
    summary.add_row("Rows capa2", f"{len(df_l2):,}")
    summary.add_row("Parquets escritos", f"{files_written:,}")
    summary.add_row("Salida", str(out_dir))
    console.print(summary)
    console.print("[bold green]OK[/bold green] Capa 2 RENT completada")


if __name__ == "__main__":
    main()
