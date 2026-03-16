# src/procesamiento/capa2/capa2_meteo.py
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    from config.settings import obtener_ruta, config  # type: ignore
except Exception:
    config = {}

    def obtener_ruta(p: str) -> Path:
        return Path(p)


console = Console()
DEBUG = False  # True para ver previews
MIN_YEAR = 2023
MAX_YEAR = 2025


def read_raw_meteo(in_dir: Path, base_name: str = "meteo_hourly_nyc") -> pd.DataFrame:
    """
    Lee datos RAW de meteo en dos formatos posibles:

    1) Formato antiguo (1 archivo):
        - {base_name}.parquet  o  {base_name}.csv

    2) Formato nuevo (muchos archivos mensuales):
        - meteo_YYYY_MM.parquet (y opcionalmente .parquet.gz)
        - (si quisieras) meteo_YYYY_MM.csv (y .csv.gz)
    """
    in_dir = Path(in_dir).resolve()
    if not in_dir.exists():
        raise FileNotFoundError(f"No existe el directorio RAW: {in_dir}")

    parquet_path = in_dir / f"{base_name}.parquet"
    csv_path = in_dir / f"{base_name}.csv"

    if parquet_path.exists():
        console.print(f"[cyan]Lectura RAW[/cyan] parquet unico: {parquet_path.name}")
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        console.print(f"[cyan]Lectura RAW[/cyan] csv unico: {csv_path.name}")
        return pd.read_csv(csv_path)

    files = sorted(in_dir.glob("meteo_*.parquet"))
    if not files:
        files = sorted(in_dir.glob("meteo_*.parquet.gz"))

    if files:
        console.print(f"[cyan]Lectura RAW[/cyan] {len(files)} archivos mensuales parquet")
        dfs = [pd.read_parquet(fp) for fp in files]
        return pd.concat(dfs, ignore_index=True)

    files = sorted(in_dir.glob("meteo_*.csv"))
    if not files:
        files = sorted(in_dir.glob("meteo_*.csv.gz"))

    if files:
        console.print(f"[cyan]Lectura RAW[/cyan] {len(files)} archivos mensuales csv")
        dfs = [pd.read_csv(fp) for fp in files]
        return pd.concat(dfs, ignore_index=True)

    raise FileNotFoundError(
        f"No encuentro {base_name}.parquet ni {base_name}.csv en {in_dir}, "
        f"ni archivos mensuales meteo_*.parquet / meteo_*.csv."
    )


def _parse_date(s: str | None) -> pd.Timestamp | None:
    if s is None:
        return None
    return pd.to_datetime(s, format="%Y-%m-%d", errors="raise")


def filter_by_range(df: pd.DataFrame, date_from: str | None, date_to: str | None) -> pd.DataFrame:
    """Filtra por rango inclusive [date_from, date_to] usando la columna 'date'."""
    if date_from is None and date_to is None:
        return df

    dt = pd.to_datetime(df["date"], errors="coerce")
    out = df.copy()
    out["_date_ts"] = dt

    if date_from is not None:
        dfrom = _parse_date(date_from)
        out = out[out["_date_ts"] >= dfrom]

    if date_to is not None:
        dto = _parse_date(date_to)
        out = out[out["_date_ts"] <= dto]

    out = out.drop(columns=["_date_ts"])
    return out


def build_layer2_meteo(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()

    # Tipos
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce").dt.date
    df2["hour"] = pd.to_numeric(df2["hour"], errors="coerce").astype("Int64")

    numeric_cols = ["temp_c", "precip_mm", "rain_mm", "snowfall_mm", "wind_kmh"]
    for c in numeric_cols:
        df2[c] = pd.to_numeric(df2.get(c), errors="coerce")

    df2["weather_code"] = pd.to_numeric(df2.get("weather_code"), errors="coerce").astype("Int64")

    # Higiene
    df2 = df2.dropna(subset=["date", "hour"])
    df2 = df2[(df2["hour"] >= 0) & (df2["hour"] <= 23)]

    # Variables temporales
    dt = pd.to_datetime(df2["date"])
    df2["year"] = dt.dt.year.astype("int")
    df2["month"] = dt.dt.month.astype("int")
    df2 = df2[(df2["year"] >= MIN_YEAR) & (df2["year"] <= MAX_YEAR)]
    dt = pd.to_datetime(df2["date"])
    # Spark dayofweek: 1=Sunday ... 7=Saturday
    df2["day_of_week"] = ((dt.dt.dayofweek + 1) % 7) + 1
    df2["is_weekend"] = df2["day_of_week"].isin([1, 7]).astype("int")
    df2["week_of_year"] = dt.dt.isocalendar().week.astype("int")

    cols = [
        "date",
        "hour",
        "year",
        "month",
        "day_of_week",
        "is_weekend",
        "week_of_year",
        "temp_c",
        "precip_mm",
        "rain_mm",
        "snowfall_mm",
        "wind_kmh",
        "weather_code",
    ]
    cols = [c for c in cols if c in df2.columns]
    df2 = df2[cols]

    if DEBUG:
        console.print(df2.head(30))

    return df2


def save_layer2_meteo(df: pd.DataFrame, out_dir: Path) -> int:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files_written = 0
    for (y, m), g in df.groupby(["year", "month"], dropna=False):
        file_name = f"meteo_{int(y)}-{int(m):02d}.parquet"
        dest_path = out_dir / file_name
        g.to_parquet(dest_path, index=False, engine="pyarrow")
        files_written += 1
    return files_written


def main():
    console.print(Panel.fit("[bold cyan]CAPA 2 - METEO: TIPADO + HIGIENE + PARTICIONADO[/bold cyan]"))

    meteo_cfg = (config.get("meteo") or {}) if isinstance(config, dict) else {}
    default_raw_name = meteo_cfg.get("out_name", "meteo_hourly_nyc")

    p = argparse.ArgumentParser(description="Capa 2 Meteo (sin Spark): tipado + higiene + particionado.")
    p.add_argument("--name", dest="raw_name", default=default_raw_name, help="Nombre base del meteo raw (formato antiguo)")
    p.add_argument("--from", dest="date_from", default=None, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--to", dest="date_to", default=None, help="YYYY-MM-DD (inclusive)")
    args = p.parse_args()

    if args.date_from is not None:
        datetime.strptime(args.date_from, "%Y-%m-%d")
    if args.date_to is not None:
        datetime.strptime(args.date_to, "%Y-%m-%d")

    project_root = Path(__file__).resolve().parents[3]
    validated_dir = (project_root / "data" / "external" / "meteo" / "validated" / "clean").resolve()
    fallback_raw_dir = (project_root / "data" / "external" / "meteo" / "raw").resolve()
    raw_dir = validated_dir if validated_dir.exists() else fallback_raw_dir
    out_dir = (project_root / "data" / "external" / "meteo" / "standarized").resolve()

    cfg_table = Table(show_header=True, header_style="bold white", title="Configuracion Capa2 Meteo")
    cfg_table.add_column("Campo", style="bold cyan")
    cfg_table.add_column("Valor")
    cfg_table.add_row("raw_dir", str(raw_dir))
    cfg_table.add_row("out_dir", str(out_dir))
    cfg_table.add_row("filtro", f"{args.date_from or '...'} -> {args.date_to or '...'}")
    console.print(cfg_table)

    with console.status("[cyan]Procesando meteo...[/cyan]"):
        df_raw = read_raw_meteo(raw_dir, base_name=args.raw_name)
        df_raw = filter_by_range(df_raw, args.date_from, args.date_to)
        df_l2 = build_layer2_meteo(df_raw)
        files_written = save_layer2_meteo(df_l2, out_dir)

    summary = Table(show_header=True, header_style="bold magenta", title="Resumen Capa2 Meteo")
    summary.add_column("Metrica", style="bold white")
    summary.add_column("Valor", justify="right")
    summary.add_row("Rows raw", f"{len(df_raw):,}")
    summary.add_row("Rows capa2", f"{len(df_l2):,}")
    summary.add_row("Parquets escritos", f"{files_written:,}")
    summary.add_row("Salida", str(out_dir))
    console.print(summary)
    console.print("[bold green]OK[/bold green] Capa 2 METEO completada")


if __name__ == "__main__":
    main()
