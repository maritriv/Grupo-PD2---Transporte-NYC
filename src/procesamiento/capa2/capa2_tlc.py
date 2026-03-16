from __future__ import annotations

import argparse
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Tuple

import pandas as pd
import pyarrow.parquet as pq
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    from config.settings import obtener_ruta  # type: ignore
except Exception:
    def obtener_ruta(p: str) -> Path:
        return Path(p)


console = Console()
DEBUG = False
MIN_YEAR = 2023
MAX_YEAR = 2025

NEEDED_COLS = [
    "tpep_pickup_datetime",
    "lpep_pickup_datetime",
    "pickup_datetime",
    "tpep_dropoff_datetime",
    "lpep_dropoff_datetime",
    "dropoff_datetime",
    "PULocationID",
    "pu_location_id",
    "DOLocationID",
    "do_location_id",
    "total_amount",
    "fare_amount",
    "tip_amount",
    "tips",
    "tolls_amount",
    "tolls",
    "airport_fee",
    "Airport_fee",
    "congestion_surcharge",
    "base_passenger_fare",
    "trip_distance",
    "trip_miles",
    "VendorID",
    "passenger_count",
    "RatecodeID",
    "payment_type",
]


def _list_parquets(folder: Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(folder.glob("*.parquet"))


def iter_validated_tlc_files(
    validated_base: Path,
    services: Iterable[str] = ("yellow", "green", "fhvhv"),
    clean_subdir: str = "clean",
) -> Iterator[Tuple[str, Path]]:
    validated_base = Path(validated_base).resolve()
    if not validated_base.exists():
        console.print(f"[bold yellow]WARNING[/bold yellow] No existe validated base: {validated_base}")
        return

    for service in services:
        folder = validated_base / service / clean_subdir
        files = _list_parquets(folder)

        if not files:
            console.print(f"[yellow]Aviso:[/yellow] No hay parquets en {folder}. Se omite {service}.")
            continue

        console.print(f"[cyan]{service}[/cyan]: {len(files)} parquets encontrados")
        for fp in files:
            yield service, fp


def _coalesce_cols(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series([pd.NA] * len(df), index=df.index)


def build_layer2_tlc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Sin df.copy() para ahorrar memoria
    df["pickup_datetime"] = _coalesce_cols(
        df,
        ["tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime"],
    )
    df["dropoff_datetime"] = _coalesce_cols(
        df,
        ["tpep_dropoff_datetime", "lpep_dropoff_datetime", "dropoff_datetime"],
    )

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"], errors="coerce")

    df["pu_location_id"] = pd.to_numeric(
        _coalesce_cols(df, ["PULocationID", "pu_location_id"]), errors="coerce"
    )
    df["do_location_id"] = pd.to_numeric(
        _coalesce_cols(df, ["DOLocationID", "do_location_id"]), errors="coerce"
    )

    num_cols = [
        "total_amount",
        "fare_amount",
        "tip_amount",
        "tips",
        "tolls_amount",
        "tolls",
        "airport_fee",
        "Airport_fee",
        "congestion_surcharge",
        "base_passenger_fare",
        "trip_distance",
        "trip_miles",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "airport_fee" not in df.columns and "Airport_fee" in df.columns:
        df["airport_fee"] = df["Airport_fee"]
    if "Airport_fee" in df.columns:
        df = df.drop(columns=["Airport_fee"])

    zero = pd.Series(0.0, index=df.index)

    airport_fee_any = df.get("airport_fee", zero).fillna(0)
    tips_any = df.get("tip_amount", zero).fillna(0)
    if "tips" in df.columns:
        tips_any = tips_any + df["tips"].fillna(0)

    tolls_any = df.get("tolls_amount", zero).fillna(0)
    if "tolls" in df.columns:
        tolls_any = tolls_any + df["tolls"].fillna(0)

    congestion_any = df.get("congestion_surcharge", zero).fillna(0)
    base_fare = df.get("base_passenger_fare", zero).fillna(0)

    if "total_amount" in df.columns:
        df["total_amount_std"] = df["total_amount"].fillna(
            base_fare + tips_any + tolls_any + airport_fee_any + congestion_any
        )
    else:
        df["total_amount_std"] = base_fare + tips_any + tolls_any + airport_fee_any + congestion_any

    df["date"] = df["pickup_datetime"].dt.date
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["hour"] = df["pickup_datetime"].dt.hour
    df = df[(df["year"] >= MIN_YEAR) & (df["year"] <= MAX_YEAR)]

    dow0 = df["pickup_datetime"].dt.dayofweek
    df["day_of_week"] = ((dow0 + 1) % 7) + 1
    df["is_weekend"] = df["day_of_week"].isin([1, 7]).astype("int8")
    df["week_of_year"] = df["pickup_datetime"].dt.isocalendar().week.astype("Int64")

    dur = (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds() / 60.0
    df["trip_duration_min"] = dur
    df.loc[(df["trip_duration_min"] < 0) | (df["trip_duration_min"] > 360), "trip_duration_min"] = pd.NA

    df = df.dropna(subset=["pickup_datetime", "date", "year", "month", "hour"])
    df = df[(df["hour"] >= 0) & (df["hour"] <= 23)]

    cols = [
        "service_type",
        "pickup_datetime",
        "dropoff_datetime",
        "date",
        "year",
        "month",
        "hour",
        "day_of_week",
        "is_weekend",
        "week_of_year",
        "trip_duration_min",
        "pu_location_id",
        "do_location_id",
        "total_amount_std",
        "total_amount",
        "fare_amount",
        "tip_amount",
        "tips",
        "tolls_amount",
        "tolls",
        "congestion_surcharge",
        "airport_fee",
        "base_passenger_fare",
        "trip_distance",
        "trip_miles",
        "VendorID",
        "passenger_count",
        "RatecodeID",
        "payment_type",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols]


def write_partitioned(df: pd.DataFrame, out_dir: Path):
    if df.empty:
        return

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for (svc, y, m), g in df.groupby(["service_type", "year", "month"], dropna=False):
        part_dir = out_dir / f"service_type={svc}" / f"year={int(y)}" / f"month={int(m)}"
        part_dir.mkdir(parents=True, exist_ok=True)
        fname = f"part_{uuid.uuid4().hex}.parquet"
        g.to_parquet(part_dir / fname, index=False, engine="pyarrow")


def get_existing_columns(fp: Path) -> list[str]:
    schema = pq.read_schema(fp)
    names = set(schema.names)
    return [c for c in NEEDED_COLS if c in names]


def iter_parquet_batches(fp: Path, batch_size: int = 200_000):
    cols = get_existing_columns(fp)
    pf = pq.ParquetFile(fp)
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        yield batch.to_pandas()


def main():
    console.print(Panel.fit("[bold cyan]CAPA 2 - TLC: ESTANDARIZACION + PARTICIONADO[/bold cyan]"))

    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default=str(obtener_ruta("data/validated")))
    p.add_argument("--out-dir", default=str(obtener_ruta("data/standarized")))
    p.add_argument("--from", dest="date_from", default=None)
    p.add_argument("--to", dest="date_to", default=None)
    p.add_argument("--mode", choices=["append", "overwrite"], default="append")
    args = p.parse_args()

    if args.date_from:
        datetime.strptime(args.date_from, "%Y-%m-%d")
    if args.date_to:
        datetime.strptime(args.date_to, "%Y-%m-%d")

    validated_base = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    cfg_table = Table(show_header=True, header_style="bold white", title="Configuracion Capa2 TLC")
    cfg_table.add_column("Campo", style="bold cyan")
    cfg_table.add_column("Valor")
    cfg_table.add_row("raw_dir", str(validated_base))
    cfg_table.add_row("out_dir", str(out_dir))
    cfg_table.add_row("mode", args.mode)
    cfg_table.add_row("filtro", f"{args.date_from or '...'} -> {args.date_to or '...'}")
    console.print(cfg_table)

    if args.mode == "overwrite" and out_dir.exists():
        console.print(f"[yellow]Limpiando salida (overwrite):[/yellow] {out_dir}")
        shutil.rmtree(out_dir)

    any_written = False
    service_stats: dict[str, dict[str, int]] = {}

    for service, fp in iter_validated_tlc_files(validated_base):
        if service not in service_stats:
            service_stats[service] = {"files": 0, "batches": 0, "rows_in": 0, "rows_out": 0}
        service_stats[service]["files"] += 1

        with console.status(f"[cyan]Procesando {service} -> {fp.name}[/cyan]"):
            for df in iter_parquet_batches(fp, batch_size=200_000):
                service_stats[service]["batches"] += 1
                service_stats[service]["rows_in"] += len(df)

                df["service_type"] = service
                df2 = build_layer2_tlc(df)

                if not df2.empty and (args.date_from or args.date_to):
                    dt = pd.to_datetime(df2["date"], errors="coerce")
                    if args.date_from:
                        df2 = df2[dt >= pd.to_datetime(args.date_from)]
                    if args.date_to:
                        df2 = df2[dt <= pd.to_datetime(args.date_to)]

                service_stats[service]["rows_out"] += len(df2)
                write_partitioned(df2, out_dir)
                any_written = any_written or (not df2.empty)

    summary = Table(show_header=True, header_style="bold magenta", title="Resumen Capa2 TLC por servicio")
    summary.add_column("Servicio", style="bold white")
    summary.add_column("Files", justify="right")
    summary.add_column("Batches", justify="right")
    summary.add_column("Rows in", justify="right")
    summary.add_column("Rows out", justify="right")

    total_files = total_batches = total_in = total_out = 0
    for service in sorted(service_stats.keys()):
        st = service_stats[service]
        total_files += st["files"]
        total_batches += st["batches"]
        total_in += st["rows_in"]
        total_out += st["rows_out"]
        summary.add_row(
            service,
            f"{st['files']:,}",
            f"{st['batches']:,}",
            f"{st['rows_in']:,}",
            f"{st['rows_out']:,}",
        )

    summary.add_row("TOTAL", f"{total_files:,}", f"{total_batches:,}", f"{total_in:,}", f"{total_out:,}")
    console.print(summary)

    if not any_written:
        console.print("[bold yellow]WARNING[/bold yellow] No se escribio nada.")
    else:
        console.print(f"[bold green]OK[/bold green] Capa 2 TLC guardada en: {out_dir}")


if __name__ == "__main__":
    main()
