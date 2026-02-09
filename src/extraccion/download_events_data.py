# src/extraccion/download_events_data.py
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pyspark.sql import SparkSession, functions as F


# Intentamos usar vuestro sistema de config/rutas
try:
    from config.settings import obtener_ruta, config  # type: ignore
except Exception:  # fallback si alguien ejecuta sin ese módulo
    config = {}
    def obtener_ruta(p: str) -> Path:
        return Path(p)

console = Console()
BASE = "https://data.cityofnewyork.us"

def csv_to_parquet(csv_path: Path, parquet_path: Path):
    spark = (
        SparkSession.builder
        .appName("EventsCSVtoParquet")
        .master("local[*]")
        .getOrCreate()
    )

    df = spark.read.option("header", True).csv(str(csv_path))

    # Tipos: date como timestamp y n_events como int
    df = (
        df.withColumn("date", F.to_timestamp("date"))
          .withColumn("n_events", F.col("n_events").cast("int"))
    )

    df.write.mode("overwrite").parquet(str(parquet_path))
    spark.stop()

# -----------------------------
# Helpers Socrata
# -----------------------------
def _fetch_view_metadata(dataset_id: str) -> dict[str, Any]:
    url = f"{BASE}/api/views/{dataset_id}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def _pick_field(columns: list[dict[str, Any]], candidates: list[str]) -> str | None:
    cand = [c.lower() for c in candidates]

    # 1) match exacto por fieldName
    for col in columns:
        fn = (col.get("fieldName") or "").lower()
        if fn in cand:
            return col.get("fieldName")

    # 2) match “contiene” por name humano
    for col in columns:
        nm = (col.get("name") or "").lower()
        for c in cand:
            if c in nm:
                return col.get("fieldName")

    return None


def _paged_socrata_json(resource_url: str, params: dict[str, Any], limit: int = 50000) -> list[dict[str, Any]]:
    """
    Socrata suele limitar resultados. Esto pagina con $limit/$offset y devuelve lista de dicts.
    """
    out: list[dict[str, Any]] = []
    offset = 0

    while True:
        p = dict(params)
        p["$limit"] = limit
        p["$offset"] = offset

        r = requests.get(resource_url, params=p, timeout=300)
        r.raise_for_status()
        batch = r.json()

        if not batch:
            break

        out.extend(batch)
        offset += limit

    return out


# -----------------------------
# Descarga eventos agregados
# -----------------------------
def download_events_aggregated(
    dataset_id: str,
    date_from: str,
    date_to: str,
    out_dir: Path,
    out_name: str = "events_daily_borough_type",
) -> dict[str, Any]:
    """
    Descarga agregado por:
      - date (día)
      - borough
      - event_type
      - n_events

    Guarda:
      - CSV (siempre)
      - Parquet (si Spark funciona)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{out_name}.csv"
    out_parquet = out_dir / f"{out_name}.parquet"

    meta = _fetch_view_metadata(dataset_id)
    cols = meta.get("columns", [])

    start_field = _pick_field(cols, ["start_date_time", "startdatetime", "start date/time", "start date", "start"])
    borough_field = _pick_field(cols, ["borough"])
    type_field = _pick_field(cols, ["event_type", "event type", "eventtype", "event_name", "event name", "name", "event"])

    if not start_field:
        raise RuntimeError("No pude detectar columna de inicio (start_...). Cambia dataset o ajusta candidates.")

    borough_field = borough_field or "borough"
    type_field = type_field or "event_name"

    select = (
        f"date_trunc_ymd({start_field}) as date,"
        f"{borough_field} as borough,"
        f"{type_field} as event_type,"
        f"count(1) as n_events"
    )
    where = (
        f"{start_field} between "
        f"'{date_from}T00:00:00.000' and '{date_to}T23:59:59.999'"
    )
    group = f"date_trunc_ymd({start_field}), {borough_field}, {type_field}"
    order = "date asc"

    url = f"{BASE}/resource/{dataset_id}.json"
    params = {"$select": select, "$where": where, "$group": group, "$order": order}

    rows = _paged_socrata_json(url, params=params, limit=50000)

    # -------------------------
    # Guardado CSV (siempre)
    # -------------------------
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "borough", "event_type", "n_events"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "date": (r.get("date") or "")[:10],   # YYYY-MM-DD
                "borough": r.get("borough"),
                "event_type": r.get("event_type"),
                "n_events": r.get("n_events"),
            })

    # -------------------------
    # Guardado Parquet (si Spark funciona)
    # -------------------------
    parquet_ok = True
    try:
        spark = (
            SparkSession.builder
            .appName("PD2-Events-Parquet")
            .master("local[*]")
            .getOrCreate()
        )

        data = []
        for r in rows:
            d = (r.get("date") or "")[:10]
            b = r.get("borough")
            t = r.get("event_type")
            n = r.get("n_events")
            try:
                n = int(n) if n is not None else None
            except Exception:
                n = None
            data.append((d, b, t, n))

        dfp = spark.createDataFrame(data, ["date", "borough", "event_type", "n_events"])
        dfp = dfp.withColumn("date", F.to_date("date"))

        dfp.write.mode("overwrite").parquet(str(out_parquet))
        spark.stop()

    except Exception as e:
        parquet_ok = False
        console.print(f"[yellow][WARN][/yellow] No se pudo guardar Parquet (se deja el CSV): {e}")

    return {
        "dataset_id": dataset_id,
        "from": date_from,
        "to": date_to,
        "rows": len(rows),
        "out_csv": str(out_csv),
        "out_parquet": str(out_parquet) if parquet_ok else None,
    }



# -----------------------------
# CLI
# -----------------------------
def main():
    # Defaults desde config.yaml si existe sección "eventos"
    eventos_cfg = (config.get("eventos") or {}) if isinstance(config, dict) else {}

    default_dataset = eventos_cfg.get("dataset_id", "bkfu-528j")  # pon aquí el id que decidáis como estándar
    default_from = eventos_cfg.get("date_from", "2024-01-01")
    default_to = eventos_cfg.get("date_to", "2025-12-31")
    default_out_name = eventos_cfg.get("out_name", "events_daily_borough_type")

    p = argparse.ArgumentParser(description="Descarga eventos NYC Open Data (agregado por día/borough/tipo).")
    p.add_argument("--dataset", default=default_dataset, help="Socrata dataset id (ej: bkfu-528j)")
    p.add_argument("--from", dest="date_from", default=default_from, help="YYYY-MM-DD")
    p.add_argument("--to", dest="date_to", default=default_to, help="YYYY-MM-DD")
    p.add_argument("--name", dest="out_name", default=default_out_name, help="Nombre base del archivo de salida")
    args = p.parse_args()

    # Validación simple de fechas
    try:
        datetime.strptime(args.date_from, "%Y-%m-%d")
        datetime.strptime(args.date_to, "%Y-%m-%d")
    except ValueError:
        raise SystemExit("❌ Fechas inválidas. Usa formato YYYY-MM-DD")

    out_dir = obtener_ruta("data/external/events")

    console.print()
    console.print(Panel.fit(
        "[bold white]NYC EVENTS (Open Data)[/bold white]\n"
        "[cyan]Descarga y agregado por día/borough/tipo[/cyan]",
        border_style="bright_blue"
    ))
    console.print(f"[yellow]Dataset:[/yellow] {args.dataset}")
    console.print(f"[yellow]Periodo:[/yellow] {args.date_from} -> {args.date_to}")
    console.print(f"[yellow]Destino:[/yellow] {out_dir}")
    console.print()

    stats = download_events_aggregated(
        dataset_id=args.dataset,
        date_from=args.date_from,
        date_to=args.date_to,
        out_dir=Path(out_dir),
        out_name=args.out_name,
    )

    # Resumen tipo tabla (como vuestro estilo)
    table = Table(title="Resumen descarga eventos", show_header=True, header_style="bold cyan")
    table.add_column("Campo", style="cyan", width=18)
    table.add_column("Valor", style="white")

    table.add_row("dataset_id", str(stats["dataset_id"]))
    table.add_row("from", str(stats["from"]))
    table.add_row("to", str(stats["to"]))
    table.add_row("rows", str(stats["rows"]))
    table.add_row("out_csv", str(stats["out_csv"]))

    console.print(table)
    console.print()
    console.print(Panel.fit(
        "[bold green]DESCARGA EVENTOS COMPLETADA[/bold green]\n"
        f"[dim]Filas agregadas: {stats['rows']}[/dim]",
        border_style="bright_green"
    ))


if __name__ == "__main__":
    main()
