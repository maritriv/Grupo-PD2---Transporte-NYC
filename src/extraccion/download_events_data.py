# src/extraccion/download_events_data.py
from __future__ import annotations
import findspark
findspark.init()

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pyspark.sql import SparkSession, functions as F

from config.settings import obtener_ruta, eventos_config


# =============================
# Configuración base (desde YAML)
# =============================
BASE_URL = eventos_config["url_base"]
DEFAULT_DATASET = eventos_config["dataset_id"]
DEFAULT_FROM = eventos_config["date_from"]
DEFAULT_TO = eventos_config["date_to"]

SOCRATA_LIMIT = eventos_config["socrata_limit"]
TIMEOUT = eventos_config["timeout_segundos"]

DEFAULT_OUT_DIR = obtener_ruta("data/external/events")
DEFAULT_OUT_NAME = "events_daily_borough_type"

console = Console()

# -----------------------------
# Helpers Spark
# -----------------------------
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
    url = f"{BASE_URL}/api/views/{dataset_id}"
    r = requests.get(url, timeout=TIMEOUT)
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


def _paged_socrata_json(
    resource_url: str,
    params: dict[str, Any],
    limit: int = SOCRATA_LIMIT
) -> list[dict[str, Any]]:
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
    out_dir: Path = DEFAULT_OUT_DIR,
    out_name: str = DEFAULT_OUT_NAME,
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

    console.print(f"\n[bold]Descargando eventos NYC[/bold]")
    console.print(f"[yellow]Dataset:[/yellow] {dataset_id}")
    console.print(f"[yellow]Periodo:[/yellow] {date_from} → {date_to}")
    console.print(f"[yellow]Destino:[/yellow] {out_dir}")
    console.rule(style="dim")

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
        f"date_extract_hh({start_field}) as hour,"
        f"{borough_field} as borough,"
        f"{type_field} as event_type,"
        f"count(1) as n_events"
    )

    where = (
        f"{start_field} between "
        f"'{date_from}T00:00:00.000' and '{date_to}T23:59:59.999'"
    )

    group = (
        f"date_trunc_ymd({start_field}), "
        f"date_extract_hh({start_field}), "
        f"{borough_field}, {type_field}"
    )

    url = f"{BASE_URL}/resource/{dataset_id}.json"
    rows = _paged_socrata_json(
        url,
        {
            "$select": select,
            "$where": where,
            "$group": group,
            "$order": "date asc, hour asc",
        }
    )

    # -------------------------
    # Guardar CSV
    # -------------------------
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "hour","borough", "event_type", "n_events"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "date": (r.get("date") or "")[:10],   # YYYY-MM-DD
                "hour": r.get("hour"),
                "borough": r.get("borough"),
                "event_type": r.get("event_type"),
                "n_events": r.get("n_events"),
            })

    # -------------------------
    # Guardar Parquet
    # -------------------------
    parquet_ok = True
    try:
        csv_to_parquet(out_csv, out_parquet)
    except Exception as e:
        parquet_ok = False
        console.print(f"[yellow]WARN:[/yellow] No se pudo generar Parquet: {e}")

    return {
        "dataset_id": dataset_id,
        "from": date_from,
        "to": date_to,
        "rows": len(rows),
        "out_csv": str(out_csv),
        "out_parquet": str(out_parquet) if parquet_ok else None,
    }


# =============================
# CLI (Click)
# =============================
@click.command()
@click.option(
    "--dataset",
    default=DEFAULT_DATASET,
    show_default=True,
    help="ID del dataset de NYC Open Data (Socrata)"
)
@click.option(
    "--date-from",
    default=DEFAULT_FROM,
    show_default=True,
    help="Fecha inicial (YYYY-MM-DD)"
)
@click.option(
    "--date-to",
    default=DEFAULT_TO,
    show_default=True,
    help="Fecha final (YYYY-MM-DD)"
)
@click.option(
    "--name",
    "out_name",
    default=DEFAULT_OUT_NAME,
    show_default=True,
    help="Nombre base de los archivos de salida"
)
def main(dataset: str, date_from: str, date_to: str, out_name: str):
    """
    Descarga eventos de NYC Open Data agregados por día, hora, borough y tipo.

    Ejemplos de uso:

        # Usando valores por defecto definidos en config.yaml
        uv run -m src.extraccion.download_events_data

        # Indicando un dataset concreto de NYC Open Data
        uv run -m src.extraccion.download_events_data --dataset bkfu-528j

        # Descargando un rango de fechas específico
        uv run -m src.extraccion.download_events_data \
            --date-from 2024-06-01 \
            --date-to 2024-06-30

        # Cambiando el nombre base de los archivos de salida
        uv run -m src.extraccion.download_events_data \
            --name events_june_2024
    """


    # Validación de fechas
    try:
        datetime.strptime(date_from, "%Y-%m-%d")
        datetime.strptime(date_to, "%Y-%m-%d")
    except ValueError:
        raise click.BadParameter("Formato de fecha inválido (usa YYYY-MM-DD)")

    stats = download_events_aggregated(
        dataset_id=dataset,
        date_from=date_from,
        date_to=date_to,
        out_name=out_name,
    )

    table = Table(
        title="Resumen descarga eventos",
        show_header=True,
        header_style="bold cyan"
    )
    table.add_column("Campo", style="cyan", width=18)
    table.add_column("Valor", style="white")

    for k, v in stats.items():
        table.add_row(k, str(v))

    console.print()
    console.print(table)
    console.print()
    console.print(
        Panel.fit(
            "[bold green]DESCARGA EVENTOS COMPLETADA[/bold green]\n"
            f"[dim]Filas agregadas: {stats['rows']}[/dim]",
            border_style="bright_green"
        )
    )


if __name__ == "__main__":
    main()
