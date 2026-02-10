# src/extraccion/download_events_data.py
from __future__ import annotations

import calendar
import itertools
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import click
import requests
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from config.settings import obtener_ruta, eventos_config

"""
download_events_data.py
│
├── Configuración
├── Helpers Socrata (HTTP / metadata)
├── Helpers Parquet (CSV → Parquet)
├── Lógica de negocio
│   ├── download_events_aggregated (interna)
│   ├── download_events_month       (unidad básica)
│   └── download_events_range       (orquestador)
└── CLI (Click)
"""


# =============================================================================
# Configuración
# =============================================================================
BASE_URL = eventos_config["url_base"]
DEFAULT_DATASET = eventos_config["dataset_id"]

SOCRATA_LIMIT = eventos_config["socrata_limit"]
TIMEOUT = eventos_config["timeout_segundos"]

DEFAULT_OUT_DIR = obtener_ruta("data/external/events/raw")

console = Console()


# =============================================================================
# Helpers Parquet
# =============================================================================
def csv_to_parquet(csv_path: Path, parquet_path: Path) -> None:
    """
    Convierte CSV a Parquet usando Pandas + PyArrow.
    Suficiente para archivos mensuales pequeños/moderados.
    """
    console.print(f"[dim]  → Convirtiendo CSV a Parquet...[/dim]")
    
    df = pd.read_csv(csv_path)
    
    # Conversión de tipos
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['hour'].astype('int32')
    df['n_events'] = df['n_events'].astype('int32')
    
    # Escribir parquet
    df.to_parquet(parquet_path, engine='pyarrow', index=False)
    
    console.print(f"[dim]  → Parquet generado[/dim]")


# =============================================================================
# Helpers Socrata
# =============================================================================
def _fetch_view_metadata(dataset_id: str) -> dict[str, Any]:
    """Obtiene metadatos del dataset de Socrata."""
    console.print(f"[dim]  → Obteniendo metadatos...[/dim]")
    
    url = f"{BASE_URL}/api/views/{dataset_id}"
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    
    return r.json()


def _pick_field(columns: list[dict[str, Any]], candidates: list[str]) -> Optional[str]:
    cand = [c.lower() for c in candidates]

    for col in columns:
        fn = (col.get("fieldName") or "").lower()
        if fn in cand:
            return col.get("fieldName")

    for col in columns:
        name = (col.get("name") or "").lower()
        if any(c in name for c in cand):
            return col.get("fieldName")

    return None


def _paged_socrata_json(
    resource_url: str,
    params: dict[str, Any],
    limit: int = SOCRATA_LIMIT,
) -> list[dict[str, Any]]:
    """
    Descarga datos de Socrata con paginación automática.
    """
    rows: list[dict[str, Any]] = []
    offset = 0

    console.print(f"[dim]  → Descargando datos agregados...[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[cyan]{task.fields[rows]} filas"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("[cyan]Descargando", total=None, rows=0)

        while True:
            p = dict(params, **{"$limit": limit, "$offset": offset})
            r = requests.get(resource_url, params=p, timeout=TIMEOUT)
            r.raise_for_status()

            batch = r.json()
            if not batch:
                break

            rows.extend(batch)
            offset += limit
            progress.update(task, rows=len(rows))

    console.print(f"[dim]  → {len(rows)} registros obtenidos[/dim]")
    return rows


# =============================================================================
# Lógica de negocio
# =============================================================================
def download_events_aggregated(
    dataset_id: str,
    date_from: str,
    date_to: str,
    out_dir: Path,
    tmp_name: str,
) -> Path:
    """
    Descarga eventos agregados en un rango de fechas.
    Devuelve la ruta del parquet generado.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_csv = out_dir / f"{tmp_name}.csv"
    tmp_parquet = out_dir / f"{tmp_name}.parquet"

    meta = _fetch_view_metadata(dataset_id)
    cols = meta.get("columns", [])

    start_field = _pick_field(cols, ["start_date_time", "start", "startdatetime"])
    borough_field = _pick_field(cols, ["borough"]) or "borough"
    type_field = _pick_field(cols, ["event_type", "event", "name"]) or "event_name"

    if not start_field:
        raise RuntimeError("No se pudo detectar la columna de fecha de inicio.")

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
        },
    )

    # CSV temporal
    console.print(f"[dim]  → Escribiendo CSV temporal...[/dim]")
    with open(tmp_csv, "w", encoding="utf-8") as f:
        f.write("date,hour,borough,event_type,n_events\n")
        for r in rows:
            f.write(
                f"{r.get('date','')[:10]},"
                f"{r.get('hour')},"
                f"{r.get('borough')},"
                f"{r.get('event_type')},"
                f"{r.get('n_events')}\n"
            )

    csv_to_parquet(tmp_csv, tmp_parquet)
    tmp_csv.unlink()

    return tmp_parquet


def download_events_month(
    dataset_id: str,
    year: int,
    month: int,
    out_dir: Path,
    tag: str = "events",
) -> dict[str, Any]:

    start = date(year, month, 1)
    end = date(year, month, calendar.monthrange(year, month)[1])

    out_dir.mkdir(parents=True, exist_ok=True)
    final_parquet = out_dir / f"{tag}_{year}_{month:02d}.parquet"

    console.print(f"\n[bold cyan]{year}-{month:02d}[/bold cyan]")

    if final_parquet.exists():
        console.print(f"[yellow]SKIP:[/yellow] {final_parquet.name}")
        return {"status": "skipped", "path": str(final_parquet)}

    tmp_name = f"__tmp_{tag}_{year}_{month:02d}"
    tmp_parquet = download_events_aggregated(
        dataset_id,
        start.isoformat(),
        end.isoformat(),
        out_dir,
        tmp_name,
    )

    tmp_parquet.rename(final_parquet)
    console.print(f"[green]OK:[/green] {final_parquet.name}")
    
    return {"status": "ok", "path": str(final_parquet)}


def download_events_range(
    dataset_id: str,
    start_year: int,
    end_year: int,
    start_month: int = 1,
    end_month: int = 12,
    out_dir: Optional[Path] = None,
) -> dict[str, int]:

    out_dir = out_dir or DEFAULT_OUT_DIR
    years = range(start_year, end_year + 1)
    months = range(start_month, end_month + 1)

    stats = {"total": 0, "ok": 0, "skipped": 0, "failed": 0}

    console.print(f"\n[bold]Descargando eventos NYC[/bold]")
    console.print(f"[yellow]Período:[/yellow] {start_year}/{start_month:02d} → {end_year}/{end_month:02d}")
    console.print(f"[yellow]Dataset:[/yellow] {dataset_id}")
    console.print(f"[yellow]Destino:[/yellow] {out_dir}")
    console.rule(style="dim")

    for year, month in itertools.product(years, months):
        stats["total"] += 1
        try:
            r = download_events_month(dataset_id, year, month, out_dir)
            stats[r["status"]] += 1
        except Exception as e:
            console.print(f"[red]ERROR:[/red] {str(e)}")
            stats["failed"] += 1

    console.print()
    console.rule(style="dim")
    console.print(f"[green]Descargados:[/green] {stats['ok']} | "
                  f"[blue]Omitidos:[/blue] {stats['skipped']} | "
                  f"[red]Fallidos:[/red] {stats['failed']} | "
                  f"[bold]Total:[/bold] {stats['total']}")
    console.print()

    return stats


# =============================================================================
# CLI
# =============================================================================
@click.command()
@click.option("--dataset", default=DEFAULT_DATASET, show_default=True)
@click.option("--start-year", type=int, required=True)
@click.option("--end-year", type=int, required=True)
@click.option("--start-month", type=click.IntRange(1, 12), default=1)
@click.option("--end-month", type=click.IntRange(1, 12), default=12)
def main(dataset: str, start_year: int, end_year: int, start_month: int, end_month: int):
    """
    Descarga eventos de NYC Open Data (Socrata) de forma particionada por mes.

    Ejemplos de uso:

        # Descargar todos los eventos de 2024
        uv run -m src.extraccion.download_events_data \
            --start-year 2024 \
            --end-year 2024

        # Descargar eventos de junio a agosto de 2023
        uv run -m src.extraccion.download_events_data \
            --start-year 2023 \
            --end-year 2023 \
            --start-month 6 \
            --end-month 8

        # Usar un dataset concreto de NYC Open Data
        uv run -m src.extraccion.download_events_data \
            --dataset bkfu-528j \
            --start-year 2022 \
            --end-year 2023
    """
    download_events_range(
        dataset_id=dataset,
        start_year=start_year,
        end_year=end_year,
        start_month=start_month,
        end_month=end_month,
    )


if __name__ == "__main__":
    main()