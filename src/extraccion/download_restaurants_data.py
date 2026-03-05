# src/extraccion/download_restaurants_data.py
from __future__ import annotations

import calendar
import itertools
from datetime import date
from pathlib import Path
from typing import Any, Optional

import click
import pandas as pd
import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from config.settings import obtener_ruta

"""
download_restaurants_data.py
│
├── Configuración
├── Helpers Socrata (HTTP / metadata)
├── Helpers Parquet (CSV → Parquet)
├── Lógica de negocio
│   ├── download_restaurants_raw (interna)
│   ├── download_restaurants_month (unidad básica)
│   └── download_restaurants_range (orquestador)
└── CLI (Click)
"""

# =============================================================================
# Configuración
# =============================================================================
BASE_URL = "https://data.cityofnewyork.us"
DEFAULT_DATASET = "43nn-pn8j"  # DOHMH New York City Restaurant Inspection Results

SOCRATA_LIMIT = 50000
TIMEOUT = 60

DEFAULT_OUT_DIR = obtener_ruta("data/external/restaurants/raw")

console = Console()


# =============================================================================
# Helpers Parquet
# =============================================================================
def csv_to_parquet(csv_path: Path, parquet_path: Path) -> None:
    """Convierte CSV a Parquet usando Pandas + PyArrow."""
    console.print("[dim]  → Convirtiendo CSV a Parquet...[/dim]")
    df = pd.read_csv(csv_path)

    # Conversión de tipos típicos (si existen)
    for col in ["inspection_date", "record_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in ["score", "zipcode"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    console.print("[dim]  → Parquet generado[/dim]")


# =============================================================================
# Helpers Socrata
# =============================================================================
def _fetch_view_metadata(dataset_id: str) -> dict[str, Any]:
    """Obtiene metadatos del dataset de Socrata."""
    console.print("[dim]  → Obteniendo metadatos...[/dim]")
    url = f"{BASE_URL}/api/views/{dataset_id}"
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _pick_field(columns: list[dict[str, Any]], candidates: list[str]) -> Optional[str]:
    """Intenta detectar un campo por fieldName o name."""
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
    """Descarga datos de Socrata con paginación automática."""
    rows: list[dict[str, Any]] = []
    offset = 0

    console.print("[dim]  → Descargando datos...[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[cyan]{task.fields[rows]} filas"),
        console=console,
        transient=True,
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
def download_restaurants_raw(
    dataset_id: str,
    date_from: str,
    date_to: str,
    out_dir: Path,
    tmp_name: str,
) -> Path:
    """
    Descarga registros (raw) del dataset en un rango de fechas.
    Devuelve la ruta del parquet generado.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_csv = out_dir / f"{tmp_name}.csv"
    tmp_parquet = out_dir / f"{tmp_name}.parquet"

    meta = _fetch_view_metadata(dataset_id)
    cols = meta.get("columns", [])

    # Campos principales (intentamos autodetectar)
    inspection_date = _pick_field(cols, ["inspection_date", "inspection date"])
    if not inspection_date:
        raise RuntimeError("No se pudo detectar la columna de inspection_date.")

    # Campos “útiles” (si no se detectan, se omiten)
    camis = _pick_field(cols, ["camis"])
    dba = _pick_field(cols, ["dba", "restaurant_name", "restaurant name"])
    boro = _pick_field(cols, ["boro", "borough"])
    cuisine = _pick_field(cols, ["cuisine_description", "cuisine", "cuisine description"])
    grade = _pick_field(cols, ["grade"])
    score = _pick_field(cols, ["score"])
    street = _pick_field(cols, ["street"])
    building = _pick_field(cols, ["building"])
    zipcode = _pick_field(cols, ["zipcode", "zip"])
    phone = _pick_field(cols, ["phone"])
    lat = _pick_field(cols, ["latitude", "lat"])
    lon = _pick_field(cols, ["longitude", "lon"])
    record_date = _pick_field(cols, ["record_date", "record date"])

    fields = [
        inspection_date,
        camis,
        dba,
        boro,
        cuisine,
        grade,
        score,
        street,
        building,
        zipcode,
        phone,
        lat,
        lon,
        record_date,
    ]
    fields = [f for f in fields if f]  # quitar None

    select = ", ".join(fields)

    where = (
        f"{inspection_date} between "
        f"'{date_from}T00:00:00.000' and '{date_to}T23:59:59.999'"
    )

    url = f"{BASE_URL}/resource/{dataset_id}.json"
    rows = _paged_socrata_json(
        url,
        {
            "$select": select,
            "$where": where,
            "$order": f"{inspection_date} asc",
        },
    )

    # CSV temporal (simple y robusto)
    console.print("[dim]  → Escribiendo CSV temporal...[/dim]")
    df = pd.DataFrame(rows)
    df.to_csv(tmp_csv, index=False, encoding="utf-8")

    csv_to_parquet(tmp_csv, tmp_parquet)
    tmp_csv.unlink()

    return tmp_parquet


def download_restaurants_month(
    dataset_id: str,
    year: int,
    month: int,
    out_dir: Path,
    tag: str = "restaurants",
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
    tmp_parquet = download_restaurants_raw(
        dataset_id=dataset_id,
        date_from=start.isoformat(),
        date_to=end.isoformat(),
        out_dir=out_dir,
        tmp_name=tmp_name,
    )

    tmp_parquet.rename(final_parquet)
    console.print(f"[green]OK:[/green] {final_parquet.name}")
    return {"status": "ok", "path": str(final_parquet)}


def download_restaurants_range(
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

    console.print("\n[bold]Descargando Restaurant Inspection Results (NYC)[/bold]")
    console.print(
        f"[yellow]Período:[/yellow] {start_year}/{start_month:02d} → {end_year}/{end_month:02d}"
    )
    console.print(f"[yellow]Dataset:[/yellow] {dataset_id}")
    console.print(f"[yellow]Destino:[/yellow] {out_dir}")
    console.rule(style="dim")

    for year, month in itertools.product(years, months):
        stats["total"] += 1
        try:
            r = download_restaurants_month(dataset_id, year, month, out_dir)
            stats[r["status"]] += 1
        except Exception as e:
            console.print(f"[red]ERROR:[/red] {str(e)}")
            stats["failed"] += 1

    console.print()
    console.rule(style="dim")
    console.print(
        f"[green]Descargados:[/green] {stats['ok']} | "
        f"[blue]Omitidos:[/blue] {stats['skipped']} | "
        f"[red]Fallidos:[/red] {stats['failed']} | "
        f"[bold]Total:[/bold] {stats['total']}"
    )
    console.print()

    return stats


# =============================================================================
# CLI
# =============================================================================
@click.command()
@click.option("--dataset", default=DEFAULT_DATASET, show_default=True)
@click.option("--start-year", type=int, default=2023, show_default=True, help="Año inicial")
@click.option("--end-year", type=int, default=2025, show_default=True, help="Año final")
@click.option("--start-month", type=click.IntRange(1, 12), default=1)
@click.option("--end-month", type=click.IntRange(1, 12), default=12)
def main(dataset: str, start_year: int, end_year: int, start_month: int, end_month: int):
    """
    Descarga el dataset de inspecciones de restaurantes de NYC (Socrata),
    particionado por mes según inspection_date.

    Ejemplos:

        uv run -m src.extraccion.download_restaurants_data --start-year 2024 --end-year 2024
        uv run -m src.extraccion.download_restaurants_data --start-year 2023 --end-year 2023 --start-month 6 --end-month 8
        uv run -m src.extraccion.download_restaurants_data --dataset 43nn-pn8j --start-year 2022 --end-year 2023
    """
    download_restaurants_range(
        dataset_id=dataset,
        start_year=start_year,
        end_year=end_year,
        start_month=start_month,
        end_month=end_month,
    )


if __name__ == "__main__":
    main()