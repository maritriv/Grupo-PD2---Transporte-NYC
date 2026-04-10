from __future__ import annotations

import calendar
from pathlib import Path
from typing import Any

import click
import pandas as pd
import requests
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from config.settings import config, obtener_ruta


BASE_URL = (
    config.get("restaurants", {}).get("url_base")
    or "https://data.cityofnewyork.us"
)
DEFAULT_DATASET = (
    config.get("restaurants", {}).get("dataset_id")
    or "43nn-pn8j"
)
DEFAULT_START_YEAR = 2023
DEFAULT_END_YEAR = 2025
SOCRATA_LIMIT = int(
    config.get("restaurants", {}).get("socrata_limit")
    or 10000
)
TIMEOUT = int(
    config.get("restaurants", {}).get("timeout_segundos")
    or 60
)
DEFAULT_OUT_DIR = obtener_ruta("data/external/restaurants/raw")

RAW_FIELDS = [
    "camis",
    "dba",
    "boro",
    "building",
    "street",
    "zipcode",
    "phone",
    "cuisine_description",
    "inspection_date",
    "action",
    "violation_code",
    "violation_description",
    "critical_flag",
    "score",
    "grade",
    "grade_date",
    "record_date",
    "inspection_type",
    "latitude",
    "longitude",
    "community_board",
    "council_district",
    "census_tract",
    "bin",
    "bbl",
    "nta",
]
REQUIRED_FIELDS = {"camis", "inspection_date", "boro"}
DATE_FIELDS = ["inspection_date", "grade_date", "record_date"]
FLOAT_FIELDS = ["latitude", "longitude"]
INT_FIELDS = ["score"]

BOROUGH_MAP = {
    "1": "Manhattan",
    "2": "Bronx",
    "3": "Brooklyn",
    "4": "Queens",
    "5": "Staten Island",
    "MANHATTAN": "Manhattan",
    "BRONX": "Bronx",
    "BROOKLYN": "Brooklyn",
    "QUEENS": "Queens",
    "STATEN ISLAND": "Staten Island",
}

console = Console()


def _fetch_view_metadata(dataset_id: str) -> dict[str, Any]:
    url = f"{BASE_URL}/api/views/{dataset_id}"
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def _paged_socrata_json(
    dataset_id: str,
    params: dict[str, Any],
    limit: int = SOCRATA_LIMIT,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    url = f"{BASE_URL}/resource/{dataset_id}.json"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[cyan]{task.fields[rows]} filas"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Descargando inspecciones", total=None, rows=0)

        while True:
            response = requests.get(
                url,
                params=dict(params, **{"$limit": limit, "$offset": offset}),
                timeout=TIMEOUT,
            )
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break

            rows.extend(batch)
            offset += limit
            progress.update(task, rows=len(rows))

    return rows


def _normalize_borough(series: pd.Series) -> pd.Series:
    clean = series.astype("string").str.strip()
    clean = clean.replace({"": pd.NA, "0": pd.NA})
    upper = clean.str.upper()
    mapped = upper.map(BOROUGH_MAP)
    return mapped.fillna(clean.str.title())


def _clean_restaurants_df(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0

    df = df.copy()

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    for col in DATE_FIELDS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in FLOAT_FIELDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in INT_FIELDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    if "boro" in df.columns:
        df["boro"] = _normalize_borough(df["boro"])

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)

    return df, removed


def _month_filename(year: int, month: int) -> str:
    return f"restaurants_{year}_{month:02d}.parquet"


def download_restaurants_month(
    year: int,
    month: int,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    dataset_id: str = DEFAULT_DATASET,
    force: bool = False,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / _month_filename(year, month)
    if parquet_path.exists() and not force:
        console.print(f"[yellow]SKIP[/yellow] {parquet_path.name} ya existe")
        return {"status": "skipped", "rows": 0, "deduped": 0, "path": str(parquet_path)}

    month_last_day = calendar.monthrange(year, month)[1]
    date_from = f"{year}-{month:02d}-01"
    date_to = f"{year}-{month:02d}-{month_last_day:02d}"

    console.print(f"[bold]Restaurantes {year}-{month:02d}[/bold]")
    console.print("[dim]  -> Consultando metadatos del dataset...[/dim]")
    metadata = _fetch_view_metadata(dataset_id)
    available_fields = {
        col.get("fieldName")
        for col in metadata.get("columns", [])
        if col.get("fieldName")
    }
    selected_fields = [field for field in RAW_FIELDS if field in available_fields]

    missing_required = REQUIRED_FIELDS - set(selected_fields)
    if missing_required:
        missing = ", ".join(sorted(missing_required))
        raise RuntimeError(f"Faltan columnas requeridas en el dataset oficial: {missing}")

    select = ",".join(selected_fields)
    where = (
        f"inspection_date between '{date_from}T00:00:00.000' "
        f"and '{date_to}T23:59:59.999'"
    )
    order_fields = [field for field in ["inspection_date", "camis", "violation_code"] if field in selected_fields]
    params = {
        "$select": select,
        "$where": where,
        "$order": ", ".join(f"{field} asc" for field in order_fields),
    }

    rows = _paged_socrata_json(dataset_id, params=params)
    raw_rows = len(rows)
    if not rows:
        empty_df = pd.DataFrame(columns=selected_fields)
        empty_df.to_parquet(parquet_path, engine="pyarrow", index=False)
        console.print(f"[yellow]Sin registros para {year}-{month:02d}[/yellow]")
        return {"status": "ok", "rows": 0, "deduped": 0, "path": str(parquet_path)}

    df = pd.DataFrame(rows)
    df = df.reindex(columns=selected_fields)
    df, deduped = _clean_restaurants_df(df)
    df.to_parquet(parquet_path, engine="pyarrow", index=False)

    console.print(
        f"[green]OK[/green] {parquet_path.name} | "
        f"filas_raw={raw_rows} filas_finales={len(df)} exact_dup_rm={deduped}"
    )
    return {
        "status": "ok",
        "rows": len(df),
        "raw_rows": raw_rows,
        "deduped": deduped,
        "path": str(parquet_path),
    }


def download_restaurants_range(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    start_month: int = 1,
    end_month: int = 12,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    dataset_id: str = DEFAULT_DATASET,
    force: bool = False,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    stats = {"successful": 0, "skipped": 0, "failed": 0, "total": 0}

    for year in range(start_year, end_year + 1):
        month_from = start_month if year == start_year else 1
        month_to = end_month if year == end_year else 12

        for month in range(month_from, month_to + 1):
            stats["total"] += 1
            try:
                result = download_restaurants_month(
                    year=year,
                    month=month,
                    out_dir=out_dir,
                    dataset_id=dataset_id,
                    force=force,
                )
                if result["status"] == "skipped":
                    stats["skipped"] += 1
                else:
                    stats["successful"] += 1
            except Exception as exc:
                stats["failed"] += 1
                console.print(
                    f"[bold red]ERROR[/bold red] restaurantes {year}-{month:02d}: {exc}"
                )

    return stats


@click.command()
@click.option("--start-year", default=DEFAULT_START_YEAR, type=int, show_default=True)
@click.option("--end-year", default=DEFAULT_END_YEAR, type=int, show_default=True)
@click.option("--start-month", default=1, type=int, show_default=True)
@click.option("--end-month", default=12, type=int, show_default=True)
@click.option("--dataset-id", default=DEFAULT_DATASET, show_default=True)
@click.option(
    "--out-dir",
    default=str(DEFAULT_OUT_DIR),
    show_default=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
@click.option("--force", is_flag=True, help="Sobrescribe los parquet ya existentes.")
def main(
    start_year: int,
    end_year: int,
    start_month: int,
    end_month: int,
    dataset_id: str,
    out_dir: Path,
    force: bool,
) -> None:
    stats = download_restaurants_range(
        start_year=start_year,
        end_year=end_year,
        start_month=start_month,
        end_month=end_month,
        dataset_id=dataset_id,
        out_dir=out_dir,
        force=force,
    )
    console.print(stats)


if __name__ == "__main__":
    main()
