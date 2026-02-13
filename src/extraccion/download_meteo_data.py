# src/extraccion/download_meteo_data.py
from __future__ import annotations

import calendar
import itertools
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

import click
import requests
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from config.settings import obtener_ruta, meteo_config

"""
download_meteo_data.py
│
├── Configuración
├── Helpers Open-Meteo (HTTP / chunking)
├── Helpers Parquet (conversión)
├── Lógica de negocio
│   ├── download_meteo_aggregated (interna)
│   ├── download_meteo_month       (unidad básica)
│   └── download_meteo_range       (orquestador)
└── CLI (Click)
"""


# =============================================================================
# Configuración
# =============================================================================
BASE_URL = meteo_config['url_base']
DEFAULT_OUT_DIR = obtener_ruta("data/external/meteo/raw")

console = Console()


# =============================================================================
# Helpers Parquet
# =============================================================================
def dataframe_to_parquet(df: pd.DataFrame, parquet_path: Path) -> None:
    """
    Convierte DataFrame a Parquet usando Pandas + PyArrow.
    """
    console.print(f"[dim]  → Escribiendo Parquet...[/dim]")
    
    # Conversión de tipos
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['hour'].astype('int32')
    
    # Convertir columnas numéricas
    numeric_cols = ['temp_c', 'precip_mm', 'rain_mm', 'snowfall_mm', 'wind_kmh', 'weather_code']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Escribir parquet
    df.to_parquet(parquet_path, engine='pyarrow', index=False)
    
    console.print(f"[dim]  → Parquet generado[/dim]")


# =============================================================================
# Helpers Open-Meteo
# =============================================================================
def _date_chunks(date_from: str, date_to: str, chunk_days: int = 31) -> Iterable[tuple[str, str]]:
    """
    Divide un rango [date_from, date_to] en trozos de chunk_days.
    Devuelve tuplas (start_date, end_date) en formato YYYY-MM-DD.
    """
    start = datetime.strptime(date_from, "%Y-%m-%d").date()
    end = datetime.strptime(date_to, "%Y-%m-%d").date()

    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=chunk_days - 1), end)
        yield (cur.isoformat(), nxt.isoformat())
        cur = nxt + timedelta(days=1)


def _fetch_open_meteo_hourly(
    latitude: float,
    longitude: float,
    date_from: str,
    date_to: str,
    timezone: str,
    hourly_vars: list[str],
    timeout: int = meteo_config['timeout'],
) -> dict[str, Any]:
    """
    Realiza una petición a Open-Meteo Archive API.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date_from,
        "end_date": date_to,
        "timezone": timezone,
        "hourly": ",".join(hourly_vars),
    }
    r = requests.get(BASE_URL, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


# =============================================================================
# Lógica de negocio
# =============================================================================
def download_meteo_aggregated(
    date_from: str,
    date_to: str,
    out_dir: Path,
    tmp_name: str,
    latitude: float,
    longitude: float,
    timezone: str,
) -> Path:
    """
    Descarga datos meteorológicos agregados en un rango de fechas.
    Devuelve la ruta del parquet generado.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_parquet = out_dir / f"{tmp_name}.parquet"

    hourly_vars = [
        "temperature_2m",
        "precipitation",
        "rain",
        "snowfall",
        "wind_speed_10m",
        "weather_code",
    ]

    rows: list[dict[str, Any]] = []

    console.print(f"[dim]  → Descargando datos meteorológicos...[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[cyan]{task.fields[rows]} filas"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("[cyan]Descargando", total=None, rows=0)

        for start_date, end_date in _date_chunks(date_from, date_to, chunk_days=31):
            payload = _fetch_open_meteo_hourly(
                latitude=latitude,
                longitude=longitude,
                date_from=start_date,
                date_to=end_date,
                timezone=timezone,
                hourly_vars=hourly_vars,
            )

            hourly = payload.get("hourly") or {}
            times: list[str] = hourly.get("time") or []

            temp = hourly.get("temperature_2m") or []
            precip = hourly.get("precipitation") or []
            rain = hourly.get("rain") or []
            snow = hourly.get("snowfall") or []
            wind = hourly.get("wind_speed_10m") or []
            wcode = hourly.get("weather_code") or []

            n = len(times)

            def _safe_get(arr: list[Any], i: int) -> Any:
                return arr[i] if isinstance(arr, list) and i < len(arr) else None

            for i in range(n):
                t = times[i]  # "YYYY-MM-DDTHH:MM"
                d = t[:10]
                try:
                    hh = int(t[11:13])
                except Exception:
                    hh = None

                rows.append(
                    {
                        "date": d,
                        "hour": hh,
                        "temp_c": _safe_get(temp, i),
                        "precip_mm": _safe_get(precip, i),
                        "rain_mm": _safe_get(rain, i),
                        "snowfall_mm": _safe_get(snow, i),
                        "wind_kmh": _safe_get(wind, i),
                        "weather_code": _safe_get(wcode, i),
                    }
                )

            progress.update(task, rows=len(rows))

    console.print(f"[dim]  → {len(rows)} registros obtenidos[/dim]")

    # Crear DataFrame y guardar
    df = pd.DataFrame(rows)
    dataframe_to_parquet(df, tmp_parquet)

    return tmp_parquet


def download_meteo_month(
    year: int,
    month: int,
    out_dir: Path,
    latitude: float,
    longitude: float,
    timezone: str,
    tag: str = "meteo",
) -> dict[str, Any]:
    """
    Descarga datos meteorológicos de un mes específico.
    """
    from datetime import date as dt_date
    
    start = dt_date(year, month, 1)
    end = dt_date(year, month, calendar.monthrange(year, month)[1])

    out_dir.mkdir(parents=True, exist_ok=True)
    final_parquet = out_dir / f"{tag}_{year}_{month:02d}.parquet"

    console.print(f"\n[bold cyan]{year}-{month:02d}[/bold cyan]")

    if final_parquet.exists():
        console.print(f"[yellow]SKIP:[/yellow] {final_parquet.name}")
        return {"status": "skipped", "path": str(final_parquet)}

    tmp_name = f"__tmp_{tag}_{year}_{month:02d}"
    tmp_parquet = download_meteo_aggregated(
        date_from=start.isoformat(),
        date_to=end.isoformat(),
        out_dir=out_dir,
        tmp_name=tmp_name,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
    )

    tmp_parquet.rename(final_parquet)
    console.print(f"[green]OK:[/green] {final_parquet.name}")
    
    return {"status": "ok", "path": str(final_parquet)}


def download_meteo_range(
    start_year: int,
    end_year: int,
    start_month: int = 1,
    end_month: int = 12,
    out_dir: Optional[Path] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    timezone: Optional[str] = None,
) -> dict[str, int]:
    """
    Descarga datos meteorológicos para un rango de años y meses.
    """
    out_dir = out_dir or DEFAULT_OUT_DIR
    
    # Usar valores de configuración si no se proporcionan
    latitude = latitude if latitude is not None else meteo_config['latitude']
    longitude = longitude if longitude is not None else meteo_config['longitude']
    timezone = timezone or meteo_config['timezone']
    
    years = range(start_year, end_year + 1)
    months = range(start_month, end_month + 1)

    stats = {"total": 0, "ok": 0, "skipped": 0, "failed": 0}

    console.print(f"\n[bold]Descargando datos meteorológicos NYC[/bold]")
    console.print(f"[yellow]Período:[/yellow] {start_year}/{start_month:02d} → {end_year}/{end_month:02d}")
    console.print(f"[yellow]Coordenadas:[/yellow] lat={latitude}, lon={longitude}")
    console.print(f"[yellow]Timezone:[/yellow] {timezone}")
    console.print(f"[yellow]Destino:[/yellow] {out_dir}")
    console.rule(style="dim")

    for year, month in itertools.product(years, months):
        stats["total"] += 1
        try:
            r = download_meteo_month(
                year=year,
                month=month,
                out_dir=out_dir,
                latitude=latitude,
                longitude=longitude,
                timezone=timezone,
            )
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
@click.option("--start-year", type=int, default=2023, show_default=True, help="Año inicial")
@click.option("--end-year", type=int, default=2025, show_default=True, help="Año final")
@click.option("--start-month", type=click.IntRange(1, 12), default=1, help="Mes inicial (1-12)")
@click.option("--end-month", type=click.IntRange(1, 12), default=12, help="Mes final (1-12)")
@click.option("--latitude", type=float, default=None, help="Latitud (default desde config)")
@click.option("--longitude", type=float, default=None, help="Longitud (default desde config)")
@click.option("--timezone", type=str, default=None, help="Timezone (default desde config)")
def main(
    start_year: int,
    end_year: int,
    start_month: int,
    end_month: int,
    latitude: Optional[float],
    longitude: Optional[float],
    timezone: Optional[str],
):
    """
    Descarga datos meteorológicos de NYC desde Open-Meteo Archive API.
    
    Ejemplos de uso:
    
        # Descargar todos los datos de 2024
        uv run -m src.extraccion.download_meteo_data \\
            --start-year 2024 \\
            --end-year 2024
        
        # Descargar datos de junio a agosto de 2023
        uv run -m src.extraccion.download_meteo_data \\
            --start-year 2023 \\
            --end-year 2023 \\
            --start-month 6 \\
            --end-month 8
        
        # Usar coordenadas personalizadas
        uv run -m src.extraccion.download_meteo_data \\
            --start-year 2024 \\
            --end-year 2024 \\
            --latitude 40.7580 \\
            --longitude -73.9855
    """
    download_meteo_range(
        start_year=start_year,
        end_year=end_year,
        start_month=start_month,
        end_month=end_month,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
    )


if __name__ == "__main__":
    main()