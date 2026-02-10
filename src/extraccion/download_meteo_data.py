# src/extraccion/download_meteo_data.py
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


# Intentamos usar vuestro sistema de config/rutas (lo mantenemos por compatibilidad),
# pero para evitar errores de rutas en Windows/ejecución por el profe, NO dependeremos de obtener_ruta.
try:
    from config.settings import obtener_ruta, config  # type: ignore
except Exception:  # fallback si alguien ejecuta sin ese módulo
    config = {}

    def obtener_ruta(p: str) -> Path:
        return Path(p)


console = Console()

# Open-Meteo Historical Weather API (Archive)
# Docs: https://open-meteo.com/en/docs/historical-weather-api
BASE = "https://archive-api.open-meteo.com/v1/archive"


def _date_chunks(date_from: str, date_to: str, chunk_days: int = 31) -> Iterable[tuple[str, str]]:
    """
    Divide un rango [date_from, date_to] en trozos de chunk_days para evitar respuestas gigantes.
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
) -> dict[str, Any]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date_from,
        "end_date": date_to,
        "timezone": timezone,
        "hourly": ",".join(hourly_vars),
    }
    r = requests.get(BASE, params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def download_meteo_hourly_nyc(
    date_from: str,
    date_to: str,
    out_dir: Path,
    out_name: str = "meteo_hourly_nyc",
    latitude: float = 40.7128,
    longitude: float = -74.0060,
    timezone: str = "America/New_York",
) -> dict[str, Any]:
    """
    Descarga meteo horario (NYC) desde Open-Meteo Archive API.

    Guarda:
      - CSV (siempre)
      - Parquet (con pandas+pyarrow si está disponible)

    Output columns:
      date (YYYY-MM-DD), hour (0-23),
      temp_c, precip_mm, rain_mm, snowfall_mm, wind_kmh, weather_code
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"{out_name}.csv"
    out_parquet = out_dir / f"{out_name}.parquet"

    hourly_vars = [
        "temperature_2m",
        "precipitation",
        "rain",
        "snowfall",
        "wind_speed_10m",
        "weather_code",
    ]

    rows: list[dict[str, Any]] = []

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

    # -------------------------
    # Guardado CSV (siempre)
    # -------------------------
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["date", "hour", "temp_c", "precip_mm", "rain_mm", "snowfall_mm", "wind_kmh", "weather_code"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # -------------------------
    # Guardado Parquet (portable: pandas+pyarrow)
    # -------------------------
    parquet_ok = True
    parquet_engine = None
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(rows)

        # Intentamos pyarrow primero (más común); si no, fastparquet
        try:
            import pyarrow  # noqa: F401  # type: ignore
            parquet_engine = "pyarrow"
        except Exception:
            try:
                import fastparquet  # noqa: F401  # type: ignore
                parquet_engine = "fastparquet"
            except Exception:
                parquet_engine = None

        if parquet_engine is None:
            raise RuntimeError(
                "No hay engine de parquet instalado. Instala 'pyarrow' (recomendado) o 'fastparquet'."
            )

        df.to_parquet(out_parquet, index=False, engine=parquet_engine)

    except Exception as e:
        parquet_ok = False
        console.print(
            f"[yellow][WARN][/yellow] No se pudo guardar Parquet con pandas/{parquet_engine or 'N/A'} "
            f"(se deja el CSV): {e}"
        )

    return {
        "source": "open-meteo-archive",
        "lat": latitude,
        "lon": longitude,
        "timezone": timezone,
        "from": date_from,
        "to": date_to,
        "rows": len(rows),
        "out_csv": str(out_csv),
        "out_parquet": str(out_parquet) if parquet_ok else None,
        "parquet_engine": parquet_engine if parquet_ok else None,
    }


# -----------------------------
# CLI
# -----------------------------
def main():
    meteo_cfg = (config.get("meteo") or {}) if isinstance(config, dict) else {}

    default_from = meteo_cfg.get("date_from", "2024-01-01")
    default_to = meteo_cfg.get("date_to", "2025-12-31")
    default_out_name = meteo_cfg.get("out_name", "meteo_hourly_nyc")
    default_lat = float(meteo_cfg.get("latitude", 40.7128))
    default_lon = float(meteo_cfg.get("longitude", -74.0060))
    default_tz = meteo_cfg.get("timezone", "America/New_York")

    p = argparse.ArgumentParser(description="Descarga meteo horario (NYC) desde Open-Meteo Archive API.")
    p.add_argument("--from", dest="date_from", default=default_from, help="YYYY-MM-DD")
    p.add_argument("--to", dest="date_to", default=default_to, help="YYYY-MM-DD")
    p.add_argument("--name", dest="out_name", default=default_out_name, help="Nombre base del archivo de salida")
    p.add_argument("--lat", dest="latitude", type=float, default=default_lat, help="Latitud (default NYC)")
    p.add_argument("--lon", dest="longitude", type=float, default=default_lon, help="Longitud (default NYC)")
    p.add_argument("--tz", dest="timezone", default=default_tz, help="Timezone (default America/New_York)")
    args = p.parse_args()

    try:
        datetime.strptime(args.date_from, "%Y-%m-%d")
        datetime.strptime(args.date_to, "%Y-%m-%d")
    except ValueError:
        raise SystemExit("Fechas inválidas. Usa formato YYYY-MM-DD")

    # ✅ Ruta robusta e independiente de la carpeta actual y de obtener_ruta:
    project_root = Path(__file__).resolve().parents[2]
    out_dir = (project_root / "data" / "external" / "meteo" / "raw").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print()
    console.print(
        Panel.fit(
            "[bold white]NYC METEO (Open-Meteo Archive)[/bold white]\n"
            "[cyan]Descarga meteo horario para NYC (date+hour)[/cyan]",
            border_style="bright_blue",
        )
    )
    console.print(f"[yellow]Periodo:[/yellow] {args.date_from} -> {args.date_to}")
    console.print(f"[yellow]Destino:[/yellow] {out_dir}")
    console.print(f"[yellow]Coords:[/yellow] lat={args.latitude}, lon={args.longitude} | tz={args.timezone}")
    console.print()

    stats = download_meteo_hourly_nyc(
        date_from=args.date_from,
        date_to=args.date_to,
        out_dir=out_dir,
        out_name=args.out_name,
        latitude=args.latitude,
        longitude=args.longitude,
        timezone=args.timezone,
    )

    table = Table(title="Resumen descarga meteo", show_header=True, header_style="bold cyan")
    table.add_column("Campo", style="cyan", width=18)
    table.add_column("Valor", style="white")

    table.add_row("source", str(stats["source"]))
    table.add_row("from", str(stats["from"]))
    table.add_row("to", str(stats["to"]))
    table.add_row("rows", str(stats["rows"]))
    table.add_row("out_csv", str(stats["out_csv"]))
    table.add_row("out_parquet", str(stats["out_parquet"]))
    table.add_row("parquet_engine", str(stats.get("parquet_engine")))

    console.print(table)
    console.print()
    console.print(
        Panel.fit(
            "[bold green]DESCARGA METEO COMPLETADA[/bold green]\n"
            f"[dim]Filas horarias: {stats['rows']}[/dim]",
            border_style="bright_green",
        )
    )


if __name__ == "__main__":
    main()