from __future__ import annotations

"""
Wrapper sencillo para el apartado EX1(a):
- usa TLC estandarizado como base
- construye un panel por (timestamp_hour, pu_location_id)
- añade lags/rolling + meteo + eventos ciudad-hora
- deja el resultado en data/aggregated/ex1a/df_demand_zone_hour_day

No toca los mains originales. Se ejecuta por separado.
"""

import argparse
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from src.procesamiento.capa3.builders.demand_zone import build_demand_zone_dataset
from src.procesamiento.capa3.common.constants import console, obtener_ruta
from src.procesamiento.capa3.common.io import cleanup_dataset_output, resolve_layer2_input_path


DEFAULT_INPUT_DIR = "data/standarized"
DEFAULT_OUTPUT_DIR = "data/aggregated/ex1a"
DEFAULT_METEO_DIR = "data/external/meteo/standarized"
DEFAULT_EVENTS_DIR = "data/external/events/standarized"
DEFAULT_DATASET_NAME = "df_demand_zone_hour_day"


def main() -> None:
    console.print(Panel.fit("[bold cyan]CAPA 3 EX1(a) - DEMANDA POR ZONA Y FRANJA[/bold cyan]"))

    p = argparse.ArgumentParser(
        description=(
            "Construye el panel de demanda del apartado a. "
            "Cada fila representa una zona TLC en una hora concreta."
        )
    )
    p.add_argument("--in-dir", default=str(obtener_ruta(DEFAULT_INPUT_DIR)), help="Ruta de TLC estandarizado (capa 2)")
    p.add_argument("--out-dir", default=str(obtener_ruta(DEFAULT_OUTPUT_DIR)), help="Directorio base de salida EX1(a)")
    p.add_argument("--meteo-dir", default=str(obtener_ruta(DEFAULT_METEO_DIR)), help="Ruta meteo estandarizada")
    p.add_argument("--events-dir", default=str(obtener_ruta(DEFAULT_EVENTS_DIR)), help="Ruta eventos estandarizada")
    p.add_argument("--min-date", default="2023-01-01", help="YYYY-MM-DD inclusive")
    p.add_argument("--max-date", default="2025-12-31", help="YYYY-MM-DD inclusive")
    p.add_argument("--mode", choices=["overwrite", "append"], default="append")
    p.add_argument("--keep-na-lags", action="store_true", help="Conservar filas con NaN en lags/rolling")
    args = p.parse_args()

    layer2_path = resolve_layer2_input_path(Path(args.in_dir))
    out_base = Path(args.out_dir).resolve()
    meteo_base = Path(args.meteo_dir).resolve()
    events_base = Path(args.events_dir).resolve()

    cfg = Table(show_header=True, header_style="bold white", title="Configuración capa3_demanda")
    cfg.add_column("Campo", style="bold cyan")
    cfg.add_column("Valor")
    cfg.add_row("in_dir", str(layer2_path))
    cfg.add_row("out_dir", str(out_base))
    cfg.add_row("meteo_dir", str(meteo_base))
    cfg.add_row("events_dir", str(events_base))
    cfg.add_row("min_date", args.min_date)
    cfg.add_row("max_date", args.max_date)
    cfg.add_row("mode", args.mode)
    cfg.add_row("keep_na_lags", str(args.keep_na_lags))
    console.print(cfg)

    if args.mode == "overwrite":
        cleanup_dataset_output(out_base, DEFAULT_DATASET_NAME, label="EX1(a)")
    else:
        console.print("[yellow]Modo append:[/yellow] se conservarán salidas previas si ya existen.")

    out_base.mkdir(parents=True, exist_ok=True)
    stats = build_demand_zone_dataset(
        layer2_path=layer2_path,
        out_base=out_base,
        meteo_base=meteo_base,
        events_base=events_base,
        min_date=args.min_date,
        max_date=args.max_date,
        drop_na_lags=not args.keep_na_lags,
        output_dataset_name=DEFAULT_DATASET_NAME,
    )

    summary = Table(show_header=True, header_style="bold green", title="Resumen final EX1(a)")
    summary.add_column("Métrica", style="bold white")
    summary.add_column("Valor", justify="right")
    summary.add_row("rows_input", f"{stats.rows_input:,}")
    summary.add_row("rows_after_filter", f"{stats.rows_after_filter:,}")
    summary.add_row("rows_base_aggregated", f"{stats.rows_base_aggregated:,}")
    summary.add_row("rows_base_completed", f"{stats.rows_base_completed:,}")
    summary.add_row("rows_out", f"{stats.rows_out:,}")
    summary.add_row("unique_zones", f"{stats.unique_zones:,}")
    summary.add_row("unique_timestamps", f"{stats.unique_timestamps:,}")
    summary.add_row("meteo_rows", f"{stats.meteo_rows:,}")
    summary.add_row("events_rows", f"{stats.events_rows:,}")
    console.print(summary)

    console.print(f"[bold green]OK[/bold green] EX1(a) guardado en: {out_base / DEFAULT_DATASET_NAME}")


if __name__ == "__main__":
    main()
