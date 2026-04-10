from __future__ import annotations

import argparse
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from src.procesamiento.capa3.builders.demand_zone import build_demand_zone_dataset
from src.procesamiento.capa3.common.constants import console, obtener_ruta
from src.procesamiento.capa3.common.io import cleanup_dataset_output, resolve_layer2_input_path


def main() -> None:
    console.print(Panel.fit("[bold cyan]CAPA 3 EX1(a) - TLC: DF_DEMAND_ZONE_HOUR_DAY[/bold cyan]"))

    p = argparse.ArgumentParser(
        description="Construye un unico dataset model-ready para Ejercicio 1(a)."
    )
    p.add_argument(
        "--in-dir",
        default=str(obtener_ruta("data/standarized")),
        help="Ruta capa 2 TLC estandarizada",
    )
    p.add_argument(
        "--out-dir",
        default=str(obtener_ruta("data/aggregated/ex1a")),
        help="Salida EX1(a)",
    )
    p.add_argument(
        "--meteo-dir",
        default=str(obtener_ruta("data/external/meteo/standarized")),
        help="Ruta meteo estandarizada",
    )
    p.add_argument(
        "--events-dir",
        default=str(obtener_ruta("data/external/events/standarized")),
        help="Ruta eventos estandarizada",
    )
    p.add_argument(
        "--restaurants-dir",
        default=str(obtener_ruta("data/external/restaurants/aggregated/df_location_static")),
        help="Ruta restaurants agregada por pu_location_id",
    )
    p.add_argument(
        "--rent-dir",
        default=str(obtener_ruta("data/external/rent/aggregated/df_location_static")),
        help="Ruta rent agregada por pu_location_id",
    )
    p.add_argument("--min-date", default="2023-01-01", help="YYYY-MM-DD (inclusive)")
    p.add_argument("--max-date", default="2025-12-31", help="YYYY-MM-DD (inclusive)")
    p.add_argument(
        "--mode",
        choices=["overwrite", "append"],
        default="append",
        help="append conserva salidas existentes; overwrite las borra antes de recalcular",
    )
    p.add_argument(
        "--keep-na-lags",
        action="store_true",
        help="Si se activa, no elimina filas con NaN en lags/rolling",
    )
    args = p.parse_args()

    layer2_path = resolve_layer2_input_path(Path(args.in_dir))
    out_base = Path(args.out_dir).resolve()
    meteo_base = Path(args.meteo_dir).resolve()
    events_base = Path(args.events_dir).resolve()
    restaurants_base = Path(args.restaurants_dir).resolve()
    rent_base = Path(args.rent_dir).resolve()
    dataset_name = "df_demand_zone_hour_day"

    cfg = Table(show_header=True, header_style="bold white", title="Configuracion EX1(a)")
    cfg.add_column("Campo", style="bold cyan")
    cfg.add_column("Valor")
    cfg.add_row("in_dir", str(layer2_path))
    cfg.add_row("out_dir", str(out_base))
    cfg.add_row("meteo_dir", str(meteo_base))
    cfg.add_row("events_dir", str(events_base))
    cfg.add_row("restaurants_dir", str(restaurants_base))
    cfg.add_row("rent_dir", str(rent_base))
    cfg.add_row("min_date", args.min_date)
    cfg.add_row("max_date", args.max_date)
    cfg.add_row("mode", args.mode)
    cfg.add_row("keep_na_lags", str(args.keep_na_lags))
    console.print(cfg)

    if args.mode == "overwrite":
        cleanup_dataset_output(out_base, dataset_name, label="EX1(a)")
    else:
        console.print("[yellow]Modo append:[/yellow] se conservaran salidas previas si ya existen.")

    out_base.mkdir(parents=True, exist_ok=True)

    stats = build_demand_zone_dataset(
        layer2_path=layer2_path,
        out_base=out_base,
        meteo_base=meteo_base,
        events_base=events_base,
        restaurants_base=restaurants_base,
        rent_base=rent_base,
        min_date=args.min_date,
        max_date=args.max_date,
        drop_na_lags=not args.keep_na_lags,
        output_dataset_name=dataset_name,
    )

    summary = Table(show_header=True, header_style="bold green", title="Resumen final EX1(a)")
    summary.add_column("Metrica", style="bold white")
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
    summary.add_row("restaurants_rows", f"{stats.restaurants_rows:,}")
    summary.add_row("rent_rows", f"{stats.rent_rows:,}")
    console.print(summary)

    console.print(f"[bold green]OK[/bold green] EX1(a) guardado en: {out_base / dataset_name}")


if __name__ == "__main__":
    main()
