from __future__ import annotations

import argparse
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from src.procesamiento.capa3.builders.stress_zone import build_stress_zone_dataset
from src.procesamiento.capa3.common.constants import console, obtener_ruta
from src.procesamiento.capa3.common.io import cleanup_dataset_output, resolve_layer2_input_path


def main() -> None:
    console.print(
        Panel.fit("[bold cyan]CAPA 3 EX STRESS - TLC: DF_STRESS_ZONE_HOUR_DAY (MODEL-READY V1)[/bold cyan]")
    )

    p = argparse.ArgumentParser(
        description=(
            "Construye datasets model-ready y de panel para stress urbano por zona-hora, "
            "con targets target_stress_t1 y target_is_stress_t1."
        )
    )
    p.add_argument(
        "--in-dir",
        default=str(obtener_ruta("data/standarized")),
        help="Ruta capa 2 TLC estandarizada",
    )
    p.add_argument(
        "--out-dir",
        default=str(obtener_ruta("data/aggregated/ex_stress")),
        help="Salida EX stress",
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
        "--zone-lookup",
        default=str(obtener_ruta("data/external/taxi_zone_lookup.csv")),
        help="CSV de lookup de zonas TLC",
    )
    p.add_argument("--min-date", default="2023-01-01", help="YYYY-MM-DD (inclusive)")
    p.add_argument("--max-date", default="2025-12-31", help="YYYY-MM-DD (inclusive)")
    p.add_argument(
        "--stress-quantile",
        type=float,
        default=0.90,
        help="Cuantil para etiquetar stress (default: 0.90)",
    )
    p.add_argument(
        "--mode",
        choices=["overwrite", "append"],
        default="append",
        help="append conserva salidas existentes; overwrite las borra antes de recalcular",
    )
    p.add_argument(
        "--keep-na-history",
        action="store_true",
        help="Si se activa, no elimina filas con NaN en lags/rolling",
    )
    p.add_argument(
        "--keep-na-targets",
        action="store_true",
        help="Si se activa, no elimina filas con target t+1 nulo",
    )
    p.add_argument(
        "--model-dataset-name",
        default="df_stress_zone_hour_day",
        help="Nombre de salida del dataset model-ready",
    )
    p.add_argument(
        "--panel-dataset-name",
        default="df_stress_zone_slot",
        help="Nombre de salida del dataset agregado para panel",
    )
    args = p.parse_args()

    layer2_path = resolve_layer2_input_path(Path(args.in_dir))
    out_base = Path(args.out_dir).resolve()
    meteo_base = Path(args.meteo_dir).resolve()
    events_base = Path(args.events_dir).resolve()
    zone_lookup_path = str(Path(args.zone_lookup).resolve())

    cfg = Table(show_header=True, header_style="bold white", title="Configuracion EX Stress")
    cfg.add_column("Campo", style="bold cyan")
    cfg.add_column("Valor")
    cfg.add_row("in_dir", str(layer2_path))
    cfg.add_row("out_dir", str(out_base))
    cfg.add_row("meteo_dir", str(meteo_base))
    cfg.add_row("events_dir", str(events_base))
    cfg.add_row("zone_lookup", zone_lookup_path)
    cfg.add_row("min_date", args.min_date)
    cfg.add_row("max_date", args.max_date)
    cfg.add_row("stress_quantile", str(args.stress_quantile))
    cfg.add_row("targets", "target_stress_t1, target_is_stress_t1")
    cfg.add_row("mode", args.mode)
    cfg.add_row("keep_na_history", str(args.keep_na_history))
    cfg.add_row("keep_na_targets", str(args.keep_na_targets))
    cfg.add_row("model_dataset_name", args.model_dataset_name)
    cfg.add_row("panel_dataset_name", args.panel_dataset_name)
    console.print(cfg)

    if args.mode == "overwrite":
        cleanup_dataset_output(out_base, args.model_dataset_name, label="EX stress (model)")
        cleanup_dataset_output(out_base, args.panel_dataset_name, label="EX stress (panel)")
    else:
        console.print("[yellow]Modo append:[/yellow] se conservaran salidas previas si ya existen.")

    out_base.mkdir(parents=True, exist_ok=True)

    stats = build_stress_zone_dataset(
        layer2_path=layer2_path,
        out_base=out_base,
        meteo_base=meteo_base,
        events_base=events_base,
        min_date=args.min_date,
        max_date=args.max_date,
        drop_na_history=not args.keep_na_history,
        drop_na_targets=not args.keep_na_targets,
        stress_quantile=args.stress_quantile,
        zone_lookup_path=zone_lookup_path,
        output_model_dataset_name=args.model_dataset_name,
        output_panel_dataset_name=args.panel_dataset_name,
    )

    summary = Table(show_header=True, header_style="bold green", title="Resumen final EX Stress")
    summary.add_column("Metrica", style="bold white")
    summary.add_column("Valor", justify="right")
    summary.add_row("rows_input", f"{stats.rows_input:,}")
    summary.add_row("rows_after_filter", f"{stats.rows_after_filter:,}")
    summary.add_row("rows_base_aggregated", f"{stats.rows_base_aggregated:,}")
    summary.add_row("rows_base_completed", f"{stats.rows_base_completed:,}")
    summary.add_row("rows_model_out", f"{stats.rows_model_out:,}")
    summary.add_row("rows_panel_out", f"{stats.rows_panel_out:,}")
    summary.add_row("unique_zones", f"{stats.unique_zones:,}")
    summary.add_row("unique_timestamps", f"{stats.unique_timestamps:,}")
    summary.add_row("meteo_rows", f"{stats.meteo_rows:,}")
    summary.add_row("events_rows", f"{stats.events_rows:,}")
    summary.add_row("threshold_now", f"{stats.threshold_now:.4f}")
    summary.add_row("threshold_target", f"{stats.threshold_target:.4f}")
    console.print(summary)

    console.print(
        f"[bold green]OK[/bold green] EX stress guardado en: "
        f"{out_base / args.model_dataset_name} y {out_base / args.panel_dataset_name}"
    )


if __name__ == "__main__":
    main()
