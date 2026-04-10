# src/procesamiento/capa1/cli.py
import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.text import Text
from typing import List

from .core_io import procesar_archivo_en_batches
from .rules_yellow import clean_yellow_batch
from .rules_green import clean_green_batch
from .rules_fhvhv import clean_fhvhv_batch
from .rules_meteo import clean_meteo_batch
from .rules_eventos import clean_eventos_batch
from .rules_rent import clean_rent_batch
from .rules_restaurants import clean_restaurants_batch

from config.settings import obtener_ruta

console = Console()

def _discover_parquet_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted([p for p in input_path.glob("*.parquet") if p.is_file()])

def _run_pipeline(service_name: str, input_dir: Path, output_dir: Path, cleaning_func):
    console.print(f"\n[bold cyan]--- Capa 1: {service_name} ---[/bold cyan]")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    parquets = _discover_parquet_files(input_dir)
    if not parquets:
        console.print(f"[yellow]No se encontraron archivos .parquet en {input_dir}[/yellow]")
        return

    table = Table(title=f"Resultados {service_name}")
    table.add_column("Archivo", style="cyan")
    table.add_column("Filas Conservadas", justify="right")
    table.add_column("Filas Eliminadas", justify="right")
    table.add_column("Filas Cambiadas", justify="right")
    table.add_column("Nulos Totales", justify="right")
    
    total_raw, total_clean = 0, 0
    total_removed, total_changed = 0, 0
    total_removed_reasons = {}

    for file_path in parquets:
        out_file = output_dir / f'{file_path.stem}.parquet'
        
        stats = procesar_archivo_en_batches(file_path, out_file, cleaning_func)
        
        prct = stats['n_clean_rows'] / stats['n_rows'] * 100
        if prct < 75:
            prct_text = Text(f"{prct:.2f}%", style="bold red")
        else:
            prct_text = Text(f"{prct:.2f}%", style="bold green")
        
        nulos = stats['null_prct']
        if nulos is None:
            nulos_text = Text("-", style="dim")
        elif nulos < 15:
            nulos_text = Text(f"{nulos:.2f}%", style="magenta")
        else:
            nulos_text = Text(f"{nulos:.2f}%", style="bold red")

        removed_rows = int(stats.get("removed_rows", stats["n_rows"] - stats["n_clean_rows"]))
        changed_rows = int(stats.get("changed_rows", 0))
        removed_reasons = stats.get("removed_reasons", {}) or {}

        table.add_row(file_path.name, prct_text, f"{removed_rows:,}", f"{changed_rows:,}", nulos_text)
        total_raw += stats['n_rows']
        total_clean += stats['n_clean_rows']
        total_removed += removed_rows
        total_changed += changed_rows
        for reason, count in removed_reasons.items():
            total_removed_reasons[reason] = int(total_removed_reasons.get(reason, 0)) + int(count)

    console.print(table)
    console.print(f"[bold green]Total {service_name}: {total_clean / total_raw * 100:.2f}% filas conservadas.[/bold green]\n")
    console.print(
        f"[bold cyan]Filas eliminadas:[/bold cyan] {total_removed:,} | "
        f"[bold cyan]Filas cambiadas:[/bold cyan] {total_changed:,}\n"
    )

    if total_removed_reasons:
        reasons_table = Table(title=f"Motivos de eliminación ({service_name})")
        reasons_table.add_column("Motivo", style="yellow")
        reasons_table.add_column("Filas", justify="right")
        for reason, count in sorted(total_removed_reasons.items(), key=lambda x: x[1], reverse=True):
            reasons_table.add_row(str(reason), f"{int(count):,}")
        console.print(reasons_table)

@click.group(help="Pipeline de la Capa 1: Validación y Limpieza de Datos TLC.")
def cli():
    pass

@cli.command(help="Procesa los datos de Yellow Taxis.")
@click.option('--input-dir', default=str(obtener_ruta("data/raw/yellow")), help="Ruta RAW")
@click.option('--output-dir', default=str(obtener_ruta("data/validated/yellow")), help="Ruta OUT")
def yellow(input_dir, output_dir):
    _run_pipeline("Yellow Taxis", Path(input_dir), Path(output_dir), clean_yellow_batch)

@cli.command(help="Procesa los datos de Green Taxis.")
@click.option('--input-dir', default=str(obtener_ruta("data/raw/green")), help="Ruta RAW")
@click.option('--output-dir', default=str(obtener_ruta("data/validated/green")), help="Ruta OUT")
def green(input_dir, output_dir):
    _run_pipeline("Green Taxis", Path(input_dir), Path(output_dir), clean_green_batch)

@cli.command(help="Procesa los datos de High Volume FHV (Uber, Lyft, etc.).")
@click.option('--input-dir', default=str(obtener_ruta("data/raw/fhvhv")), help="Ruta RAW")
@click.option('--output-dir', default=str(obtener_ruta("data/validated/fhvhv")), help="Ruta OUT")
def fhvhv(input_dir, output_dir):
    _run_pipeline("HVFHV", Path(input_dir), Path(output_dir), clean_fhvhv_batch)

@cli.command(help="Procesa los datos Metereológicos.")
@click.option('--input-dir', default=str(obtener_ruta("data/external/meteo/raw")), help="Ruta RAW")
@click.option('--output-dir', default=str(obtener_ruta("data/external/meteo/validated")), help="Ruta OUT")
def meteo(input_dir, output_dir):
    _run_pipeline("Meteo", Path(input_dir), Path(output_dir), clean_meteo_batch)

@cli.command(help="Procesa los datos de Eventos.")
@click.option('--input-dir', default=str(obtener_ruta("data/external/events/raw")), help="Ruta RAW")
@click.option('--output-dir', default=str(obtener_ruta("data/external/events/validated")), help="Ruta OUT")
def events(input_dir, output_dir):
    _run_pipeline("Events", Path(input_dir), Path(output_dir), clean_eventos_batch)

@cli.command(help="Procesa los datos de Alquiler.")
@click.option('--input-dir', default=str(obtener_ruta("data/external/rent/raw")), help="Ruta RAW")
@click.option('--output-dir', default=str(obtener_ruta("data/external/rent/validated")), help="Ruta OUT")
def rent(input_dir, output_dir):
    _run_pipeline("Rent", Path(input_dir), Path(output_dir), clean_rent_batch)

@cli.command(help="Procesa los datos de Restaurantes.")
@click.option('--input-dir', default=str(obtener_ruta("data/external/restaurants/raw")), help="Ruta RAW")
@click.option('--output-dir', default=str(obtener_ruta("data/external/restaurants/validated")), help="Ruta OUT")
def restaurants(input_dir, output_dir):
    _run_pipeline("Restaurants", Path(input_dir), Path(output_dir), clean_restaurants_batch)

@cli.command(help="Ejecuta TODO el pipeline de la capa 1 secuencialmente.")
@click.pass_context
def all(ctx):
    console.print("[bold magenta]EJECUTANDO TODO EL PIPELINE CAPA 1[/bold magenta]")
    ctx.invoke(yellow)
    ctx.invoke(green)
    ctx.invoke(fhvhv)
    ctx.invoke(meteo)
    ctx.invoke(events)
    ctx.invoke(rent)
    ctx.invoke(restaurants)

if __name__ == '__main__':
    cli()
