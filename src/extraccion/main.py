# src/extraccion/descarga_masiva.py
from enum import Enum
from typing import Optional, Callable, Dict, Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Imports de scripts específicos
from src.extraccion.download_tlc_data import download_service_data
from src.extraccion.download_events_data import download_events_range
from src.extraccion.download_meteo_data import download_meteo_range
from src.extraccion.download_rent_data import download_rent_snapshot 
from src.extraccion.download_restaurants_data import download_restaurants_range
from src.extraccion.download_taxi_zones import (
    download_taxi_lookup,
    download_taxi_zones_shapefile,
)

from config.settings import (
    obtener_ruta,
    config,
    servicios_habilitados,
    eventos_config,
    meteo_config,
    restaurants_config,
)

console = Console()

class DownloadMode(str, Enum):
    ALL = "all"
    TLC = "tlc"
    EVENTS = "events"
    METEO = "meteo"
    RENT = "rent"
    RESTAURANTS = "restaurants"
    ZONES = "zones"

# =============================================================================
# Orquestador Genérico
# =============================================================================

def run_download_task(name: str, color: str, func: Callable, **kwargs) -> Dict[str, Any]:
    """Ejecuta una tarea de descarga con un formato visual consistente."""
    console.print()
    console.print(Panel.fit(f"[bold white]{name.upper()}[/bold white]", border_style=color))
    console.rule(f"[{color}]Procesando {name}[/{color}]", style=color)
    
    try:
        stats = func(**kwargs)
        # Normalizar stats para la tabla de resumen
        if not isinstance(stats, dict): 
            stats = {"successful": 1, "total": 1, "skipped": 0, "failed": 0}
        
        console.print(f"\n[bold green]{name} completado[/bold green]")
        return {"status": "completado", "stats": stats, "color": color, "type": name.lower()}
    except Exception as e:
        console.print(f"[bold red]ERROR en {name}:[/bold red] {e}")
        return {"status": "error", "error": str(e), "color": color, "type": name.lower()}

# =============================================================================
# Lógica específica por grupo
# =============================================================================

def download_all_services():
    """Lógica para los múltiples servicios de TLC (Yellow, Green, etc.)"""
    results = {}
    services = {
        'yellow': ('Yellow Taxi', 'yellow'),
        'green': ('Green Taxi', 'green'),
        'fhvhv': ('HVFHV', 'magenta')
    }
    
    for s_key, (desc, color) in services.items():
        if s_key in servicios_habilitados:
            res = run_download_task(
                desc, color, download_service_data,
                service=s_key,
                years=range(config['descarga']['start_year'], config['descarga']['end_year'] + 1),
                months=range(config['descarga']['start_month'], config['descarga']['end_month'] + 1),
                data_dir=obtener_ruta("data/raw")
            )
            results[s_key] = res
            results[s_key]["type"] = "tlc" # Forzar tipo para el resumen
    return results

# =============================================================================
# Resumen y CLI
# =============================================================================

def print_summary(results: dict):
    """Imprime una tabla consolidada con todo lo descargado."""
    console.print()
    console.rule("[bold cyan]RESUMEN DE DESCARGA MASIVA[/bold cyan]", style="cyan")
    
    table = Table(show_header=True, header_style="bold white")
    table.add_column("Categoría/Servicio", width=30)
    table.add_column("Estado", justify="center")
    table.add_column("Descargados", justify="right")
    table.add_column("Total", justify="right")

    for key, res in results.items():
        color = res.get("color", "white")
        status = "[green]OK[/green]" if res["status"] == "completado" else "[red]ERROR[/red]"
        stats = res.get("stats", {})
        
        # Obtener métricas (manejando diferentes nombres de keys en stats)
        ok = stats.get('successful', stats.get('ok', 0))
        total = stats.get('total', 0)
        
        table.add_row(f"[{color}]{key.upper()}[/{color}]", status, str(ok), str(total))

    console.print(table)

def download_all(mode: DownloadMode):
    results = {}
    
    # 1. TLC
    if mode in (DownloadMode.ALL, DownloadMode.TLC):
        results.update(download_all_services())

    # 2. EVENTOS
    if mode in (DownloadMode.ALL, DownloadMode.EVENTS):
        results['events'] = run_download_task(
            "Eventos NYC", "magenta", download_events_range,
            dataset_id=eventos_config['dataset_id'],
            start_year=eventos_config['start_year'], end_year=eventos_config['end_year'],
            start_month=eventos_config['start_month'], end_month=eventos_config['end_month'],
            out_dir=obtener_ruta("data/external/events/raw")
        )

    # 3. METEO
    if mode in (DownloadMode.ALL, DownloadMode.METEO):
        results['meteo'] = run_download_task(
            "Meteorología", "blue", download_meteo_range,
            start_year=meteo_config['start_year'], end_year=meteo_config['end_year'],
            start_month=meteo_config['start_month'], end_month=meteo_config['end_month'],
            out_dir=obtener_ruta("data/external/meteo/raw"),
            latitude=meteo_config['latitude'], longitude=meteo_config['longitude']
        )

    # 4. RENT (Airbnb Proxy)
    if mode in (DownloadMode.ALL, DownloadMode.RENT):
        results['rent'] = run_download_task(
            "Rent Data (NYC)", "green", download_rent_snapshot,
            url=config.get('rent_url') or None,
            out_dir=obtener_ruta("data/external/rent/raw"),
            provider="acs",
            dataset_kind="summary",
        )

    # 5. RESTAURANTS
    if mode in (DownloadMode.ALL, DownloadMode.RESTAURANTS):
        results['restaurants'] = run_download_task(
            "Restaurantes NYC", "red", download_restaurants_range,
            dataset_id=restaurants_config['dataset_id'],
            start_year=restaurants_config['start_year'], end_year=restaurants_config['end_year'],
            start_month=restaurants_config['start_month'], end_month=restaurants_config['end_month'],
            out_dir=obtener_ruta("data/external/restaurants/raw")
        )

    # 6. TAXI ZONES
    if mode in (DownloadMode.ALL, DownloadMode.ZONES):
        results['zones_lookup'] = run_download_task(
            "Taxi Zones Lookup", "yellow", download_taxi_lookup,
            out_dir=obtener_ruta("data/external")
        )

        results['zones_shapefile'] = run_download_task(
            "Taxi Zones Shapefile", "yellow", download_taxi_zones_shapefile,
            out_dir=obtener_ruta("data/external")
        )

    print_summary(results)

@click.command()
@click.option("--mode", type=click.Choice([m.value for m in DownloadMode]), default="all")
def main(mode: str):
    download_all(DownloadMode(mode))

if __name__ == "__main__":
    main()
