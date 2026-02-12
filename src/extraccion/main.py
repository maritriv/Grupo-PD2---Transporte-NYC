# src/extraccion/descarga_masiva.py
from enum import Enum
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.extraccion.download_tlc_data import download_service_data
from src.extraccion.download_events_data import download_events_range
from src.extraccion.download_meteo_data import download_meteo_range
from config.settings import obtener_ruta, config, servicios_habilitados, eventos_config, meteo_config


console = Console()


class DownloadMode(str, Enum):
    """Modos de descarga disponibles."""
    ALL = "all"
    TLC = "tlc"
    EVENTS = "events"
    METEO = "meteo"


# =============================================================================
# Descarga de servicios TLC
# =============================================================================

def download_all_services():
    """
    Descarga datos de todos los servicios de NYC TLC.
    """
    services = {
        'yellow': ('Yellow Taxi (taxis amarillos tradicionales)', 'yellow'),
        'green': ('Green Taxi (taxis "boro" fuera de Manhattan)', 'green'),
        'fhv': ('FHV (For-Hire Vehicles, livery, black car)', 'blue'),
        'fhvhv': ('HVFHV (Uber, Lyft, alto volumen)', 'magenta')
    }

    services = {
        k: v for k, v in services.items()
        if k in servicios_habilitados  # Filtrar por los servicios habilitados
    }
    
    total_services = len(services)

    # Configuración descarga
    descarga_config = config['descarga']
    start_year = descarga_config['start_year']
    end_year = descarga_config['end_year']
    months = range(descarga_config['start_month'], descarga_config['end_month'] + 1)
    
    # Header principal
    console.print()
    console.print(Panel.fit(
        "[bold white]NYC TAXI & LIMOUSINE COMMISSION[/bold white]\n"
        "[cyan]Descarga de Datos TLC[/cyan]",
        border_style="bright_blue"
    ))
    
    console.print(f"[yellow]Período:[/yellow] {start_year} - {end_year}")
    console.print(f"[yellow]Servicios a descargar:[/yellow] {total_services}")
    console.print(f"[yellow]Destino:[/yellow] data/raw/servicio/")
    console.print()
    
    # Almacenar resultados de cada servicio
    results = {}
    
    # Procesar cada servicio
    for idx, (service_key, (description, color)) in enumerate(services.items(), 1):
        console.rule(f"[{color}]{description}[/{color}]", style=color)
        console.print(f"[dim]Progreso global: [{idx}/{total_services}][/dim]\n")
        
        try:
            stats = download_service_data(
                service=service_key,
                years=range(start_year, end_year + 1),
                months=months,
                data_dir=obtener_ruta("data/raw")
            )
            results[service_key] = {
                "status": "completado",
                "stats": stats,
                "color": color,
                "type": "tlc"
            }
            console.print(f"[bold green]{description} - Completado[/bold green]\n")
            
        except Exception as e:
            console.print(f"[bold red]ERROR:[/bold red] {e}")
            console.print("[yellow]Continuando con el siguiente servicio...[/yellow]\n")
            results[service_key] = {
                "status": "error",
                "error": str(e),
                "color": color,
                "type": "tlc"
            }
            continue
    
    return results


# =============================================================================
# Descarga de eventos NYC
# =============================================================================

def download_all_events():
    """
    Descarga datos de eventos de NYC Open Data.
    """
    # Configuración descarga
    start_year = eventos_config['start_year']
    end_year = eventos_config['end_year']
    start_month = eventos_config['start_month']
    end_month = eventos_config['end_month']
    
    dataset_id = eventos_config['dataset_id']
    
    # Header
    console.print()
    console.print(Panel.fit(
        "[bold white]NYC OPEN DATA[/bold white]\n"
        "[cyan]Descarga de Eventos[/cyan]",
        border_style="bright_magenta"
    ))
    
    console.print(f"[yellow]Período:[/yellow] {start_year} - {end_year}")
    console.print(f"[yellow]Meses:[/yellow] {start_month} - {end_month}")
    console.print(f"[yellow]Dataset:[/yellow] {dataset_id}")
    console.print(f"[yellow]Destino:[/yellow] data/external/events/")
    console.print()
    
    console.rule("[magenta]Procesando eventos[/magenta]", style="magenta")
    console.print()
    
    results = {}
    
    try:
        stats = download_events_range(
            dataset_id=dataset_id,
            start_year=start_year,
            end_year=end_year,
            start_month=start_month,
            end_month=end_month,
            out_dir=obtener_ruta("data/external/events/raw")
        )
        
        results['events'] = {
            "status": "completado",
            "stats": stats,
            "color": "magenta",
            "type": "events"
        }
        
        console.print(f"\n[bold green] Descarga de eventos completada[/bold green]")
        console.print(f"[dim]Total: {stats['total']} | Descargados: {stats.get('ok', 0)} | "
                     f"Omitidos: {stats['skipped']} | Fallidos: {stats['failed']}[/dim]\n")
        
    except Exception as e:
        console.print(f"[bold red]ERROR:[/bold red] {e}")
        results['events'] = {
            "status": "error",
            "error": str(e),
            "color": "magenta",
            "type": "events"
        }
    
    return results


# =============================================================================
# Descarga de datos meteorológicos
# =============================================================================

def download_all_meteo():
    """
    Descarga datos meteorológicos de Open-Meteo.
    """
    # Configuración descarga
    start_year = meteo_config['start_year']
    end_year = meteo_config['end_year']
    start_month = meteo_config['start_month']
    end_month = meteo_config['end_month']
    
    latitude = meteo_config['latitude']
    longitude = meteo_config['longitude']
    timezone = meteo_config['timezone']
    
    # Header
    console.print()
    console.print(Panel.fit(
        "[bold white]OPEN-METEO ARCHIVE[/bold white]\n"
        "[cyan]Descarga de Datos Meteorológicos[/cyan]",
        border_style="bright_blue"
    ))
    
    console.print(f"[yellow]Período:[/yellow] {start_year} - {end_year}")
    console.print(f"[yellow]Meses:[/yellow] {start_month} - {end_month}")
    console.print(f"[yellow]Coordenadas:[/yellow] lat={latitude}, lon={longitude}")
    console.print(f"[yellow]Timezone:[/yellow] {timezone}")
    console.print(f"[yellow]Destino:[/yellow] data/external/meteo/")
    console.print()
    
    console.rule("[blue]Procesando datos meteorológicos[/blue]", style="blue")
    console.print()
    
    results = {}
    
    try:
        stats = download_meteo_range(
            start_year=start_year,
            end_year=end_year,
            start_month=start_month,
            end_month=end_month,
            out_dir=obtener_ruta("data/external/meteo/raw"),
            latitude=latitude,
            longitude=longitude,
            timezone=timezone,
        )
        
        results['meteo'] = {
            "status": "completado",
            "stats": stats,
            "color": "blue",
            "type": "meteo"
        }
        
        console.print(f"\n[bold green] Descarga de datos meteorológicos completada[/bold green]")
        console.print(f"[dim]Total: {stats['total']} | Descargados: {stats.get('ok', 0)} | "
                     f"Omitidos: {stats['skipped']} | Fallidos: {stats['failed']}[/dim]\n")
        
    except Exception as e:
        console.print(f"[bold red]ERROR:[/bold red] {e}")
        results['meteo'] = {
            "status": "error",
            "error": str(e),
            "color": "blue",
            "type": "meteo"
        }
    
    return results


# =============================================================================
# Resumen y reportes
# =============================================================================

def print_summary(results: dict):
    """
    Imprime un resumen consolidado de todas las descargas.
    
    Args:
        results: Diccionario con resultados de TLC, eventos y/o meteo
    """
    console.print()
    console.rule("[bold cyan]RESUMEN DE DESCARGA MASIVA[/bold cyan]", style="cyan")
    console.print()
    
    # Separar resultados por tipo
    tlc_results = {k: v for k, v in results.items() if v.get("type") == "tlc"}
    events_results = {k: v for k, v in results.items() if v.get("type") == "events"}
    meteo_results = {k: v for k, v in results.items() if v.get("type") == "meteo"}
    
    # Tabla de servicios TLC
    if tlc_results:
        table_tlc = Table(
            title="Servicios TLC",
            show_header=True,
            header_style="bold cyan"
        )
        table_tlc.add_column("Servicio", style="cyan", width=20)
        table_tlc.add_column("Estado", justify="center", width=12)
        table_tlc.add_column("Descargados", justify="right", width=12)
        table_tlc.add_column("Omitidos", justify="right", width=12)
        table_tlc.add_column("Fallidos", justify="right", width=12)
        table_tlc.add_column("Total", justify="right", width=12)
        
        total_downloaded = 0
        total_skipped = 0
        total_failed = 0
        total_files = 0
        services_with_errors = 0
        
        service_names = {
            'yellow': 'Yellow Taxi',
            'green': 'Green Taxi',
            'fhv': 'FHV',
            'fhvhv': 'HVFHV'
        }
        
        for service_key, result in tlc_results.items():
            service_name = service_names.get(service_key, service_key)
            color = result.get("color", "white")
            
            if result["status"] == "completado":
                stats = result["stats"]
                table_tlc.add_row(
                    f"[{color}]{service_name}[/{color}]",
                    "[green] OK[/green]",
                    f"[green]{stats['successful']}[/green]",
                    f"[blue]{stats['skipped']}[/blue]",
                    f"[red]{stats['failed']}[/red]" if stats['failed'] > 0 else "[dim]0[/dim]",
                    f"{stats['total']}"
                )
                total_downloaded += stats['successful']
                total_skipped += stats['skipped']
                total_failed += stats['failed']
                total_files += stats['total']
            else:
                table_tlc.add_row(
                    f"[{color}]{service_name}[/{color}]",
                    "[red] ERROR[/red]",
                    "[dim]-[/dim]",
                    "[dim]-[/dim]",
                    "[dim]-[/dim]",
                    "[dim]-[/dim]"
                )
                services_with_errors += 1
        
        # Añadir fila de totales
        table_tlc.add_section()
        table_tlc.add_row(
            "[bold]TOTAL TLC[/bold]",
            "",
            f"[bold green]{total_downloaded}[/bold green]",
            f"[bold blue]{total_skipped}[/bold blue]",
            f"[bold red]{total_failed}[/bold red]" if total_failed > 0 else "[bold dim]0[/bold dim]",
            f"[bold]{total_files}[/bold]"
        )
        
        console.print(table_tlc)
        console.print()
    
    # Tabla de eventos
    if events_results:
        table_events = Table(
            title="Eventos NYC",
            show_header=True,
            header_style="bold magenta"
        )
        table_events.add_column("Dataset", style="magenta", width=20)
        table_events.add_column("Estado", justify="center", width=12)
        table_events.add_column("Descargados", justify="right", width=12)
        table_events.add_column("Omitidos", justify="right", width=12)
        table_events.add_column("Fallidos", justify="right", width=12)
        table_events.add_column("Total", justify="right", width=12)
        
        for key, result in events_results.items():
            color = result.get("color", "magenta")
            
            if result["status"] == "completado":
                stats = result["stats"]
                # Manejar tanto 'ok' como 'successful' por compatibilidad
                downloaded = stats.get('ok', stats.get('successful', 0))
                
                table_events.add_row(
                    f"[{color}]NYC Events[/{color}]",
                    "[green] OK[/green]",
                    f"[green]{downloaded}[/green]",
                    f"[blue]{stats['skipped']}[/blue]",
                    f"[red]{stats['failed']}[/red]" if stats['failed'] > 0 else "[dim]0[/dim]",
                    f"{stats['total']}"
                )
            else:
                table_events.add_row(
                    f"[{color}]NYC Events[/{color}]",
                    "[red]✗ ERROR[/red]",
                    "[dim]-[/dim]",
                    "[dim]-[/dim]",
                    "[dim]-[/dim]",
                    "[dim]-[/dim]"
                )
        
        console.print(table_events)
        console.print()
    
    # Tabla de datos meteorológicos
    if meteo_results:
        table_meteo = Table(
            title="Datos Meteorológicos",
            show_header=True,
            header_style="bold blue"
        )
        table_meteo.add_column("Fuente", style="blue", width=20)
        table_meteo.add_column("Estado", justify="center", width=12)
        table_meteo.add_column("Descargados", justify="right", width=12)
        table_meteo.add_column("Omitidos", justify="right", width=12)
        table_meteo.add_column("Fallidos", justify="right", width=12)
        table_meteo.add_column("Total", justify="right", width=12)
        
        for key, result in meteo_results.items():
            color = result.get("color", "blue")
            
            if result["status"] == "completado":
                stats = result["stats"]
                # Manejar tanto 'ok' como 'successful' por compatibilidad
                downloaded = stats.get('ok', stats.get('successful', 0))
                
                table_meteo.add_row(
                    f"[{color}]Open-Meteo NYC[/{color}]",
                    "[green] OK[/green]",
                    f"[green]{downloaded}[/green]",
                    f"[blue]{stats['skipped']}[/blue]",
                    f"[red]{stats['failed']}[/red]" if stats['failed'] > 0 else "[dim]0[/dim]",
                    f"{stats['total']}"
                )
            else:
                table_meteo.add_row(
                    f"[{color}]Open-Meteo NYC[/{color}]",
                    "[red]✗ ERROR[/red]",
                    "[dim]-[/dim]",
                    "[dim]-[/dim]",
                    "[dim]-[/dim]",
                    "[dim]-[/dim]"
                )
        
        console.print(table_meteo)
        console.print()
    
    # Mensaje final
    total_errors = sum(1 for r in results.values() if r["status"] == "error")
    total_items = len(results)
    
    if total_errors == 0:
        console.print(Panel.fit(
            "[bold green] DESCARGA MASIVA COMPLETADA EXITOSAMENTE[/bold green]",
            border_style="bright_green"
        ))
    else:
        console.print(Panel.fit(
            f"[bold yellow]⚠ DESCARGA COMPLETADA CON ADVERTENCIAS[/bold yellow]\n"
            f"[dim]Items con problemas: {total_errors}/{total_items}[/dim]",
            border_style="yellow"
        ))
    
    console.print()


# =============================================================================
# Función principal
# =============================================================================

def download_all(mode: DownloadMode = DownloadMode.ALL):
    """
    Descarga datos según el modo especificado.
    
    Args:
        mode: Modo de descarga (all, tlc, events, meteo)
    """
    results = {}
    
    # Header global
    console.print()
    console.print(Panel.fit(
        "[bold white]NYC DATA PIPELINE[/bold white]\n"
        "[cyan]Sistema de Descarga Masiva[/cyan]",
        border_style="bright_cyan"
    ))
    
    if mode in (DownloadMode.ALL, DownloadMode.TLC):
        tlc_results = download_all_services()
        results.update(tlc_results)
    
    if mode in (DownloadMode.ALL, DownloadMode.EVENTS):
        events_results = download_all_events()
        results.update(events_results)
    
    if mode in (DownloadMode.ALL, DownloadMode.METEO):
        meteo_results = download_all_meteo()
        results.update(meteo_results)
    
    # Mostrar resumen
    print_summary(results)


# =============================================================================
# CLI
# =============================================================================

@click.command()
@click.option(
    "--mode",
    type=click.Choice([m.value for m in DownloadMode], case_sensitive=False),
    default=DownloadMode.ALL.value,
    show_default=True,
    help="Modo de descarga: 'all' (TLC + eventos + meteo), 'tlc' (solo servicios TLC), 'events' (solo eventos NYC), 'meteo' (solo datos meteorológicos)"
)
def main(mode: str):
    """
    Descarga masiva de datos de NYC (TLC, eventos y/o meteorológicos).
    
    Ejemplos de uso:
    
        # Descargar todo (TLC + eventos + meteo)
        uv run -m src.extraccion.descarga_masiva
        
        # Solo servicios TLC
        uv run -m src.extraccion.descarga_masiva --mode tlc
        
        # Solo eventos NYC
        uv run -m src.extraccion.descarga_masiva --mode events
        
        # Solo datos meteorológicos
        uv run -m src.extraccion.descarga_masiva --mode meteo
    """
    download_all(DownloadMode(mode))


if __name__ == "__main__":
    main()