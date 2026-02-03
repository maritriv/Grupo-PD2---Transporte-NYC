# src/features/build_features.py
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from download_tlc_data import download_service_data

console = Console()

# | Tipo servicio           | Descripción breve                                                | Años aproximados disponibles               |
# | ----------------------- | ---------------------------------------------------------------- | ------------------------------------------ |
# | Yellow Taxi             | Taxis amarillos tradicionales en toda la ciudad.                 | Desde 2009–2010 hasta 2025+.               |
# | Green Taxi              | Taxis "boro" (fuera de Manhattan core).                          | Desde ~2013–2014 hasta 2025+.              |
# | FHV                     | For-Hire Vehicles (livery, black car, etc.) no high volume.      | Desde ~2015 hasta 2025+.                   |
# | High Volume FHV (HVFHV) | Uber, Lyft y otras bases de "alto volumen".                      | Desde 2019 hasta 2025+.                    |


def download_all_services(start_year: int = 2023, end_year: int = 2025):
    """
    Descarga datos de todos los servicios de NYC TLC.
    
    Args:
        start_year: Año inicial de descarga
        end_year: Año final de descarga (inclusive)
    """
    services = {
        'yellow': ('Yellow Taxi (taxis amarillos tradicionales)', 'yellow'),
        'green': ('Green Taxi (taxis "boro" fuera de Manhattan)', 'green'),
        'fhv': ('FHV (For-Hire Vehicles, livery, black car)', 'blue'),
        'fhvhv': ('HVFHV (Uber, Lyft, alto volumen)', 'magenta')
    }
    
    total_services = len(services)
    
    # Header principal
    console.print()
    console.print(Panel.fit(
        "[bold white]NYC TAXI & LIMOUSINE COMMISSION[/bold white]\n"
        "[cyan]Descarga Masiva de Datos[/cyan]",
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
                months=range(1, 13),
                data_dir=Path("data/raw")
            )
            results[service_key] = {
                "status": "completado",
                "stats": stats,
                "color": color
            }
            console.print(f"[bold green]{description} - Completado[/bold green]\n")
            
        except Exception as e:
            console.print(f"[bold red]ERROR:[/bold red] {e}")
            console.print("[yellow]Continuando con el siguiente servicio...[/yellow]\n")
            results[service_key] = {
                "status": "error",
                "error": str(e),
                "color": color
            }
            continue
    
    # Resumen final
    console.print()
    console.rule("[bold cyan]RESUMEN DE DESCARGA MASIVA[/bold cyan]", style="cyan")
    console.print()
    
    # Crear tabla de resumen
    table = Table(title="Estadísticas por Servicio", show_header=True, header_style="bold cyan")
    table.add_column("Servicio", style="cyan", width=20)
    table.add_column("Estado", justify="center", width=12)
    table.add_column("Descargados", justify="right", width=12)
    table.add_column("Omitidos", justify="right", width=12)
    table.add_column("Fallidos", justify="right", width=12)
    table.add_column("Total", justify="right", width=12)
    
    # Totales generales
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0
    total_files = 0
    services_with_errors = 0
    
    for service_key, result in results.items():
        service_name = services[service_key][0].split(' (')[0]  # Nombre corto
        color = result["color"]
        
        if result["status"] == "completado":
            stats = result["stats"]
            table.add_row(
                f"[{color}]{service_name}[/{color}]",
                "[green]OK[/green]",
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
            table.add_row(
                f"[{color}]{service_name}[/{color}]",
                "[red]ERROR[/red]",
                "[dim]-[/dim]",
                "[dim]-[/dim]",
                "[dim]-[/dim]",
                "[dim]-[/dim]"
            )
            services_with_errors += 1
    
    # Añadir fila de totales
    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]",
        "",
        f"[bold green]{total_downloaded}[/bold green]",
        f"[bold blue]{total_skipped}[/bold blue]",
        f"[bold red]{total_failed}[/bold red]" if total_failed > 0 else "[bold dim]0[/bold dim]",
        f"[bold]{total_files}[/bold]"
    )
    
    console.print(table)
    console.print()
    
    # Mensaje final según el resultado
    if services_with_errors == 0 and total_failed == 0:
        console.print(Panel.fit(
            "[bold green]DESCARGA MASIVA COMPLETADA EXITOSAMENTE[/bold green]\n"
            f"[dim]Total de archivos procesados: {total_files}[/dim]",
            border_style="bright_green"
        ))
    elif services_with_errors > 0:
        console.print(Panel.fit(
            f"[bold yellow]DESCARGA COMPLETADA CON ADVERTENCIAS[/bold yellow]\n"
            f"[dim]Servicios con problemas: {services_with_errors}/{total_services}[/dim]\n"
            f"[dim]Archivos fallidos: {total_failed}[/dim]",
            border_style="yellow"
        ))
    else:
        console.print(Panel.fit(
            f"[bold yellow]DESCARGA COMPLETADA CON ERRORES MENORES[/bold yellow]\n"
            f"[dim]Archivos fallidos: {total_failed}/{total_files}[/dim]",
            border_style="yellow"
        ))
    
    console.print()


if __name__ == "__main__":
    download_all_services(start_year=2020, end_year=2025)