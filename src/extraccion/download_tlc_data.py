# src/extraccion/download_tlc_data.py
import itertools
from pathlib import Path
from typing import Optional, Sequence

import click
import requests
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
)

from config.settings import obtener_ruta, config

# Configuración base
BASE_URL = config["descarga"]["url_base"]
DEFAULT_DATA_DIR = obtener_ruta("data/raw")

# Consola Rich
console = Console()

# Diccionario de errores HTTP comunes
HTTP_ERRORS = {
    400: "Bad Request - Solicitud mal formada",
    401: "Unauthorized - Se requiere autenticación",
    403: "Forbidden - Acceso prohibido al recurso",
    404: "Not Found - Archivo no encontrado",
    429: "Too Many Requests - Demasiadas solicitudes",
    500: "Internal Server Error - Error del servidor",
    502: "Bad Gateway - Gateway no válido",
    503: "Service Unavailable - Servicio no disponible",
    504: "Gateway Timeout - Timeout del gateway",
}

ALL_SERVICES = ("yellow", "green", "fhv", "fhvhv")


def build_url(service: str, year: int, month: int) -> str:
    """Construye la URL del archivo parquet según el formato TLC."""
    return f"{BASE_URL}/{service}_tripdata_{year}-{month:02d}.parquet"


def get_http_error_description(status_code: int) -> str:
    """Devuelve una descripción legible del código de error HTTP."""
    return HTTP_ERRORS.get(status_code, "Error desconocido")


def download_file(url: str, dest_path: Path) -> bool:
    """
    Descarga un archivo desde una URL.

    Returns:
        bool: True si se descargó correctamente, False en caso contrario.
    """
    if dest_path.exists():
        console.print(f"[dim]SKIP: Ya existe: {dest_path.name}[/dim]")
        return True

    console.print(f"[cyan]Descargando:[/cyan] [dim]{url}[/dim]")

    try:
        resp = requests.get(url, stream=True, timeout=30)

        if resp.status_code == 200:
            # Crear directorio si no existe
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Obtener tamaño del archivo si está disponible
            total_size = int(resp.headers.get("content-length", 0))

            with open(dest_path, "wb") as f:
                if total_size > 0:
                    # Con progress bar si conocemos el tamaño
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        DownloadColumn(),
                        TransferSpeedColumn(),
                        console=console,
                        transient=True,  # Desaparece cuando termina
                    ) as progress:
                        task = progress.add_task(f"[cyan]{dest_path.name}", total=total_size)
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                progress.update(task, advance=len(chunk))
                else:
                    # Sin progress bar si no conocemos el tamaño
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            return True

        error_desc = get_http_error_description(resp.status_code)
        console.print(f"[red]ERROR {resp.status_code}:[/red] {error_desc}")
        console.print(f"[dim]   URL: {url}[/dim]")
        return False

    except requests.exceptions.Timeout:
        console.print("[red]TIMEOUT:[/red] La descarga tardó demasiado (>30s)")
        console.print(f"[dim]   URL: {url}[/dim]")
        return False
    except requests.exceptions.ConnectionError:
        console.print("[red]CONNECTION ERROR:[/red] No se pudo conectar al servidor")
        console.print(f"[dim]   URL: {url}[/dim]")
        return False
    except requests.exceptions.RequestException as e:
        console.print(f"[red]NETWORK ERROR:[/red] {str(e)}")
        console.print(f"[dim]   URL: {url}[/dim]")
        return False
    except IOError as e:
        console.print("[red]IO ERROR:[/red] No se pudo guardar el archivo")
        console.print(f"[dim]   Ruta: {dest_path}[/dim]")
        console.print(f"[dim]   Detalle: {str(e)}[/dim]")
        return False
    except Exception as e:
        console.print(f"[red]UNEXPECTED ERROR:[/red] {type(e).__name__}")
        console.print(f"[dim]   Detalle: {str(e)}[/dim]")
        return False


def download_service_data(
    service: str,
    years: range,
    months: range,
    data_dir: Optional[Path] = None,
):
    """
    Descarga datos de un servicio específico para los años y meses indicados.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    # Crear subdirectorio para el servicio
    service_dir = data_dir / service
    service_dir.mkdir(parents=True, exist_ok=True)

    total = len(list(itertools.product(years, months)))
    successful = 0
    skipped = 0
    failed = 0

    console.print(f"\n[bold]Descargando datos de '{service}'[/bold]")
    console.print(f"[yellow]Años:[/yellow] {min(years)}-{max(years)}, [yellow]Meses:[/yellow] {min(months)}-{max(months)}")
    console.print(f"[yellow]Destino:[/yellow] {service_dir}")
    console.rule(style="dim")

    for idx, (year, month) in enumerate(itertools.product(years, months), 1):
        url = build_url(service, year, month)
        filename = f"{service}_tripdata_{year}-{month:02d}.parquet"
        dest_path = service_dir / filename

        console.print(f"\n[bold cyan][{idx}/{total}][/bold cyan] ", end="")

        if dest_path.exists():
            console.print(f"[dim]SKIP: Ya existe: {dest_path.name}[/dim]")
            skipped += 1
        else:
            if download_file(url, dest_path):
                successful += 1
            else:
                failed += 1

    # Resumen final
    console.rule(style="dim")
    console.print(f"Completado: {successful}/{total} descargados | {skipped} omitidos | {failed} fallidos")

    if failed > 0:
        console.print(f"[yellow]ADVERTENCIA: {failed} archivo(s) no se pudieron descargar[/yellow]")

    return {"total": total, "successful": successful, "failed": failed, "skipped": skipped}


@click.command()
@click.option(
    "--service",
    "-s",
    multiple=True,
    type=click.Choice(list(ALL_SERVICES), case_sensitive=False),
    help="Servicio(s) TLC a descargar. Si no se indica ninguno, se descargan todos.",
)
@click.option("--start-year", type=int, default=2023, help="Año inicial (inclusive)")
@click.option("--end-year", type=int, default=2025, help="Año final (inclusive)")
@click.option("--start-month", type=click.IntRange(1, 12), default=1, help="Mes inicial (1-12)")
@click.option("--end-month", type=click.IntRange(1, 12), default=12, help="Mes final (1-12)")
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directorio base donde guardar los datos (por defecto: data/raw)",
)
def main(
    service: Sequence[str],
    start_year: int,
    end_year: int,
    start_month: int,
    end_month: int,
    data_dir: Optional[Path],
):
    """
    Descarga datos de NYC TLC (Taxi & Limousine Commission).
    Si no se indica --service, descarga: yellow, green, fhv, fhvhv.
    """
    years = range(start_year, end_year + 1)
    months = range(start_month, end_month + 1)

    # Si no pasan servicios, descargamos todos
    services = [s.lower() for s in service] if service else list(ALL_SERVICES)

    console.print(f"[bold]Servicios seleccionados:[/bold] {', '.join(services)}")

    global_stats = {"total": 0, "successful": 0, "failed": 0, "skipped": 0}

    for s in services:
        stats = download_service_data(
            service=s,
            years=years,
            months=months,
            data_dir=data_dir,
        )
        for k in global_stats:
            global_stats[k] += stats[k]

    console.print("\n[bold green]RESUMEN GLOBAL[/bold green]")
    console.print(
        f"Total: {global_stats['total']} | "
        f"OK: {global_stats['successful']} | "
        f"SKIP: {global_stats['skipped']} | "
        f"FAILED: {global_stats['failed']}"
    )


if __name__ == "__main__":
    main()