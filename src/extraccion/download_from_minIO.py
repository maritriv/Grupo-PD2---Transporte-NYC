# src/extraccion/download_from_minIO.py
"""
Script para descargar todos los datos desde MinIO manteniendo la estructura de directorios.
"""
from pathlib import Path
from typing import Optional, List, Dict, Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TaskID,
)
from rich.table import Table

from config.minio_manager import MinioManager
from config.settings import obtener_ruta

console = Console()


def humanize_bytes(size_bytes: int) -> str:
    """Convierte bytes a formato legible (KB, MB, GB)."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def create_progress() -> Progress:
    """Crea una barra de progreso personalizada."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
        transient=False,
    )


def download_file_with_progress(
    minio: MinioManager,
    remote_path: str,
    local_path: Path,
    file_size: int,
    progress: Progress,
    task: TaskID,
) -> bool:
    """
    Descarga un archivo de MinIO mostrando el progreso.
    
    Args:
        minio: Instancia de MinioManager
        remote_path: Ruta del archivo en MinIO
        local_path: Ruta local donde guardar el archivo
        file_size: Tamaño del archivo en bytes
        progress: Objeto Progress de Rich
        task: ID de la tarea en el progress
        
    Returns:
        True si se descargó correctamente, False en caso contrario
    """
    try:
        # Crear directorio destino si no existe
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Descargar el archivo
        minio.descargar_archivo(remote_path, local_path)
        
        # Actualizar progreso
        progress.update(task, advance=file_size)
        
        return True
        
    except Exception as e:
        console.print(f"[red]ERROR al descargar {remote_path}:[/red] {str(e)}")
        return False


def download_from_minio(
    prefix: str = "data/",
    dest_dir: Optional[Path] = None,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """
    Descarga todos los archivos desde MinIO manteniendo la estructura de directorios.
    
    Args:
        prefix: Prefijo para filtrar archivos en MinIO (por defecto "data/")
        dest_dir: Directorio destino (por defecto: raíz del proyecto)
        skip_existing: Si True, omite archivos que ya existen localmente
        
    Returns:
        Diccionario con estadísticas de la descarga
    """
    # Inicializar MinioManager
    console.print("[cyan]Inicializando conexión con MinIO...[/cyan]")
    minio = MinioManager()
    
    # Directorio destino (raíz del proyecto si no se especifica)
    if dest_dir is None:
        dest_dir = obtener_ruta('data').parent
    else:
        dest_dir = Path(dest_dir)
    
    # Listar todos los archivos en MinIO
    console.print(f"[cyan]Listando archivos en MinIO (prefijo: {prefix})...[/cyan]")
    
    try:
        archivos = minio.listar_archivos(prefix=prefix, recursive=True)
    except Exception as e:
        console.print(f"[red]ERROR al listar archivos:[/red] {str(e)}")
        return {
            "total": 0,
            "successful": 0,
            "skipped": 0,
            "failed": -1,
            "total_size": 0
        }
    
    if not archivos:
        console.print("[yellow]No se encontraron archivos para descargar.[/yellow]")
        return {
            "total": 0,
            "successful": 0,
            "skipped": 0,
            "failed": 0,
            "total_size": 0
        }
    
    # Estadísticas
    total_files = len(archivos)
    total_size = sum(f["tamaño"] for f in archivos)
    successful = 0
    skipped = 0
    failed = 0
    downloaded_size = 0
    
    # Header
    console.print()
    console.print(Panel.fit(
        "[bold white]DESCARGA DESDE MINIO[/bold white]\n"
        f"[cyan]Archivos encontrados: {total_files}[/cyan]\n"
        f"[cyan]Tamaño total: {humanize_bytes(total_size)}[/cyan]",
        border_style="bright_blue"
    ))
    console.print(f"[yellow]Prefijo:[/yellow] {prefix}")
    console.print(f"[yellow]Destino:[/yellow] {dest_dir}")
    console.print(f"[yellow]Omitir existentes:[/yellow] {'Sí' if skip_existing else 'No'}")
    console.rule(style="dim")
    console.print()
    
    # Preparar archivos a descargar
    files_to_download = []
    size_to_download = 0
    
    for archivo in archivos:
        # Remover el prefijo base_dir de MinIO si existe
        remote_path = archivo["nombre"]
        if remote_path.startswith(minio.base_dir):
            remote_path = remote_path[len(minio.base_dir):]
        
        # Calcular ruta local
        local_path = dest_dir / remote_path
        
        # Verificar si ya existe
        if skip_existing and local_path.exists():
            skipped += 1
            console.print(f"[dim]SKIP: {remote_path}[/dim]")
        else:
            files_to_download.append({
                "remote": archivo["nombre"],
                "local": local_path,
                "size": archivo["tamaño"],
                "display_name": remote_path
            })
            size_to_download += archivo["tamaño"]
    
    # Si no hay nada que descargar
    if not files_to_download:
        console.print()
        console.print("[green]Todos los archivos ya están descargados[/green]")
        return {
            "total": total_files,
            "successful": 0,
            "skipped": skipped,
            "failed": 0,
            "total_size": total_size
        }
    
    # Descargar archivos
    console.print()
    console.print(f"[bold]Descargando {len(files_to_download)} archivos ({humanize_bytes(size_to_download)})...[/bold]")
    console.print()
    
    with create_progress() as progress:
        overall_task = progress.add_task(
            "[cyan]Progreso total",
            total=size_to_download
        )
        
        for idx, file_info in enumerate(files_to_download, 1):
            # Mostrar archivo actual
            file_task = progress.add_task(
                f"[cyan][{idx}/{len(files_to_download)}] {file_info['display_name']}",
                total=file_info['size']
            )
            
            # Descargar
            success = download_file_with_progress(
                minio=minio,
                remote_path=file_info["remote"],
                local_path=file_info["local"],
                file_size=file_info["size"],
                progress=progress,
                task=overall_task
            )
            
            if success:
                successful += 1
                downloaded_size += file_info["size"]
                progress.update(file_task, completed=file_info['size'])
            else:
                failed += 1
            
            # Remover tarea individual
            progress.remove_task(file_task)
    
    console.print()
    console.rule(style="dim")
    
    # Resumen
    stats = {
        "total": total_files,
        "successful": successful,
        "skipped": skipped,
        "failed": failed,
        "total_size": total_size,
        "downloaded_size": downloaded_size
    }
    
    return stats


def print_summary(stats: Dict[str, Any]):
    """Imprime un resumen de la descarga en formato tabla."""
    console.print()
    
    table = Table(
        title="Resumen de Descarga",
        show_header=True,
        header_style="bold cyan"
    )
    
    table.add_column("Métrica", style="cyan", width=25)
    table.add_column("Valor", justify="right", width=20)
    
    table.add_row("Total archivos", f"{stats['total']}")
    table.add_row(
        "Descargados",
        f"[green]{stats['successful']}[/green]"
    )
    table.add_row(
        "Omitidos (ya existían)",
        f"[blue]{stats['skipped']}[/blue]"
    )
    table.add_row(
        "Fallidos",
        f"[red]{stats['failed']}[/red]" if stats['failed'] > 0 else "[dim]0[/dim]"
    )
    
    table.add_section()
    table.add_row("Tamaño total", humanize_bytes(stats['total_size']))
    table.add_row(
        "Tamaño descargado",
        f"[green]{humanize_bytes(stats.get('downloaded_size', 0))}[/green]"
    )
    
    console.print(table)
    console.print()
    
    # Mensaje final
    if stats['failed'] == 0:
        console.print(Panel.fit(
            "[bold green]DESCARGA COMPLETADA EXITOSAMENTE[/bold green]",
            border_style="bright_green"
        ))
    else:
        console.print(Panel.fit(
            f"[bold yellow]DESCARGA COMPLETADA CON ADVERTENCIAS[/bold yellow]\n"
            f"[dim]Archivos con problemas: {stats['failed']}/{stats['total']}[/dim]",
            border_style="yellow"
        ))
    
    console.print()


@click.command()
@click.option(
    "--prefix",
    "-p",
    default="data/",
    show_default=True,
    help="Prefijo para filtrar archivos en MinIO (ej: 'data/raw/', 'data/external/')"
)
@click.option(
    "--dest-dir",
    "-d",
    type=click.Path(path_type=Path),
    default=None,
    help="Directorio destino (por defecto: raíz del proyecto)"
)
@click.option(
    "--no-skip",
    is_flag=True,
    default=False,
    help="Descargar todos los archivos, incluso si ya existen localmente"
)
def main(prefix: str, dest_dir: Optional[Path], no_skip: bool):
    """
    Descarga archivos desde MinIO manteniendo la estructura de directorios.
    
    Por defecto, descarga todos los archivos en la carpeta 'data/' de MinIO
    y los guarda en la raíz del proyecto manteniendo la misma estructura.
    
    Los archivos que ya existen localmente se omiten automáticamente
    (usa --no-skip para forzar la descarga).
    
    Ejemplos de uso:
    
        # Descargar todo desde data/
        uv run -m src.extraccion.download_from_minio
        
        # Descargar solo data/raw/
        uv run -m src.extraccion.download_from_minio --prefix data/raw/
        
        # Descargar a un directorio específico
        uv run -m src.extraccion.download_from_minio --dest-dir /path/to/dest
        
        # Forzar descarga de todos los archivos
        uv run -m src.extraccion.download_from_minio --no-skip
    """
    skip_existing = not no_skip
    
    stats = download_from_minio(
        prefix=prefix,
        dest_dir=dest_dir,
        skip_existing=skip_existing
    )
    
    print_summary(stats)


if __name__ == "__main__":
    main()