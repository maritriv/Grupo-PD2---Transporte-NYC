# src/extraccion/download_taxi_zones.py
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Optional

import click
import requests
from rich.console import Console

from config.settings import obtener_ruta

# =============================================================================
# Configuración
# =============================================================================
URL_TAXI_ZONES_CSV = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
URL_TAXI_ZONES_ZIP = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
DEFAULT_OUT_DIR = obtener_ruta("data/external")

console = Console()

# =============================================================================
# Lógica de negocio
# =============================================================================
def download_taxi_lookup(
    url: str = URL_TAXI_ZONES_CSV,
    out_dir: Optional[Path] = None
) -> Path:
    """
    Descarga el maestro de zonas de taxi (NYC) y lo guarda como CSV.
    """
    out_dir = out_dir or DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    target_path = out_dir / "taxi_zone_lookup.csv"

    if target_path.exists():
        console.print(f"[yellow]SKIP:[/yellow] {target_path.name} ya existe.")
        return target_path

    console.print(f"[cyan]Descargando maestro de zonas (CSV)...[/cyan]")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(target_path, "wb") as f:
            f.write(response.content)
            
        console.print(f"[green]OK:[/green] Guardado en {target_path}")
    except Exception as e:
        console.print(f"[bold red]Error al descargar el CSV:[/bold red] {e}")
        raise

    return target_path

def download_taxi_shapefile(
    url: str = URL_TAXI_ZONES_ZIP,
    out_dir: Optional[Path] = None
) -> Path:
    """
    Descarga y descomprime el shapefile de zonas de Taxi de NYC.
    """
    out_dir = out_dir or DEFAULT_OUT_DIR
    
    # Creamos la subcarpeta específica para el shapefile
    target_dir = out_dir / "taxi_zones"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Comprobamos si ya está extraído (buscando el archivo principal .shp)
    shapefile_path = target_dir / "taxi_zones.shp"

    if shapefile_path.exists():
        console.print(f"[yellow]SKIP:[/yellow] Los Shapefiles ya existen en {target_dir.name}/.")
        return target_dir

    console.print(f"[cyan]Descargando y extrayendo shapefiles (ZIP)...[/cyan]")
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Leemos el ZIP directamente desde la memoria (RAM) y lo extraemos
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(path=out_dir)
            
        console.print(f"[green]OK:[/green] Shapefiles extraídos en {target_dir}")
    except Exception as e:
        console.print(f"[bold red]Error al procesar el ZIP:[/bold red] {e}")
        raise

    return target_dir

# =============================================================================
# CLI
# =============================================================================
@click.command()
@click.option("--out-dir", default=None, help="Carpeta de destino")
def main(out_dir: Optional[str]):
    """
    Descarga el CSV y los Shapefiles de referencia de zonas de Taxi de NYC.
    """
    path = Path(out_dir) if out_dir else None
    
    # Ejecutamos ambas descargas
    download_taxi_lookup(out_dir=path)
    download_taxi_shapefile(out_dir=path)

if __name__ == "__main__":
    main()