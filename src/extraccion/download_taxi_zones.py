# src/extraccion/download_taxi_zones.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import pandas as pd
import requests
from rich.console import Console

from config.settings import obtener_ruta

# =============================================================================
# Configuración
# =============================================================================
URL_TAXI_ZONES = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
DEFAULT_OUT_DIR = obtener_ruta("data/external")

console = Console()

# =============================================================================
# Lógica de negocio
# =============================================================================
def download_taxi_lookup(
    url: str = URL_TAXI_ZONES,
    out_dir: Optional[Path] = None
) -> Path:
    """
    Descarga el maestro de zonas de taxi (NYC) y lo guarda como CSV.
    Equivalente a: wget -P data/external/ URL
    """
    out_dir = out_dir or DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    target_path = out_dir / "taxi_zone_lookup.csv"

    if target_path.exists():
        console.print(f"[yellow]SKIP:[/yellow] {target_path.name} ya existe.")
        return target_path

    console.print(f"[cyan]Descargando maestro de zonas...[/cyan]")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(target_path, "wb") as f:
            f.write(response.content)
            
        console.print(f"[green]OK:[/green] Guardado en {target_path}")
    except Exception as e:
        console.print(f"[bold red]Error al descargar:[/bold red] {e}")
        raise

    return target_path

# =============================================================================
# CLI
# =============================================================================
@click.command()
@click.option("--out-dir", default=None, help="Carpeta de destino")
def main(out_dir: Optional[str]):
    """
    Descarga el CSV de referencia de zonas de Taxi de NYC.
    """
    path = Path(out_dir) if out_dir else None
    download_taxi_lookup(out_dir=path)

if __name__ == "__main__":
    main()