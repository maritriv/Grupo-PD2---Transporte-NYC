# src/data/download_tlc_data.py
import itertools
from pathlib import Path
from typing import Optional

import click
import requests


# Configuración base
BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


def build_url(service: str, year: int, month: int) -> str:
    """Construye la URL del archivo parquet según el formato TLC."""
    return f"{BASE_URL}/{service}_tripdata_{year}-{month:02d}.parquet"


def download_file(url: str, dest_path: Path) -> bool:
    """
    Descarga un archivo desde una URL.
    
    Returns:
        bool: True si se descargó correctamente, False en caso contrario.
    """
    if dest_path.exists():
        print(f"Saltando (ya existe): {dest_path.name}")
        return True
    
    print(f"Descargando: {url}")
    try:
        resp = requests.get(url, stream=True, timeout=30)
        if resp.status_code == 200:
            # Crear directorio si no existe
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Guardado en: {dest_path}")
            return True
        else:
            print(f"ERROR {resp.status_code} al descargar {url}")
            return False
    except Exception as e:
        print(f"Excepción al descargar {url}: {e}")
        return False


def download_service_data(
    service: str,
    years: range,
    months: range,
    data_dir: Optional[Path] = None
):
    """
    Descarga datos de un servicio específico para los años y meses indicados.
    
    Args:
        service: Tipo de servicio ("yellow", "green", "fhv", "fhvhv")
        years: Rango de años a descargar
        months: Rango de meses a descargar
        data_dir: Directorio base donde guardar los datos (por defecto data/raw)
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    
    # Crear subdirectorio para el servicio
    service_dir = data_dir / service
    service_dir.mkdir(parents=True, exist_ok=True)
    
    total = len(list(itertools.product(years, months)))
    current = 0
    successful = 0
    
    print(f"\nDescargando datos de '{service}' taxi")
    print(f"Años: {min(years)}-{max(years)}, Meses: {min(months)}-{max(months)}")
    print(f"Destino: {service_dir}")
    print(f"--" * 60)
    
    for year, month in itertools.product(years, months):
        current += 1
        url = build_url(service, year, month)
        filename = f"{service}_tripdata_{year}-{month:02d}.parquet"
        dest_path = service_dir / filename
        
        print(f"\n[{current}/{total}] ", end="")
        if download_file(url, dest_path):
            successful += 1
    
    print(f"\n{'--' * 60}")
    print(f"Completado: {successful}/{total} archivos descargados correctamente")


@click.command()
@click.option(
    "--service",
    "-s",
    type=click.Choice(["yellow", "green", "fhv", "fhvhv"], case_sensitive=False),
    required=True,
    help="Tipo de servicio de taxi a descargar"
)
@click.option(
    "--start-year",
    type=int,
    default=2023,
    help="Año inicial (inclusive)"
)
@click.option(
    "--end-year",
    type=int,
    default=2025,
    help="Año final (inclusive)"
)
@click.option(
    "--start-month",
    type=click.IntRange(1, 12),
    default=1,
    help="Mes inicial (1-12)"
)
@click.option(
    "--end-month",
    type=click.IntRange(1, 12),
    default=12,
    help="Mes final (1-12)"
)
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directorio base donde guardar los datos (por defecto: data/raw)"
)
def main(
    service: str,
    start_year: int,
    end_year: int,
    start_month: int,
    end_month: int,
    data_dir: Optional[Path]
):
    """
    Descarga datos de NYC TLC (Taxi & Limousine Commission).
    
    Ejemplos de uso:
    
        uv run -m src.data.download_tlc_data --service yellow
        
        uv run -m src.data.download_tlc_data -s green --start-year 2024 --end-year 2024
        
        uv run -m src.data.download_tlc_data -s fhvhv --start-year 2023 --end-year 2023 --start-month 6 --end-month 8
    """
    years = range(start_year, end_year + 1)
    months = range(start_month, end_month + 1)
    
    download_service_data(
        service=service.lower(),
        years=years,
        months=months,
        data_dir=data_dir
    )


if __name__ == "__main__":
    main()