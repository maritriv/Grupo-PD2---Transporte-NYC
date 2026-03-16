# src/extraccion/download_rent_data.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import pandas as pd
import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from config.settings import obtener_ruta

"""
download_rent_data.py
│
├── Configuración
├── Helpers (descarga HTTP, lectura CSV/CSV.GZ)
├── Lógica de negocio (download_rent_snapshot)
└── CLI (Click)
"""

# =============================================================================
# Configuración
# =============================================================================
# Nota: esto es "rent proxy" vía Airbnb (precio por noche). Es útil como señal socioeconómica.
DEFAULT_URL_GZ = (
    "https://data.insideairbnb.com/united-states/ny/new-york-city/2024-01-05/data/listings.csv.gz"
)
DEFAULT_URL_CSV = (
    "https://data.insideairbnb.com/united-states/ny/new-york-city/2024-01-05/visualisations/listings.csv"
)

TIMEOUT = 120
DEFAULT_OUT_DIR = obtener_ruta("data/external/rent/raw")

console = Console()


# =============================================================================
# Helpers
# =============================================================================
def _download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"[dim]  → Descargando: {url}[/dim]")

    with requests.get(url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[cyan]{task.fields[mb]} MB"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Descargando", total=total or None, mb=0.0)

            downloaded = 0
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress.update(task, completed=downloaded if total else None, mb=downloaded / 1e6)

    console.print("[dim]  → Descarga completada[/dim]")


def _read_insideairbnb_csv(path: Path) -> pd.DataFrame:
    # pandas detecta gzip por extensión si compression="infer"
    return pd.read_csv(path, compression="infer", low_memory=False).astype(str)


# =============================================================================
# Lógica de negocio
# =============================================================================
def download_rent_snapshot(
    url: str,
    out_dir: Optional[Path] = None,
    tag: str = "rent_insideairbnb",
) -> Path:
    """
    Descarga un snapshot de listings de InsideAirbnb y lo guarda en parquet.
    """
    out_dir = out_dir or DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Nombres
    filename = url.split("?")[0].split("/")[-1]
    tmp_path = out_dir / f"__tmp_{filename}"
    final_parquet = out_dir / f"{tag}.parquet"

    if final_parquet.exists():
        console.print(f"[yellow]SKIP:[/yellow] {final_parquet.name}")
        return final_parquet

    # Descarga
    _download_file(url, tmp_path)

    # Leer y guardar
    console.print("[dim]  → Leyendo CSV y generando Parquet...[/dim]")
    df = _read_insideairbnb_csv(tmp_path)
    df.to_parquet(final_parquet, engine="pyarrow", index=False)

    # Limpiar tmp
    try:
        tmp_path.unlink()
    except Exception:
        pass

    console.print(f"[green]OK:[/green] {final_parquet.name} ({len(df)} filas)")
    return final_parquet


# =============================================================================
# CLI
# =============================================================================
@click.command()
@click.option("--url", default=DEFAULT_URL_GZ, show_default=True, help="URL de InsideAirbnb (listings.csv.gz o listings.csv)")
@click.option("--out-dir", default=None, help="Ruta destino (si no se pasa, usa la de settings)")
@click.option("--tag", default="rent_insideairbnb", show_default=True, help="Nombre base del parquet final")
def main(url: str, out_dir: Optional[str], tag: str):
    """
    Descarga un snapshot de 'alquiler' (proxy) desde InsideAirbnb para NYC.

    Ejemplos:

        uv run -m src.extraccion.download_rent_data
        uv run -m src.extraccion.download_rent_data --url "https://.../listings.csv"
        uv run -m src.extraccion.download_rent_data --tag rent_2025Q4
    """
    out_path = Path(out_dir) if out_dir else None
    download_rent_snapshot(url=url, out_dir=out_path, tag=tag)


if __name__ == "__main__":
    main()