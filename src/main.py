from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def ejecutar_modulo(modulo: str, project_root: Path) -> None:
    """
    Ejecuta un módulo Python usando el intérprete activo (uv/venv),
    desde la raíz del proyecto.
    """
    print(f"\n[INFO] Ejecutando: {modulo}")
    print("-" * 60)

    result = subprocess.run(
        [sys.executable, "-m", modulo],
        cwd=str(project_root),
    )

    if result.returncode != 0:
        raise RuntimeError(f"Error ejecutando {modulo}")


def main() -> None:
    # Estamos en: src/main.py
    # Subimos 1 nivel para llegar a la raíz del proyecto
    project_root = Path(__file__).resolve().parents[1]

    modulos = [
        "src.extraccion.main",
        "src.procesamiento.main",
    ]

    for m in modulos:
        ejecutar_modulo(m, project_root)

    print("\nPIPELINE COMPLETO EJECUTADO CORRECTAMENTE")


if __name__ == "__main__":
    main()