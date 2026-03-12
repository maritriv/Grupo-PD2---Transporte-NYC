from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def ejecutar_modulo(modulo: str, project_root: Path) -> None:
    """
    Ejecuta un módulo Python (python -m ...) usando el intérprete activo (uv/venv),
    desde la raíz del proyecto para que las rutas relativas funcionen correctamente.
    """
    print(f"\n[INFO] Ejecutando: {modulo}")
    print("-" * 50)

    result = subprocess.run(
        [sys.executable, "-m", modulo],
        cwd=str(project_root),
    )

    if result.returncode != 0:
        raise RuntimeError(f"Error ejecutando {modulo}")


def main() -> None:
    # Estamos en: src/procesamiento/capa1/main.py
    # Raíz del proyecto = 3 niveles arriba
    project_root = Path(__file__).resolve().parents[3]

    modulos = [
        "src.procesamiento.capa1.capa1_green",
        "src.procesamiento.capa1.capa1_yellow",
        "src.procesamiento.capa1.capa1_fhvhv",
        "src.procesamiento.capa1.capa1_eventos",
        "src.procesamiento.capa1.capa1_meteo",
    ]

    for m in modulos:
        ejecutar_modulo(m, project_root)

    print("\nCAPA 1 COMPLETADA CORRECTAMENTE")


if __name__ == "__main__":
    main()
