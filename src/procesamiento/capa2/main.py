from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def ejecutar_modulo(modulo: str, project_root: Path) -> None:
    """
    Ejecuta un módulo Python (python -m ...) usando el intérprete activo (uv/venv),
    desde la raíz del proyecto para que las rutas relativas funcionen bien.
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
    # Estamos en: src/procesamiento/capa2/main.py -> raíz = 3 parents arriba
    project_root = Path(__file__).resolve().parents[3]

    modulos = [
        "src.procesamiento.capa2.capa2_eventos",
        "src.procesamiento.capa2.capa2_meteo",
        "src.procesamiento.capa2.capa2_tlc",
    ]

    for m in modulos:
        ejecutar_modulo(m, project_root)

    print("\nCAPA 2 COMPLETADA CORRECTAMENTE")


if __name__ == "__main__":
    main()