from __future__ import annotations

from pathlib import Path

from config.pipeline_runner import ejecutar_modulo, print_done, print_stage


def main() -> None:
    # Estamos en: src/main.py
    # Subimos 1 nivel para llegar a la raiz del proyecto
    project_root = Path(__file__).resolve().parents[1]
    print_stage("PIPELINE PRINCIPAL", "Extraccion + Procesamiento")

    modulos = [
        "src.extraccion.main",
        "src.procesamiento.main",
    ]

    for m in modulos:
        ejecutar_modulo(m, [], project_root)

    print_done("PIPELINE COMPLETO EJECUTADO CORRECTAMENTE")


if __name__ == "__main__":
    main()
