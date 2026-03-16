from __future__ import annotations

from pathlib import Path

from src.pipeline_runner import ejecutar_modulo, print_done, print_stage


def main() -> None:
    # Estamos en: src/procesamiento/capa3/main.py
    # Raiz del proyecto = 3 niveles arriba
    project_root = Path(__file__).resolve().parents[3]
    print_stage("CAPA 3", "Agregaciones finales para analitica y ML")

    modulos = [
        "src.procesamiento.capa3.capa3_eventos",
        "src.procesamiento.capa3.capa3_meteo",
        "src.procesamiento.capa3.capa3_tlc",
    ]

    for m in modulos:
        ejecutar_modulo(m, project_root)

    print_done("CAPA 3 COMPLETADA CORRECTAMENTE")


if __name__ == "__main__":
    main()
