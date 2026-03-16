from __future__ import annotations

from pathlib import Path

from config.pipeline_runner import ejecutar_modulo, print_done, print_stage

def main() -> None:
    # Estamos en: src/procesamiento/capa2/main.py
    # Raiz del proyecto = 3 niveles arriba
    project_root = Path(__file__).resolve().parents[3]
    print_stage("CAPA 2", "Estandarizacion y joins por dominio")

    modulos = [
        "src.procesamiento.capa2.capa2_eventos",
        "src.procesamiento.capa2.capa2_meteo",
        "src.procesamiento.capa2.capa2_tlc",
    ]

    for m in modulos:
        ejecutar_modulo(m, [], project_root)

    print_done("CAPA 2 COMPLETADA CORRECTAMENTE")


if __name__ == "__main__":
    main()
