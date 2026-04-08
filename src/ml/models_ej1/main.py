from __future__ import annotations

from pathlib import Path

from config.pipeline_runner import ejecutar_modulo, print_done, print_stage


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    print_stage("MODELOS EJ1", "Ejecución de modelos del ejercicio 2")

    modulos = [
        "src.ml.models_ej1.model_a_demanda",
        "src.ml.models_ej1.model_b_propinas",
    ]

    for m in modulos:
        ejecutar_modulo(m, [], project_root)

    print_done("MODELOS EJ2 COMPLETADOS CORRECTAMENTE")


if __name__ == "__main__":
    main()
