from __future__ import annotations

from pathlib import Path

from config.pipeline_runner import ejecutar_modulo, print_done, print_stage


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    print_stage("PIPELINE ML", "Dataset + Modelos")

    modulos = [
        "src.ml.dataset.main",
        "src.ml.models_ej1.main",
        "src.ml.models_ej2.main",
    ]

    for m in modulos:
        ejecutar_modulo(m, [], project_root)

    print_done("PIPELINE ML COMPLETADO CORRECTAMENTE")


if __name__ == "__main__":
    main()


    from __future__ import annotations