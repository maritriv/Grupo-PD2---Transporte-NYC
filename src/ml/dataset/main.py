from __future__ import annotations

from pathlib import Path

from config.pipeline_runner import ejecutar_modulo, print_done, print_stage


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    print_stage("ML DATASET", "Build + Split + Feature preprocessing")

    modulos = [
        "src.ml.dataset.a_build_dataset",
        "src.ml.dataset.b_split_dataset",
    ]

    for m in modulos:
        ejecutar_modulo(m, [], project_root)

    print_done("DATASET ML COMPLETADO CORRECTAMENTE")


if __name__ == "__main__":
    main()