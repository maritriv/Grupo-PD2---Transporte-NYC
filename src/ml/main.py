from __future__ import annotations

from pathlib import Path

from src.pipeline_runner import ejecutar_modulo, print_done, print_stage


def main() -> None:
    # Estamos en: src/ml/main.py
    # Subimos 2 niveles para llegar a la raiz del proyecto
    project_root = Path(__file__).resolve().parents[2]
    print_stage("PIPELINE ML", "Dataset + Modelos")

    modulos = [
        "src.ml.dataset.main",
        "src.ml.models.main",
    ]

    for m in modulos:
        ejecutar_modulo(m, project_root)

    print_done("PIPELINE ML COMPLETADO CORRECTAMENTE")


if __name__ == "__main__":
    main()
