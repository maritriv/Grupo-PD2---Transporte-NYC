from __future__ import annotations

from pathlib import Path

from src.pipeline_runner import ejecutar_modulo, print_done, print_stage


def main() -> None:
    # Estamos en: src/ml/models/main.py
    # Subimos 3 niveles para llegar a la raiz del proyecto
    project_root = Path(__file__).resolve().parents[3]
    print_stage("ML MODELOS", "Baselines y modelos finales")

    modulos = [
        "src.ml.models.a_model_baseline",
        "src.ml.models.b_model_random_forest",
        "src.ml.models.c_model_boosting",
        "src.ml.models.d_model_nn",
        "src.ml.models.e_model_spark",
    ]

    for m in modulos:
        ejecutar_modulo(m, project_root)

    print_done("MODELOS ML COMPLETADOS CORRECTAMENTE")


if __name__ == "__main__":
    main()
