from __future__ import annotations

from pathlib import Path

from config.pipeline_runner import ejecutar_modulo, print_done, print_stage


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    print_stage("ML MODELOS EJERCICIO 2", "Baselines y modelos finales")

    modulos = [
        "src.ml.models_ej2.a_model_baseline",
        "src.ml.models_ej2.b_model_random_forest",
        "src.ml.models_ej2.c_model_boosting",
        "src.ml.models_ej2.d_model_nn",
        "src.ml.models_ej2.e_model_spark",
    ]

    for m in modulos:
        ejecutar_modulo(m, [], project_root)

    print_done("MODELOS ML COMPLETADOS CORRECTAMENTE")


if __name__ == "__main__":
    main()