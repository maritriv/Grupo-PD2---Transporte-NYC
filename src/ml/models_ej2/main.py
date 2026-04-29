from __future__ import annotations

from pathlib import Path

from config.pipeline_runner import ejecutar_modulo, print_done, print_stage

TARGET_COL = "target_stress_t24"
TARGET_CLF_COL = "target_is_stress_t24"


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    print_stage(
        "ML MODELOS EJERCICIO 2",
        "Entrenamiento y comparación de modelos a 24 horas",
    )

    runs = [
        (
            "src.ml.models_ej2.a_model_baseline",
            [
                "--target-col", TARGET_COL,
                "--outputs-dir", "outputs/ejercicio2/reports/baseline",
            ],
        ),
        (
            "src.ml.models_ej2.b_model_random_forest",
            [
                "--target-col", TARGET_COL,
                "--outputs-dir", "outputs/ejercicio2/reports/random_forest",
            ],
        ),
        (
            "src.ml.models_ej2.c_model_boosting",
            [
                "--target-reg", TARGET_COL,
                "--target-clf", TARGET_CLF_COL,
                "--outputs-dir", "outputs/ejercicio2/reports/boosting",
            ],
        ),
        (
            "src.ml.models_ej2.d_model_nn",
            [
                "--target-col", TARGET_COL,
                "--outputs-dir", "outputs/ejercicio2/reports/neural_network",
            ],
        ),
        (
            "src.ml.models_ej2.e_model_spark",
            [
                "--target-col", TARGET_COL,
                "--outputs-dir", "outputs/ejercicio2/reports/gbt",
            ],
        ),
        (
            "src.ml.models_ej2.f_compare_reports",
            [],
        ),
    ]

    for modulo, args in runs:
        ejecutar_modulo(modulo, args, project_root)

    print_done("MODELOS EJERCICIO 2 COMPLETADOS Y COMPARADOS")


if __name__ == "__main__":
    main()