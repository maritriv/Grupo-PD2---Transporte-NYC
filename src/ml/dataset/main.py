from __future__ import annotations

from pathlib import Path

from config.pipeline_runner import ejecutar_modulo, print_done, print_stage


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    print_stage("ML DATASET", "Build + Split + Feature preprocessing")

    jobs = [
        {
            "name": "ej1a",
            "build_module": "src.ml.dataset.a_build_dataset_ej1a",
            "input_path": "data/ml/dataset_demanda_completo.parquet",
            "prefix": "dataset_demanda_completo",
        },
        {
            "name": "ej1b",
            "build_module": "src.ml.dataset.a_build_dataset_ej1b",
            "input_path": "data/ml/dataset_propinas_completo.parquet",
            "prefix": "dataset_propinas_completo",
        },
        {
            "name": "ej2",
            "build_module": "src.ml.dataset.a_build_dataset_ej2",
            "input_path": "data/ml/dataset_completo.parquet",
            "prefix": "dataset_completo",
        },
    ]

    for job in jobs:
        print_stage("BUILD DATASET", f"Construyendo {job['name']}")
        ejecutar_modulo(job["build_module"], [], project_root)

        dataset_fp = project_root / job["input_path"]
        if not dataset_fp.exists():
            raise FileNotFoundError(
                f"No se ha generado el dataset esperado para {job['name']}: {dataset_fp}"
            )

        print_stage("SPLIT DATASET", f"Generando splits para {job['name']}")
        ejecutar_modulo(
            "src.ml.dataset.b_split_dataset",
            [
                "--input", job["input_path"],
                "--prefix", job["prefix"],
            ],
            project_root,
        )

    print_done("DATASET ML COMPLETADO CORRECTAMENTE")


if __name__ == "__main__":
    main()