from __future__ import annotations

from pathlib import Path

from config.pipeline_runner import console, ejecutar_modulo, print_done, print_stage


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    print_stage("MODELOS EJ1", "Ejecucion de modelos del ejercicio 1")

    modulos = [
        ("src.ml.models_ej1.model_a_demanda", project_root / "src/ml/models_ej1/model_a_demanda.py"),
        ("src.ml.models_ej1.model_b_propinas", project_root / "src/ml/models_ej1/model_b_propinas.py"),
    ]

    for m, file_path in modulos:
        if not file_path.exists() or file_path.stat().st_size == 0:
            console.print(f"[yellow][WARN][/yellow] Se omite {m} porque {file_path.name} esta vacio.")
            continue
        ejecutar_modulo(m, [], project_root)

    print_done("MODELOS EJ1 COMPLETADOS CORRECTAMENTE")


if __name__ == "__main__":
    main()
