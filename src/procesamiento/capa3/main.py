from __future__ import annotations

from pathlib import Path

from config.pipeline_runner import ejecutar_modulo, print_done, print_stage


def main() -> None:
    """
    Orquestador principal de Capa 3.
    
    Estructura en 2 pasos:
    1. AGREGADOS BASE: Tablas reutilizables para análisis (TLC, eventos, meteo, rent, restaurants)
    2. DATASETS EJERCICIOS: Datasets model-ready para cada ejercicio (1a, 1b, 1c, 1d, 2)
    """
    project_root = Path(__file__).resolve().parents[3]
    print_stage("CAPA 3", "Agregaciones y datasets para análisis/ML")

    # =========================================================================
    # PASO 1: Agregados base (reutilizables, independientes)
    # =========================================================================
    print_stage("PASO 1", "Agregados base")
    
    agregados = [
        "src.procesamiento.capa3.aggregates.tlc",
        "src.procesamiento.capa3.aggregates.eventos",
        "src.procesamiento.capa3.aggregates.meteo",
        "src.procesamiento.capa3.aggregates.rent",
        "src.procesamiento.capa3.aggregates.restaurants",
    ]
    
    for m in agregados:
        ejecutar_modulo(m, [], project_root)

    # =========================================================================
    # PASO 2: Datasets para ejercicios (dependen de Paso 1)
    # =========================================================================
    print_stage("PASO 2", "Ensamblaje datasets para ejercicios")
    
    ejercicios = [
        "src.procesamiento.capa3.ejercicios.ex1a_demand",   # Ej.1a: predecir zona máxima demanda
        "src.procesamiento.capa3.ejercicios.ex1b_tips",     # Ej.1b: predecir propina
        "src.procesamiento.capa3.ejercicios.ex1c_patterns", # Ej.1c: clasif. demanda bajo/medio/alto
        "src.procesamiento.capa3.ejercicios.ex1d_socioeconomic", # Ej.1d: poder adquisitivo
        "src.procesamiento.capa3.ejercicios.ex2_stress",    # Ej.2: estrés urbano
    ]
    
    for m in ejercicios:
        ejecutar_modulo(m, [], project_root)

    print_done("CAPA 3 COMPLETADA CORRECTAMENTE")


if __name__ == "__main__":
    main()
