"""
Ejercicio 2: Mapa de estrés urbano

Dashboard de visualización de estabilidad del sistema urbano de transporte.

Combina demanda + variabilidad de precio para identificar:
- Zonas y horas con comportamiento inestable
- Contextos críticos para operación
- Patrones recurrentes vs episodios puntuales

Genera: data/aggregated/ex2/df_stress_urban_zone_hour/

Wrapper que ejecuta la lógica de construcción de stress urbano.
"""
from __future__ import annotations

from src.procesamiento.capa3.pipelines.run_stress_zone import main


if __name__ == "__main__":
    main()
