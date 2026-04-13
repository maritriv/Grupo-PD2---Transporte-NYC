"""
Ejercicio 1a: Predecir zona de máxima demanda

Wrapper que ejecuta el análisis de demanda a nivel zona-hora-día
con features enriquecidas (lags, rolling means, variables externas).

Genera: data/aggregated/ex1a/df_demand_zone_hour_day/
"""
from __future__ import annotations

from src.procesamiento.capa3.pipelines.run_demand_zone import main


if __name__ == "__main__":
    main()
