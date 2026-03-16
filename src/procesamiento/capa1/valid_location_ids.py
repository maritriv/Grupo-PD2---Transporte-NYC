# src/procesamiento/capa1/valid_location_ids.py
import logging
import pandas as pd
from typing import Optional, Set
from config.settings import obtener_ruta

logger = logging.getLogger(__name__)

def load_valid_location_ids() -> Optional[Set[int]]:
    """
    Carga IDs válidos de zonas TLC desde un CSV, si existe.
    Espera una columna LocationID.
    Si no hay CSV, devuelve None y simplemente no se aplica esta validación.
    """
    location_csv = obtener_ruta('data/external/taxi_zone_lookup.csv')

    if not location_csv.exists():
        logger.warning("Taxi zone lookup does not exist")
        return None

    zones = pd.read_csv(location_csv)
    if "LocationID" not in zones.columns:
        raise ValueError(f"El CSV de zonas TLC no contiene columna 'LocationID': {location_csv}")

    ids = pd.to_numeric(zones["LocationID"], errors="coerce").dropna().astype(int)
    return set(ids.tolist())

valid_location_ids: Optional[Set[int]] = load_valid_location_ids()