# src/procesamiento/capa1/valid_location_ids.py
import logging
import pandas as pd
from typing import Optional, Set, Dict
from config.settings import obtener_ruta

logger = logging.getLogger(__name__)

def load_location_mapping() -> Optional[Dict[int, str]]:
    """
    Carga el CSV de zonas TLC y devuelve un diccionario:
    { LocationID: Borough }
    """
    location_csv = obtener_ruta('data/external/taxi_zone_lookup.csv')

    if not location_csv.exists():
        logger.warning("Taxi zone lookup does not exist")
        return None

    zones = pd.read_csv(location_csv)
    if "LocationID" not in zones.columns or "Borough" not in zones.columns:
        raise ValueError(f"El CSV de zonas TLC no contiene las columnas necesarias: {location_csv}")

    zones["LocationID"] = pd.to_numeric(zones["LocationID"], errors="coerce").dropna().astype(int)
    zones['Borough'] = zones['Borough'].astype(str).str.strip().str.title()

    return dict(zip(zones["LocationID"], zones["Borough"]))

location_to_borough: Optional[Dict[int, str]] = load_location_mapping()
valid_location_ids: Optional[Set[int]] = set(location_to_borough.keys()) if location_to_borough else None