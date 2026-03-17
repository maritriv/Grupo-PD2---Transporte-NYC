from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.pipeline_runner import console
from src.ml.dataset.modules.display import print_step_status


def load_zone_lookup(
    project_root: Path,
    path: str = "data/external/taxi_zone_lookup.csv",
) -> pd.DataFrame | None:
    """Carga el lookup de zonas TLC para mapear pu_location_id -> borough."""
    fp = (project_root / path).resolve()
    if not fp.exists():
        console.print(f"[bold yellow]WARNING[/bold yellow] No se encontró taxi_zone_lookup.csv en: {fp}")
        return None

    zones = pd.read_csv(fp)
    expected_candidates = {"LocationID", "Borough"}
    if not expected_candidates.issubset(set(zones.columns)):
        console.print(
            f"[bold yellow]WARNING[/bold yellow] taxi_zone_lookup.csv no tiene columnas esperadas {expected_candidates}"
        )
        return None

    zones = zones[["LocationID", "Borough"]].copy()
    zones.columns = ["pu_location_id", "borough"]
    zones["pu_location_id"] = pd.to_numeric(zones["pu_location_id"], errors="coerce").astype("Int64")
    zones["borough"] = zones["borough"].astype(str).str.strip()

    zones.loc[zones["borough"].isin(["Unknown", "N/A", "NA", "nan"]), "borough"] = pd.NA
    zones = zones.dropna(subset=["pu_location_id"]).drop_duplicates(subset=["pu_location_id"])

    print_step_status("Zone lookup", f"cargado desde {fp} | rows={len(zones):,}")
    return zones


def add_borough_to_tlc(tlc: pd.DataFrame, zones: pd.DataFrame | None) -> pd.DataFrame:
    """Añade borough al dataset TLC a partir del lookup de zonas."""
    if zones is None:
        tlc = tlc.copy()
        tlc["borough"] = pd.NA
        return tlc

    tlc = tlc.merge(zones, on="pu_location_id", how="left")
    borough_non_null = tlc["borough"].notna().sum()
    print_step_status("Borough", f"añadido a TLC | filas con borough={borough_non_null:,}")
    return tlc