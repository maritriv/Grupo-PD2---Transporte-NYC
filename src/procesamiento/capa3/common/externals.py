from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .constants import ALLOWED_MAX_DATE, ALLOWED_MIN_DATE, console
from .io import list_all_parquets


def load_meteo_features(
    meteo_base: Path,
    min_date: str,
    max_date: str,
) -> pd.DataFrame:
    files = list_all_parquets(meteo_base)
    if not files:
        console.print(f"[yellow]Aviso:[/yellow] No se encontraron parquets de meteo en {meteo_base}")
        return pd.DataFrame(columns=["date", "hour", "temp_c", "precip_mm"])

    dfs: List[pd.DataFrame] = []
    keep_cols = ["date", "hour", "temp_c", "precip_mm"]

    for fp in files:
        df = pd.read_parquet(fp, columns=keep_cols).copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype("Int32")
        df["temp_c"] = pd.to_numeric(df["temp_c"], errors="coerce")
        df["precip_mm"] = pd.to_numeric(df["precip_mm"], errors="coerce")
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True).dropna(subset=["date", "hour"])

    dmin = max(pd.to_datetime(min_date), ALLOWED_MIN_DATE)
    dmax = min(pd.to_datetime(max_date), ALLOWED_MAX_DATE)
    out = out[(out["date"] >= dmin) & (out["date"] <= dmax)]
    out = out[(out["hour"] >= 0) & (out["hour"] <= 23)]

    out = (
        out.groupby(["date", "hour"], dropna=False)
        .agg(
            temp_c=("temp_c", "mean"),
            precip_mm=("precip_mm", "mean"),
        )
        .reset_index()
    )

    out["temp_c"] = pd.to_numeric(out["temp_c"], errors="coerce").astype("float32")
    out["precip_mm"] = pd.to_numeric(out["precip_mm"], errors="coerce").astype("float32")
    return out


def load_event_features(
    events_base: Path,
    min_date: str,
    max_date: str,
) -> pd.DataFrame:
    files = list_all_parquets(events_base)
    if not files:
        console.print(f"[yellow]Aviso:[/yellow] No se encontraron parquets de eventos en {events_base}")
        return pd.DataFrame(columns=["date", "hour", "city_n_events", "city_has_event"])

    dfs: List[pd.DataFrame] = []
    keep_cols = ["date", "hour", "n_events"]

    for fp in files:
        df = pd.read_parquet(fp, columns=keep_cols).copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype("Int32")
        df["n_events"] = pd.to_numeric(df["n_events"], errors="coerce").astype("Int32")
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True).dropna(subset=["date", "hour", "n_events"])

    dmin = max(pd.to_datetime(min_date), ALLOWED_MIN_DATE)
    dmax = min(pd.to_datetime(max_date), ALLOWED_MAX_DATE)
    out = out[(out["date"] >= dmin) & (out["date"] <= dmax)]
    out = out[(out["hour"] >= 0) & (out["hour"] <= 23)]

    agg = (
        out.groupby(["date", "hour"], dropna=False)
        .agg(city_n_events=("n_events", "sum"))
        .reset_index()
    )

    agg["city_n_events"] = pd.to_numeric(agg["city_n_events"], errors="coerce").fillna(0).astype("float32")
    agg["city_has_event"] = (agg["city_n_events"] > 0).astype("Int8")
    return agg

