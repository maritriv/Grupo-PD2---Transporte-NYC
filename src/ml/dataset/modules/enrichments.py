from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pipeline_runner import console
from src.ml.dataset.modules.display import print_step_status
from src.ml.dataset.modules.io import ensure_cols, read_partitioned_parquet_dir


def join_meteo(
    tlc: pd.DataFrame,
    project_root: Path,
    meteo_path: str,
) -> pd.DataFrame:
    """Une meteo por date + hour."""
    meteo_fp = project_root / meteo_path
    if meteo_fp.exists():
        meteo = pd.read_parquet(meteo_fp)
        ensure_cols(meteo, ["date", "hour"], "METEO")

        meteo["date"] = pd.to_datetime(meteo["date"], errors="coerce").dt.date
        meteo["hour"] = pd.to_numeric(meteo["hour"], errors="coerce").astype("Int64")

        meteo_cols = ["date", "hour"]
        for c in [
            "rain_mm_sum",
            "temp_c_mean",
            "wind_kmh_mean",
            "precip_mm_sum",
            "snowfall_mm_sum",
            "weather_code_mode",
        ]:
            if c in meteo.columns:
                meteo_cols.append(c)

        meteo = meteo[meteo_cols].dropna(subset=["date", "hour"])
        df = tlc.merge(meteo, on=["date", "hour"], how="left")
        print_step_status("Meteo", f"unido | {len(df):,} filas | {len(df.columns)} columnas")
        return df

    console.print(f"[bold yellow]WARNING[/bold yellow] No se encontró meteo en: {meteo_fp}")
    df = tlc.copy()
    df["rain_mm_sum"] = pd.NA
    df["temp_c_mean"] = pd.NA
    df["wind_kmh_mean"] = pd.NA
    df["precip_mm_sum"] = pd.NA
    df["snowfall_mm_sum"] = pd.NA
    df["weather_code_mode"] = pd.NA
    return df


def join_events(
    df: pd.DataFrame,
    project_root: Path,
    events_dir: str,
) -> pd.DataFrame:
    """Une eventos agregados a nivel ciudad y borough."""
    events_base = project_root / events_dir
    if events_base.exists():
        ev = read_partitioned_parquet_dir(events_base)
        ensure_cols(ev, ["date", "hour", "n_events"], "EVENTOS")

        ev["date"] = pd.to_datetime(ev["date"], errors="coerce").dt.date
        ev["hour"] = pd.to_numeric(ev["hour"], errors="coerce").astype("Int64")
        ev["n_events"] = pd.to_numeric(ev["n_events"], errors="coerce").fillna(0)

        if "borough" in ev.columns:
            ev["borough"] = ev["borough"].astype(str).str.strip()
            ev.loc[ev["borough"].isin(["Unknown", "N/A", "NA", "nan"]), "borough"] = pd.NA

        ev_city = ev.groupby(["date", "hour"], as_index=False).agg(
            event_count_city=("n_events", "sum")
        )
        df = df.merge(ev_city, on=["date", "hour"], how="left")

        if "borough" in ev.columns:
            ev_borough = (
                ev.dropna(subset=["borough"])
                .groupby(["borough", "date", "hour"], as_index=False)
                .agg(event_count_borough=("n_events", "sum"))
            )
            df = df.merge(ev_borough, on=["borough", "date", "hour"], how="left")
        else:
            df["event_count_borough"] = 0
            console.print(
                "[bold yellow]WARNING[/bold yellow] EVENTOS sin columna 'borough': no se pudo generar event_count_borough"
            )
    else:
        console.print(f"[bold yellow]WARNING[/bold yellow] No se encontró eventos agregados en: {events_base}")
        df["event_count_city"] = 0
        df["event_count_borough"] = 0

    df["event_count_city"] = pd.to_numeric(df["event_count_city"], errors="coerce").fillna(0).astype(int)
    df["event_count_borough"] = pd.to_numeric(df["event_count_borough"], errors="coerce").fillna(0).astype(int)

    print_step_status(
        "Eventos",
        f"city>0: {(df['event_count_city'] > 0).sum():,} filas | borough>0: {(df['event_count_borough'] > 0).sum():,} filas",
    )
    return df