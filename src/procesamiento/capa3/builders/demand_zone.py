from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from rich.table import Table

from src.procesamiento.capa3.common.constants import ALLOWED_MAX_DATE, ALLOWED_MIN_DATE, console
from src.procesamiento.capa3.common.externals import load_event_features, load_meteo_features
from src.procesamiento.capa3.common.io import iter_month_partitions, write_partitioned_dataset

NEEDED_TLC_COLS = [
    "date",
    "year",
    "month",
    "hour",
    "day_of_week",
    "is_weekend",
    "week_of_year",
    "pu_location_id",
    "is_valid_for_demand",
]

# OJO: no incluimos num_trips en la salida final para no duplicar target
FINAL_COLS = [
    "timestamp_hour",
    "date",
    "month",
    "hour",
    "hour_block_3h",
    "day_of_week",
    "is_weekend",
    "pu_location_id",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_3h",
    "rolling_mean_24h",
    # Meteo
    "temp_c",
    "precip_mm",
    # Eventos ciudad-hora
    "city_n_events",
    "city_has_event",
    # Target
    "target_n_trips",
]


def normalize_tlc(
    df: pd.DataFrame,
    min_date: str,
    max_date: str,
) -> pd.DataFrame:
    missing = [c for c in NEEDED_TLC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas TLC requeridas: {missing}")

    out = df[NEEDED_TLC_COLS].copy()

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int32")
    out["month"] = pd.to_numeric(out["month"], errors="coerce").astype("Int32")
    out["hour"] = pd.to_numeric(out["hour"], errors="coerce").astype("Int32")
    out["day_of_week"] = pd.to_numeric(out["day_of_week"], errors="coerce").astype("Int32")
    out["is_weekend"] = pd.to_numeric(out["is_weekend"], errors="coerce").astype("Int8")
    out["week_of_year"] = pd.to_numeric(out["week_of_year"], errors="coerce").astype("Int32")
    out["pu_location_id"] = pd.to_numeric(out["pu_location_id"], errors="coerce").astype("Int32")
    out["is_valid_for_demand"] = pd.to_numeric(out["is_valid_for_demand"], errors="coerce").astype("Int8")

    out = out.dropna(
        subset=[
            "date",
            "year",
            "month",
            "hour",
            "day_of_week",
            "is_weekend",
            "week_of_year",
            "pu_location_id",
            "is_valid_for_demand",
        ]
    )

    dmin = max(pd.to_datetime(min_date), ALLOWED_MIN_DATE)
    dmax = min(pd.to_datetime(max_date), ALLOWED_MAX_DATE)

    out = out[(out["date"] >= dmin) & (out["date"] <= dmax)]
    out = out[(out["hour"] >= 0) & (out["hour"] <= 23)]
    out = out[out["is_valid_for_demand"] == 1]

    return out


def aggregate_tlc_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Todos los servicios juntos.
    1 fila = (date, hour, pu_location_id)
    """
    if df.empty:
        return pd.DataFrame()

    g = (
        df.groupby(
            [
                "date",
                "year",
                "month",
                "hour",
                "day_of_week",
                "is_weekend",
                "week_of_year",
                "pu_location_id",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="num_trips")
    )

    g["timestamp_hour"] = pd.to_datetime(g["date"], errors="coerce") + pd.to_timedelta(
        pd.to_numeric(g["hour"], errors="coerce"), unit="h"
    )
    g["num_trips"] = pd.to_numeric(g["num_trips"], errors="coerce").fillna(0).astype("Int32")

    return g[
        [
            "timestamp_hour",
            "date",
            "year",
            "month",
            "hour",
            "day_of_week",
            "is_weekend",
            "week_of_year",
            "pu_location_id",
            "num_trips",
        ]
    ].copy()


def build_complete_grid(df_base: pd.DataFrame) -> pd.DataFrame:
    """
    Completa:
      todas las timestamp_hour observadas x todas las zonas observadas
    donde falte una combinacion -> num_trips = 0
    """
    if df_base.empty:
        return df_base

    ts_min = pd.to_datetime(df_base["timestamp_hour"], errors="coerce").min()
    ts_max = pd.to_datetime(df_base["timestamp_hour"], errors="coerce").max()
    if pd.isna(ts_min) or pd.isna(ts_max):
        return pd.DataFrame(columns=df_base.columns)

    # Serie horaria continua para que lag_1h/24h/168h representen horas reales.
    timestamps = pd.date_range(start=ts_min, end=ts_max, freq="h")
    zones = pd.Index(sorted(df_base["pu_location_id"].dropna().unique()))

    full_index = pd.MultiIndex.from_product(
        [timestamps, zones],
        names=["timestamp_hour", "pu_location_id"],
    )

    base_idx = df_base.set_index(["timestamp_hour", "pu_location_id"])
    df_full = base_idx.reindex(full_index).reset_index()

    ts = pd.to_datetime(df_full["timestamp_hour"], errors="coerce")
    df_full["date"] = ts.dt.floor("D")
    df_full["year"] = ts.dt.year.astype("Int32")
    df_full["month"] = ts.dt.month.astype("Int32")
    df_full["hour"] = ts.dt.hour.astype("Int32")

    # misma convencion que capa 2:
    # domingo=1, lunes=2, ..., sabado=7
    dow0 = ts.dt.dayofweek
    df_full["day_of_week"] = (((dow0 + 1) % 7) + 1).astype("Int32")
    df_full["is_weekend"] = df_full["day_of_week"].isin([1, 7]).astype("Int8")
    df_full["week_of_year"] = ts.dt.isocalendar().week.astype("Int32")

    df_full["pu_location_id"] = pd.to_numeric(df_full["pu_location_id"], errors="coerce").astype("Int32")
    df_full["num_trips"] = pd.to_numeric(df_full["num_trips"], errors="coerce").fillna(0).astype("Int32")

    return df_full[
        [
            "timestamp_hour",
            "date",
            "year",
            "month",
            "hour",
            "day_of_week",
            "is_weekend",
            "week_of_year",
            "pu_location_id",
            "num_trips",
        ]
    ].copy()


def build_model_table(
    df_base_full: pd.DataFrame,
    df_meteo: pd.DataFrame,
    df_events: pd.DataFrame,
    drop_na_lags: bool = True,
) -> pd.DataFrame:
    if df_base_full.empty:
        return pd.DataFrame(columns=FINAL_COLS)

    df = df_base_full.sort_values(["pu_location_id", "timestamp_hour"]).copy()
    df["num_trips"] = pd.to_numeric(df["num_trips"], errors="coerce").astype("float32")

    grp = df.groupby("pu_location_id", sort=False)
    df["lag_1h"] = grp["num_trips"].shift(1)
    df["lag_24h"] = grp["num_trips"].shift(24)
    df["lag_168h"] = grp["num_trips"].shift(168)

    lag_1h = pd.to_numeric(df["lag_1h"], errors="coerce")
    df["rolling_mean_3h"] = (
        lag_1h.groupby(df["pu_location_id"])
        .rolling(window=3, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["rolling_mean_24h"] = (
        lag_1h.groupby(df["pu_location_id"])
        .rolling(window=24, min_periods=24)
        .mean()
        .reset_index(level=0, drop=True)
    )

    if df_meteo is not None and not df_meteo.empty:
        df = df.merge(df_meteo, on=["date", "hour"], how="left")

    if df_events is not None and not df_events.empty:
        df = df.merge(df_events, on=["date", "hour"], how="left")
    else:
        df["city_n_events"] = 0.0
        df["city_has_event"] = 0

    # El target es la demanda real de esa fila
    df["target_n_trips"] = pd.to_numeric(df["num_trips"], errors="coerce").astype("Int32")
    df["hour_block_3h"] = (pd.to_numeric(df["hour"], errors="coerce") // 3).astype("Int8")

    if "city_n_events" not in df.columns:
        df["city_n_events"] = 0.0
    df["city_n_events"] = pd.to_numeric(df["city_n_events"], errors="coerce").fillna(0).astype("float32")

    if "city_has_event" in df.columns:
        city_has_event = pd.to_numeric(df["city_has_event"], errors="coerce")
        city_has_event = city_has_event.fillna((df["city_n_events"] > 0).astype("Int8"))
        df["city_has_event"] = city_has_event.astype("Int8")
    else:
        df["city_has_event"] = (df["city_n_events"] > 0).astype("Int8")

    # Tipado
    for c in [
        "lag_1h",
        "lag_24h",
        "lag_168h",
        "rolling_mean_3h",
        "rolling_mean_24h",
        "temp_c",
        "precip_mm",
        "city_n_events",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    for c in [
        "month",
        "hour",
        "day_of_week",
        "pu_location_id",
        "target_n_trips",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int32")

    for c in ["is_weekend", "city_has_event", "hour_block_3h"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int8")

    if drop_na_lags:
        needed = [
            "lag_1h",
            "lag_24h",
            "lag_168h",
            "rolling_mean_3h",
            "rolling_mean_24h",
        ]
        df = df.dropna(subset=needed).copy()

    for c in FINAL_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    df = df.sort_values(["timestamp_hour", "pu_location_id"]).reset_index(drop=True)
    return df[FINAL_COLS].copy()


@dataclass
class BuildStats:
    months_detected: int = 0
    files_detected: int = 0
    rows_input: int = 0
    rows_after_filter: int = 0
    rows_base_aggregated: int = 0
    rows_base_completed: int = 0
    rows_out: int = 0
    unique_zones: int = 0
    unique_timestamps: int = 0
    meteo_rows: int = 0
    events_rows: int = 0


def build_demand_zone_dataset(
    layer2_path: Path,
    out_base: Path,
    meteo_base: Path,
    events_base: Path,
    min_date: str,
    max_date: str,
    drop_na_lags: bool = True,
    output_dataset_name: str = "df_demand_zone_hour_day",
) -> BuildStats:
    stats = BuildStats()

    month_parts = list(iter_month_partitions(layer2_path))
    if not month_parts:
        raise FileNotFoundError(
            f"No hay parquets dentro de {layer2_path} con estructura year=YYYY/month=MM"
        )

    stats.months_detected = len(month_parts)
    stats.files_detected = sum(len(files) for _, _, files in month_parts)

    info = Table(show_header=True, header_style="bold white", title="Entrada EX1(a) TLC")
    info.add_column("Metrica", style="bold cyan")
    info.add_column("Valor", justify="right")
    info.add_row("Meses detectados", f"{stats.months_detected:,}")
    info.add_row("Parquets detectados", f"{stats.files_detected:,}")
    console.print(info)

    monthly_bases: List[pd.DataFrame] = []
    month_summary_rows: List[Tuple[str, int, int, int]] = []

    for mi, (year, month, files) in enumerate(month_parts, start=1):
        dfs_month: List[pd.DataFrame] = []

        with console.status(
            f"[cyan]Procesando mes {mi}/{stats.months_detected}: {year}-{month:02d} ({len(files)} parquets)...[/cyan]"
        ):
            for fp in files:
                df_raw = pd.read_parquet(fp, columns=NEEDED_TLC_COLS)
                stats.rows_input += len(df_raw)

                df = normalize_tlc(df_raw, min_date=min_date, max_date=max_date)
                stats.rows_after_filter += len(df)

                if not df.empty:
                    dfs_month.append(df)

            if dfs_month:
                df_month = pd.concat(dfs_month, ignore_index=True)
                df_base_month = aggregate_tlc_month(df_month)
                stats.rows_base_aggregated += len(df_base_month)
                monthly_bases.append(df_base_month)

                month_summary_rows.append(
                    (
                        f"{year}-{month:02d}",
                        len(files),
                        len(df_month),
                        len(df_base_month),
                    )
                )
            else:
                month_summary_rows.append((f"{year}-{month:02d}", len(files), 0, 0))

    if not monthly_bases:
        raise ValueError("Tras filtrar, no quedaron datos validos para demanda.")

    df_base = pd.concat(monthly_bases, ignore_index=True)
    df_base = df_base.sort_values(["timestamp_hour", "pu_location_id"]).reset_index(drop=True)

    stats.unique_zones = int(df_base["pu_location_id"].nunique())
    stats.unique_timestamps = int(df_base["timestamp_hour"].nunique())

    df_base_full = build_complete_grid(df_base)
    stats.rows_base_completed = len(df_base_full)

    df_meteo = load_meteo_features(meteo_base, min_date=min_date, max_date=max_date)
    stats.meteo_rows = len(df_meteo)

    df_events = load_event_features(events_base, min_date=min_date, max_date=max_date)
    stats.events_rows = len(df_events)

    df_out = build_model_table(
        df_base_full=df_base_full,
        df_meteo=df_meteo,
        df_events=df_events,
        drop_na_lags=drop_na_lags,
    )
    stats.rows_out = len(df_out)

    write_partitioned_dataset(
        df_out,
        out_base / output_dataset_name,
        partition_cols=["date"],
    )

    month_summary = Table(show_header=True, header_style="bold magenta", title="Resumen mensual EX1(a)")
    month_summary.add_column("Mes", style="bold white")
    month_summary.add_column("Parquets", justify="right")
    month_summary.add_column("Filas validas", justify="right")
    month_summary.add_column("Filas base agg", justify="right")
    for mes, n_files, n_valid, n_base in month_summary_rows:
        month_summary.add_row(mes, f"{n_files:,}", f"{n_valid:,}", f"{n_base:,}")
    console.print(month_summary)

    outputs = Table(show_header=True, header_style="bold magenta", title="Salida EX1(a)")
    outputs.add_column("Dataset", style="bold white")
    outputs.add_column("Path")
    outputs.add_row(output_dataset_name, str(out_base / output_dataset_name))
    console.print(outputs)

    return stats

