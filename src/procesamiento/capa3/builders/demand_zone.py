from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from rich.table import Table

from src.procesamiento.capa3.common.constants import ALLOWED_MAX_DATE, ALLOWED_MIN_DATE, console
from src.procesamiento.capa3.common.externals import (
    load_event_features,
    load_meteo_features,
    load_rent_zone_features_yearly,
    load_restaurants_zone_features_yearly,
)
from src.procesamiento.capa3.common.io import iter_month_partitions, safe_remove_dir, write_partitioned_dataset

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

# num_trips se conserva como predictora.
FINAL_COLS = [
    "timestamp_hour",
    "date",
    "year",
    "month",
    "hour",
    "hour_block_3h",
    "day_of_week",
    "is_weekend",
    "pu_location_id",
    "num_trips",
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
    # Variables estaticas por zona
    "n_restaurants_zone",
    "n_cuisines_zone",
    "rent_price_zone",
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


def _month_time_bounds(
    year: int,
    month: int,
    min_dt: pd.Timestamp,
    max_dt: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    month_start = pd.Timestamp(year=year, month=month, day=1)
    month_last_day = month_start + pd.offsets.MonthEnd(0)
    month_end = month_last_day + pd.Timedelta(hours=23)

    start = max(month_start, min_dt)
    end = min(month_end, max_dt)
    if start > end:
        return None
    return start, end


def _read_month_base(tmp_base_dir: Path, year: int, month: int) -> pd.DataFrame:
    part_dir = tmp_base_dir / f"year={year}" / f"month={month}"
    files = sorted(part_dir.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(fp) for fp in files], ignore_index=True)


def build_month_grid(
    df_base_month: pd.DataFrame,
    zone_ids: list[int],
    year: int,
    month: int,
    min_dt: pd.Timestamp,
    max_dt: pd.Timestamp,
) -> pd.DataFrame:
    bounds = _month_time_bounds(year, month, min_dt=min_dt, max_dt=max_dt)
    if bounds is None:
        return pd.DataFrame()
    start, end = bounds

    timestamps = pd.date_range(start=start, end=end, freq="h")
    if len(timestamps) == 0 or len(zone_ids) == 0:
        return pd.DataFrame()

    full_index = pd.MultiIndex.from_product(
        [timestamps, zone_ids],
        names=["timestamp_hour", "pu_location_id"],
    )

    if df_base_month.empty:
        base_idx = pd.DataFrame(columns=["timestamp_hour", "pu_location_id", "num_trips"])
        base_idx = base_idx.set_index(["timestamp_hour", "pu_location_id"])
    else:
        base_idx = df_base_month.set_index(["timestamp_hour", "pu_location_id"])[["num_trips"]]

    out = base_idx.reindex(full_index).reset_index()

    ts = pd.to_datetime(out["timestamp_hour"], errors="coerce")
    out["date"] = ts.dt.floor("D")
    out["year"] = ts.dt.year.astype("Int32")
    out["month"] = ts.dt.month.astype("Int32")
    out["hour"] = ts.dt.hour.astype("Int32")
    dow0 = ts.dt.dayofweek
    out["day_of_week"] = (((dow0 + 1) % 7) + 1).astype("Int32")
    out["is_weekend"] = out["day_of_week"].isin([1, 7]).astype("Int8")
    out["week_of_year"] = ts.dt.isocalendar().week.astype("Int32")

    out["pu_location_id"] = pd.to_numeric(out["pu_location_id"], errors="coerce").astype("Int32")
    out["num_trips"] = pd.to_numeric(out["num_trips"], errors="coerce").fillna(0).astype("Int32")

    return out[
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
    df_restaurants_zone_yearly: pd.DataFrame | None = None,
    df_rent_zone_yearly: pd.DataFrame | None = None,
    drop_na_lags: bool = True,
) -> pd.DataFrame:
    if df_base_full.empty:
        return pd.DataFrame(columns=FINAL_COLS)

    df = df_base_full.sort_values(["pu_location_id", "timestamp_hour"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype("Int32")
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

    def _lookup_by_year_zone(
        base_df: pd.DataFrame,
        features_df: pd.DataFrame | None,
        value_cols: list[str],
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=base_df.index)
        for c in value_cols:
            out[c] = pd.NA

        if features_df is None or features_df.empty:
            return out

        feat_cols = [c for c in ["year", "pu_location_id", *value_cols] if c in features_df.columns]
        feat = features_df[feat_cols].copy()
        if "year" not in feat.columns or "pu_location_id" not in feat.columns:
            return out

        feat["year"] = pd.to_numeric(feat["year"], errors="coerce").astype("Int32")
        feat["pu_location_id"] = pd.to_numeric(feat["pu_location_id"], errors="coerce").astype("Int32")
        feat = feat.dropna(subset=["year", "pu_location_id"])
        feat = feat.sort_values(["year", "pu_location_id"]).drop_duplicates(
            subset=["year", "pu_location_id"], keep="last"
        )
        for c in value_cols:
            if c in feat.columns:
                feat[c] = pd.to_numeric(feat[c], errors="coerce")
            else:
                feat[c] = pd.NA

        lookup = feat.set_index(["year", "pu_location_id"])[value_cols]
        key_index = pd.MultiIndex.from_arrays(
            [
                pd.to_numeric(base_df["year"], errors="coerce").astype("Int32"),
                pd.to_numeric(base_df["pu_location_id"], errors="coerce").astype("Int32"),
            ],
            names=["year", "pu_location_id"],
        )
        mapped = lookup.reindex(key_index).reset_index(drop=True)
        mapped.index = base_df.index
        return mapped

    mapped_rest = _lookup_by_year_zone(
        base_df=df,
        features_df=df_restaurants_zone_yearly,
        value_cols=["n_restaurants_zone", "n_cuisines_zone"],
    )
    df["n_restaurants_zone"] = mapped_rest["n_restaurants_zone"]
    df["n_cuisines_zone"] = mapped_rest["n_cuisines_zone"]

    mapped_rent = _lookup_by_year_zone(
        base_df=df,
        features_df=df_rent_zone_yearly,
        value_cols=["rent_price_zone"],
    )
    df["rent_price_zone"] = mapped_rent["rent_price_zone"]

    # Target a horizonte de 1 hora (t+1) por zona.
    df["target_n_trips"] = grp["num_trips"].shift(-1)
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

    for c in ["n_restaurants_zone", "n_cuisines_zone"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if "rent_price_zone" not in df.columns:
        df["rent_price_zone"] = pd.NA
    df["rent_price_zone"] = pd.to_numeric(df["rent_price_zone"], errors="coerce")

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
        "n_restaurants_zone",
        "n_cuisines_zone",
        "rent_price_zone",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    for c in [
        "year",
        "month",
        "hour",
        "day_of_week",
        "pu_location_id",
        "num_trips",
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
    restaurants_rows: int = 0
    rent_rows: int = 0


def build_demand_zone_dataset(
    layer2_path: Path,
    out_base: Path,
    meteo_base: Path,
    events_base: Path,
    restaurants_base: Path,
    rent_base: Path,
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
    min_dt = max(pd.to_datetime(min_date), ALLOWED_MIN_DATE)
    max_dt = min(pd.to_datetime(max_date), ALLOWED_MAX_DATE) + pd.Timedelta(hours=23)

    df_meteo = load_meteo_features(meteo_base, min_date=min_date, max_date=max_date)
    stats.meteo_rows = len(df_meteo)

    df_events = load_event_features(events_base, min_date=min_date, max_date=max_date)
    stats.events_rows = len(df_events)

    df_restaurants_zone = load_restaurants_zone_features_yearly(
        restaurants_base=restaurants_base,
        min_date=min_date,
        max_date=max_date,
    )
    stats.restaurants_rows = len(df_restaurants_zone)

    df_rent_zone = load_rent_zone_features_yearly(
        rent_base=rent_base,
        min_date=min_date,
        max_date=max_date,
    )
    stats.rent_rows = len(df_rent_zone)

    tmp_base_dir = out_base / "_tmp_demand_base"
    if tmp_base_dir.exists():
        safe_remove_dir(tmp_base_dir)
    tmp_base_dir.mkdir(parents=True, exist_ok=True)

    month_summary_rows: List[Tuple[str, int, int, int]] = []
    month_keys_written: list[tuple[int, int]] = []
    zone_ids_set: set[int] = set()
    seen_ts: set[pd.Timestamp] = set()

    base_cols = [
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

    try:
        # Pass 1: agregacion mensual a base compacta en disco.
        for mi, (year, month, files) in enumerate(month_parts, start=1):
            if _month_time_bounds(year, month, min_dt=min_dt, max_dt=max_dt) is None:
                month_summary_rows.append((f"{year}-{month:02d}", len(files), 0, 0))
                continue

            dfs_month: List[pd.DataFrame] = []
            with console.status(
                f"[cyan]Pass 1/2 - Base mensual {mi}/{stats.months_detected}: {year}-{month:02d} ({len(files)} parquets)...[/cyan]"
            ):
                for fp in files:
                    df_raw = pd.read_parquet(fp, columns=NEEDED_TLC_COLS)
                    stats.rows_input += len(df_raw)

                    df_norm = normalize_tlc(df_raw, min_date=min_date, max_date=max_date)
                    stats.rows_after_filter += len(df_norm)
                    if not df_norm.empty:
                        dfs_month.append(df_norm)

                if dfs_month:
                    df_month = pd.concat(dfs_month, ignore_index=True)
                    df_base_month = aggregate_tlc_month(df_month)
                    stats.rows_base_aggregated += len(df_base_month)

                    month_summary_rows.append(
                        (
                            f"{year}-{month:02d}",
                            len(files),
                            len(df_month),
                            len(df_base_month),
                        )
                    )

                    if not df_base_month.empty:
                        zone_ids_set.update(
                            pd.to_numeric(df_base_month["pu_location_id"], errors="coerce")
                            .dropna()
                            .astype(int)
                            .unique()
                            .tolist()
                        )
                        write_partitioned_dataset(
                            df_base_month,
                            tmp_base_dir,
                            partition_cols=["year", "month"],
                        )
                        month_keys_written.append((year, month))
                else:
                    month_summary_rows.append((f"{year}-{month:02d}", len(files), 0, 0))

        if not month_keys_written or not zone_ids_set:
            raise ValueError("Tras filtrar, no quedaron datos validos para demanda.")

        zone_ids = sorted(zone_ids_set)
        stats.unique_zones = len(zone_ids)

        # Pass 2: construir features/target por mes, con continuidad temporal por zona.
        history_tail = pd.DataFrame(columns=base_cols)
        pending_rows = pd.DataFrame(columns=FINAL_COLS)

        for mi, (year, month) in enumerate(sorted(month_keys_written), start=1):
            with console.status(
                f"[cyan]Pass 2/2 - Model-ready mensual {mi}/{len(month_keys_written)}: {year}-{month:02d}[/cyan]"
            ):
                df_base_month = _read_month_base(tmp_base_dir, year, month)
                df_month_grid = build_month_grid(
                    df_base_month=df_base_month,
                    zone_ids=zone_ids,
                    year=year,
                    month=month,
                    min_dt=min_dt,
                    max_dt=max_dt,
                )
                if df_month_grid.empty:
                    continue

                stats.rows_base_completed += len(df_month_grid)
                seen_ts.update(
                    pd.to_datetime(df_month_grid["timestamp_hour"], errors="coerce")
                    .dropna()
                    .unique()
                    .tolist()
                )

                df_month_grid["__is_current"] = 1
                if history_tail.empty:
                    hist = pd.DataFrame(columns=df_month_grid.columns)
                else:
                    hist = history_tail.copy()
                    hist["__is_current"] = 0

                augmented = pd.concat([hist, df_month_grid], ignore_index=True)
                augmented = augmented.sort_values(["pu_location_id", "timestamp_hour"]).reset_index(drop=True)

                df_aug_out = build_model_table(
                    df_base_full=augmented,
                    df_meteo=df_meteo,
                    df_events=df_events,
                    df_restaurants_zone_yearly=df_restaurants_zone,
                    df_rent_zone_yearly=df_rent_zone,
                    drop_na_lags=drop_na_lags,
                )

                bounds = _month_time_bounds(year, month, min_dt=min_dt, max_dt=max_dt)
                if bounds is None:
                    continue
                start, end = bounds
                month_mask = (
                    (pd.to_datetime(df_aug_out["timestamp_hour"], errors="coerce") >= start)
                    & (pd.to_datetime(df_aug_out["timestamp_hour"], errors="coerce") <= end)
                )
                df_month_out = df_aug_out.loc[month_mask].copy()
                if df_month_out.empty:
                    continue

                # Resolver target t+1 sin perder fronteras entre meses:
                # las filas sin futuro inmediato quedan en pending y se emiten
                # cuando se procese el siguiente mes.
                combined = pd.concat([pending_rows, df_month_out], ignore_index=True)
                combined = combined.sort_values(["pu_location_id", "timestamp_hour"]).reset_index(drop=True)
                combined["target_n_trips"] = (
                    combined.groupby("pu_location_id", sort=False)["num_trips"].shift(-1)
                )
                combined["target_n_trips"] = pd.to_numeric(
                    combined["target_n_trips"], errors="coerce"
                ).astype("Int32")

                emit = combined[combined["target_n_trips"].notna()].copy()
                pending_rows = combined[combined["target_n_trips"].isna()].copy()

                if not emit.empty:
                    write_partitioned_dataset(
                        emit,
                        out_base / output_dataset_name,
                        partition_cols=["year", "month"],
                    )
                    stats.rows_out += len(emit)

                hist_base = augmented[base_cols].copy()
                hist_base = hist_base.sort_values(["pu_location_id", "timestamp_hour"])
                history_tail = (
                    hist_base.groupby("pu_location_id", group_keys=False)
                    .tail(168)
                    .reset_index(drop=True)
                )

        stats.unique_timestamps = len(seen_ts)
    finally:
        if tmp_base_dir.exists():
            safe_remove_dir(tmp_base_dir)

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
