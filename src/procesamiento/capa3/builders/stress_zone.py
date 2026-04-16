from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from rich.table import Table

from src.procesamiento.capa3.common.constants import ALLOWED_MAX_DATE, ALLOWED_MIN_DATE, console
from src.procesamiento.capa3.common.externals import (
    load_event_features,
    load_meteo_features,
    load_rent_zone_features_yearly,
    load_restaurants_zone_features_yearly,
)
from src.procesamiento.capa3.common.io import (
    iter_month_partitions,
    list_all_parquets,
    safe_remove_dir,
    write_partitioned_dataset,
)

TARGET_HORIZONS = [1, 3, 24]
TARGET_STRESS_COLS = [f"target_stress_t{h}" for h in TARGET_HORIZONS]
TARGET_IS_STRESS_COLS = [f"target_is_stress_t{h}" for h in TARGET_HORIZONS]

NEEDED_TLC_COLS = [
    "date",
    "year",
    "month",
    "hour",
    "day_of_week",
    "is_weekend",
    "week_of_year",
    "pu_location_id",
    "total_amount_std",
    "is_valid_for_demand",
    "is_valid_for_price",
]

MODEL_COLS = [
    "timestamp_hour",
    "date",
    "year",
    "month",
    "hour",
    "hour_block_3h",
    "day_of_week",
    "is_weekend",
    "pu_location_id",
    "borough",
    "n_trips",
    "price_variability",
    "avg_price",
    "lag_1h_trips",
    "lag_24h_trips",
    "lag_168h_trips",
    "roll_3h_trips",
    "roll_24h_trips",
    "lag_1h_price_variability",
    "lag_24h_price_variability",
    "roll_3h_price_variability",
    "roll_24h_price_variability",
    "lag_1h_avg_price",
    "lag_24h_avg_price",
    "roll_24h_avg_price",
    "temp_c",
    "precip_mm",
    "city_n_events",
    "city_has_event",
    "n_restaurants_zone",
    "n_cuisines_zone",
    "rent_price_zone",
    "z_price_variability",
    "z_log1p_num_trips",
    "stress_score",
    "is_stress_now",
    *TARGET_STRESS_COLS,
    *TARGET_IS_STRESS_COLS,
]

MODEL_PRELABEL_COLS = [c for c in MODEL_COLS if c not in {"is_stress_now", *TARGET_IS_STRESS_COLS}]

PANEL_COLS = [
    "pu_location_id",
    "borough",
    "day_of_week",
    "hour",
    "hour_block_3h",
    "n_obs",
    "stress_rate",
    "stress_score_mean",
    "stress_score_p90",
    "n_trips_mean",
    "n_trips_p90",
    "price_variability_mean",
    "price_variability_p90",
    "avg_price_mean",
    "temp_c_mean",
    "precip_mm_mean",
    "city_n_events_mean",
    "city_has_event_rate",
]

HISTORY_HOURS = 168


@dataclass
class RunningMoments:
    n: int = 0
    s: float = 0.0
    ss: float = 0.0

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        self.n += int(x.size)
        self.s += float(np.sum(x))
        self.ss += float(np.sum(x * x))

    def mean_std(self) -> tuple[float, float]:
        if self.n <= 0:
            return 0.0, 1.0
        mean = self.s / self.n
        var = (self.ss / self.n) - (mean * mean)
        if var < 0:
            var = 0.0
        std = float(np.sqrt(var))
        if std <= 0:
            std = 1.0
        return float(mean), std


@dataclass
class BuildStats:
    months_detected: int = 0
    files_detected: int = 0
    rows_input: int = 0
    rows_after_filter: int = 0
    rows_base_aggregated: int = 0
    rows_base_completed: int = 0
    rows_model_out: int = 0
    rows_panel_out: int = 0
    unique_zones: int = 0
    unique_timestamps: int = 0
    meteo_rows: int = 0
    events_rows: int = 0
    restaurants_rows: int = 0
    rent_rows: int = 0
    threshold_now: float = 0.0
    threshold_target: float = 0.0
    threshold_target_t1: float = 0.0
    threshold_target_t3: float = 0.0
    threshold_target_t24: float = 0.0


def _zscore_from_params(s: pd.Series, mean: float, std: float) -> pd.Series:
    arr = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    denom = std if std > 0 else 1.0
    return ((arr - mean) / denom).astype("float32")


def _q90(s: pd.Series) -> float:
    if s.empty:
        return 0.0
    return float(pd.to_numeric(s, errors="coerce").dropna().quantile(0.90))


def _month_time_bounds(year: int, month: int, min_dt: pd.Timestamp, max_dt: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    month_start = pd.Timestamp(year=year, month=month, day=1)
    month_last_day = month_start + pd.offsets.MonthEnd(0)
    month_end = month_last_day + pd.Timedelta(hours=23)

    start = max(month_start, min_dt)
    end = min(month_end, max_dt)
    if start > end:
        return None
    return start, end


def _compute_temporal_split_bounds(
    timestamps: pd.Series | list[pd.Timestamp] | list[np.datetime64],
    train_frac: float,
    val_frac: float,
    gap_steps: int = 0,
) -> dict[str, pd.Timestamp | None]:
    """
    Calcula los límites temporales usando la misma lógica del split train/val/test.
    """
    if train_frac <= 0 or train_frac >= 1:
        raise ValueError("train_frac debe estar entre 0 y 1.")
    if val_frac < 0 or val_frac >= 1:
        raise ValueError("val_frac debe estar entre 0 y 1 (puede ser 0).")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac debe ser menor que 1.")
    if gap_steps < 0:
        raise ValueError("gap_steps no puede ser negativo.")

    ts = pd.to_datetime(pd.Series(list(timestamps)), errors="coerce")
    unique_ts = ts.dropna().drop_duplicates().sort_values().reset_index(drop=True)
    n_steps = len(unique_ts)
    if n_steps < 3:
        raise ValueError(
            "Muy pocos timestamps unicos para hacer split temporal fiable. "
            f"Timestamps disponibles: {n_steps}"
        )

    n_train = int(n_steps * train_frac)
    n_val = int(n_steps * val_frac)
    has_val = val_frac > 0
    gap_slots = gap_steps * (2 if has_val else 1)
    n_test = n_steps - n_train - n_val - gap_slots

    if n_train <= 0:
        raise ValueError("El bloque train queda vacio. Aumenta train_frac.")
    if has_val and n_val <= 0:
        raise ValueError(
            "val_frac > 0 pero el bloque de validacion queda vacio. "
            "Aumenta val_frac o desactivalo con val_frac=0."
        )
    if n_test <= 0:
        raise ValueError(
            "El bloque test queda vacio tras aplicar fracciones y gap_steps. "
            "Reduce gap_steps o ajusta train_frac/val_frac."
        )

    train_end_idx = n_train - 1
    train_end_ts = pd.Timestamp(unique_ts.iloc[train_end_idx])

    if has_val:
        val_start_idx = train_end_idx + 1 + gap_steps
        val_end_idx = val_start_idx + n_val - 1
        test_start_idx = val_end_idx + 1 + gap_steps
        return {
            "train_end": train_end_ts,
            "val_start": pd.Timestamp(unique_ts.iloc[val_start_idx]),
            "val_end": pd.Timestamp(unique_ts.iloc[val_end_idx]),
            "test_start": pd.Timestamp(unique_ts.iloc[test_start_idx]),
        }

    test_start_idx = train_end_idx + 1 + gap_steps
    return {
        "train_end": train_end_ts,
        "val_start": None,
        "val_end": None,
        "test_start": pd.Timestamp(unique_ts.iloc[test_start_idx]),
    }


def _read_month_base(tmp_base_dir: Path, year: int, month: int) -> pd.DataFrame:
    part_dir = tmp_base_dir / f"year={year}" / f"month={month}"
    files = sorted(part_dir.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(fp) for fp in files], ignore_index=True)


def _rolling_from_shifted(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    shift_n: int,
    window: int,
    min_periods: int,
) -> pd.Series:
    shifted = df.groupby(group_col, sort=False)[value_col].shift(shift_n)
    out = (
        shifted.groupby(df[group_col], sort=False)
        .rolling(window=window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return pd.to_numeric(out, errors="coerce")


def load_zone_lookup(project_root: Path, zone_lookup_path: str) -> pd.DataFrame:
    fp = (project_root / zone_lookup_path).resolve()
    if not fp.exists():
        console.print(f"[yellow]Aviso:[/yellow] No se encontro taxi_zone_lookup.csv en {fp}")
        return pd.DataFrame(columns=["pu_location_id", "borough"])

    zones = pd.read_csv(fp)
    if "LocationID" not in zones.columns or "Borough" not in zones.columns:
        console.print(
            f"[yellow]Aviso:[/yellow] taxi_zone_lookup.csv sin columnas esperadas en {fp}"
        )
        return pd.DataFrame(columns=["pu_location_id", "borough"])

    out = zones[["LocationID", "Borough"]].copy()
    out.columns = ["pu_location_id", "borough"]
    out["pu_location_id"] = pd.to_numeric(out["pu_location_id"], errors="coerce").astype("Int32")
    out["borough"] = out["borough"].astype("string").str.strip()
    out.loc[out["borough"].isin(["Unknown", "N/A", "NA", "nan"]), "borough"] = pd.NA
    out = out.dropna(subset=["pu_location_id"]).drop_duplicates(subset=["pu_location_id"])
    return out


def normalize_tlc(df: pd.DataFrame, min_date: str, max_date: str) -> pd.DataFrame:
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
    out["total_amount_std"] = pd.to_numeric(out["total_amount_std"], errors="coerce")
    out["is_valid_for_demand"] = pd.to_numeric(out["is_valid_for_demand"], errors="coerce").astype("Int8")
    out["is_valid_for_price"] = pd.to_numeric(out["is_valid_for_price"], errors="coerce").astype("Int8")

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
            "is_valid_for_price",
        ]
    )

    dmin = max(pd.to_datetime(min_date), ALLOWED_MIN_DATE)
    dmax = min(pd.to_datetime(max_date), ALLOWED_MAX_DATE)

    out = out[(out["date"] >= dmin) & (out["date"] <= dmax)]
    out = out[(out["hour"] >= 0) & (out["hour"] <= 23)]
    out = out[out["is_valid_for_demand"] == 1]
    return out


def aggregate_tlc_month(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    keys = [
        "date",
        "year",
        "month",
        "hour",
        "day_of_week",
        "is_weekend",
        "week_of_year",
        "pu_location_id",
    ]

    trips = df.groupby(keys, dropna=False).size().reset_index(name="n_trips")

    price_df = df[
        (pd.to_numeric(df["is_valid_for_price"], errors="coerce") == 1)
        & (pd.to_numeric(df["total_amount_std"], errors="coerce") > 0)
    ].copy()

    if not price_df.empty:
        avg_price = (
            price_df.groupby(keys, dropna=False)["total_amount_std"]
            .mean()
            .reset_index(name="avg_price")
        )
        q = (
            price_df.groupby(keys, dropna=False)["total_amount_std"]
            .quantile([0.25, 0.75])
            .unstack(level=-1)
            .reset_index()
            .rename(columns={0.25: "p25", 0.75: "p75"})
        )
        q["price_variability"] = (q["p75"] - q["p25"]).clip(lower=0.0)
        q = q[keys + ["price_variability"]]

        out = trips.merge(avg_price, on=keys, how="left")
        out = out.merge(q, on=keys, how="left")
    else:
        out = trips.copy()
        out["avg_price"] = np.nan
        out["price_variability"] = np.nan

    out["avg_price"] = pd.to_numeric(out["avg_price"], errors="coerce").fillna(0.0).astype("float32")
    out["price_variability"] = pd.to_numeric(out["price_variability"], errors="coerce").fillna(0.0).astype("float32")
    out["n_trips"] = pd.to_numeric(out["n_trips"], errors="coerce").fillna(0).astype("Int32")

    out["timestamp_hour"] = pd.to_datetime(out["date"], errors="coerce") + pd.to_timedelta(
        pd.to_numeric(out["hour"], errors="coerce"), unit="h"
    )

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
            "n_trips",
            "avg_price",
            "price_variability",
        ]
    ].copy()


def _aggregate_tlc_month_duckdb(
    files: list[Path],
    min_date: str,
    max_date: str,
) -> tuple[pd.DataFrame, int, int]:
    import duckdb

    if not files:
        return pd.DataFrame(), 0, 0

    dmin = max(pd.to_datetime(min_date), ALLOWED_MIN_DATE).date()
    dmax = min(pd.to_datetime(max_date), ALLOWED_MAX_DATE).date()
    file_list = [str(fp) for fp in files]

    cte_filtered = """
    WITH raw AS (
      SELECT
        TRY_CAST(date AS DATE) AS date,
        TRY_CAST(year AS INTEGER) AS year,
        TRY_CAST(month AS INTEGER) AS month,
        TRY_CAST(hour AS INTEGER) AS hour,
        TRY_CAST(day_of_week AS INTEGER) AS day_of_week,
        TRY_CAST(is_weekend AS INTEGER) AS is_weekend,
        TRY_CAST(week_of_year AS INTEGER) AS week_of_year,
        TRY_CAST(pu_location_id AS INTEGER) AS pu_location_id,
        TRY_CAST(total_amount_std AS DOUBLE) AS total_amount_std,
        TRY_CAST(is_valid_for_demand AS INTEGER) AS is_valid_for_demand,
        TRY_CAST(is_valid_for_price AS INTEGER) AS is_valid_for_price
      FROM read_parquet(?)
    ),
    filtered AS (
      SELECT *
      FROM raw
      WHERE
        date IS NOT NULL
        AND year IS NOT NULL
        AND month IS NOT NULL
        AND hour IS NOT NULL
        AND day_of_week IS NOT NULL
        AND is_weekend IS NOT NULL
        AND week_of_year IS NOT NULL
        AND pu_location_id IS NOT NULL
        AND is_valid_for_demand IS NOT NULL
        AND is_valid_for_price IS NOT NULL
        AND date >= ?
        AND date <= ?
        AND hour BETWEEN 0 AND 23
        AND is_valid_for_demand = 1
    )
    """

    q_counts = (
        cte_filtered
        + """
    SELECT
      (SELECT COUNT(*) FROM raw) AS rows_input,
      (SELECT COUNT(*) FROM filtered) AS rows_after_filter
    """
    )

    q_agg = (
        cte_filtered
        + """
    SELECT
      date,
      year,
      month,
      hour,
      day_of_week,
      is_weekend,
      week_of_year,
      pu_location_id,
      CAST(COUNT(*) AS INTEGER) AS n_trips,
      CAST(
        COALESCE(
          AVG(CASE WHEN is_valid_for_price = 1 AND total_amount_std > 0 THEN total_amount_std END),
          0.0
        ) AS DOUBLE
      ) AS avg_price,
      CAST(
        COALESCE(
          quantile_cont(CASE WHEN is_valid_for_price = 1 AND total_amount_std > 0 THEN total_amount_std END, 0.75)
          - quantile_cont(CASE WHEN is_valid_for_price = 1 AND total_amount_std > 0 THEN total_amount_std END, 0.25),
          0.0
        ) AS DOUBLE
      ) AS price_variability
    FROM filtered
    GROUP BY
      date, year, month, hour, day_of_week, is_weekend, week_of_year, pu_location_id
    """
    )

    con = duckdb.connect(database=":memory:")
    try:
        rows_input, rows_after_filter = con.execute(q_counts, [file_list, dmin, dmax]).fetchone()
        out = con.execute(q_agg, [file_list, dmin, dmax]).fetchdf()
    finally:
        con.close()

    if out.empty:
        return pd.DataFrame(), int(rows_input or 0), int(rows_after_filter or 0)

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int32")
    out["month"] = pd.to_numeric(out["month"], errors="coerce").astype("Int32")
    out["hour"] = pd.to_numeric(out["hour"], errors="coerce").astype("Int32")
    out["day_of_week"] = pd.to_numeric(out["day_of_week"], errors="coerce").astype("Int32")
    out["is_weekend"] = pd.to_numeric(out["is_weekend"], errors="coerce").astype("Int8")
    out["week_of_year"] = pd.to_numeric(out["week_of_year"], errors="coerce").astype("Int32")
    out["pu_location_id"] = pd.to_numeric(out["pu_location_id"], errors="coerce").astype("Int32")
    out["n_trips"] = pd.to_numeric(out["n_trips"], errors="coerce").fillna(0).astype("Int32")
    out["avg_price"] = pd.to_numeric(out["avg_price"], errors="coerce").fillna(0.0).astype("float32")
    out["price_variability"] = pd.to_numeric(out["price_variability"], errors="coerce").fillna(0.0).clip(lower=0.0).astype("float32")

    out["timestamp_hour"] = pd.to_datetime(out["date"], errors="coerce") + pd.to_timedelta(
        pd.to_numeric(out["hour"], errors="coerce"), unit="h"
    )

    out = out[
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
            "n_trips",
            "avg_price",
            "price_variability",
        ]
    ].copy()

    return out, int(rows_input or 0), int(rows_after_filter or 0)


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
        base_idx = pd.DataFrame(columns=["timestamp_hour", "pu_location_id", "n_trips", "avg_price", "price_variability"])
        base_idx = base_idx.set_index(["timestamp_hour", "pu_location_id"])
    else:
        base_idx = df_base_month.set_index(["timestamp_hour", "pu_location_id"])[["n_trips", "avg_price", "price_variability"]]

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
    out["n_trips"] = pd.to_numeric(out["n_trips"], errors="coerce").fillna(0).astype("Int32")
    out["avg_price"] = pd.to_numeric(out["avg_price"], errors="coerce").fillna(0.0).astype("float32")
    out["price_variability"] = pd.to_numeric(out["price_variability"], errors="coerce").fillna(0.0).astype("float32")

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
            "n_trips",
            "avg_price",
            "price_variability",
        ]
    ].copy()


def _build_panel_with_duckdb(model_dir: Path) -> pd.DataFrame:
    import duckdb

    glob_path = str((model_dir / "**" / "*.parquet").resolve())

    q = f"""
    SELECT
      pu_location_id,
      borough,
      day_of_week,
      hour,
      hour_block_3h,
      COUNT(*) AS n_obs,
      AVG(CAST(is_stress_now AS DOUBLE)) AS stress_rate,
      AVG(stress_score) AS stress_score_mean,
      quantile_cont(stress_score, 0.9) AS stress_score_p90,
      AVG(n_trips) AS n_trips_mean,
      quantile_cont(n_trips, 0.9) AS n_trips_p90,
      AVG(price_variability) AS price_variability_mean,
      quantile_cont(price_variability, 0.9) AS price_variability_p90,
      AVG(avg_price) AS avg_price_mean,
      AVG(temp_c) AS temp_c_mean,
      AVG(precip_mm) AS precip_mm_mean,
      AVG(city_n_events) AS city_n_events_mean,
      AVG(CAST(city_has_event AS DOUBLE)) AS city_has_event_rate
    FROM read_parquet('{glob_path}')
    GROUP BY 1,2,3,4,5
    """

    con = duckdb.connect(database=":memory:")
    panel = con.execute(q).fetchdf()
    con.close()
    return panel


def _build_panel_with_pandas(model_dir: Path) -> pd.DataFrame:
    files = list_all_parquets(model_dir)
    if not files:
        return pd.DataFrame(columns=PANEL_COLS)
    df = pd.concat([pd.read_parquet(fp) for fp in files], ignore_index=True)

    grp_cols = ["pu_location_id", "borough", "day_of_week", "hour", "hour_block_3h"]
    panel = (
        df.groupby(grp_cols, dropna=False, as_index=False)
        .agg(
            n_obs=("timestamp_hour", "size"),
            stress_rate=("is_stress_now", "mean"),
            stress_score_mean=("stress_score", "mean"),
            stress_score_p90=("stress_score", lambda s: _q90(pd.to_numeric(s, errors="coerce"))),
            n_trips_mean=("n_trips", "mean"),
            n_trips_p90=("n_trips", lambda s: _q90(pd.to_numeric(s, errors="coerce"))),
            price_variability_mean=("price_variability", "mean"),
            price_variability_p90=("price_variability", lambda s: _q90(pd.to_numeric(s, errors="coerce"))),
            avg_price_mean=("avg_price", "mean"),
            temp_c_mean=("temp_c", "mean"),
            precip_mm_mean=("precip_mm", "mean"),
            city_n_events_mean=("city_n_events", "mean"),
            city_has_event_rate=("city_has_event", "mean"),
        )
    )
    return panel


def build_stress_zone_dataset(
    layer2_path: Path,
    out_base: Path,
    meteo_base: Path,
    events_base: Path,
    restaurants_base: Path,
    rent_base: Path,
    min_date: str,
    max_date: str,
    drop_na_history: bool = True,
    drop_na_targets: bool = True,
    stress_quantile: float = 0.90,
    zone_lookup_path: str = "data/external/taxi_zone_lookup.csv",
    output_model_dataset_name: str = "df_stress_zone_hour_day",
    output_panel_dataset_name: str = "df_stress_zone_slot",
) -> BuildStats:
    stats = BuildStats()
    project_root = Path(__file__).resolve().parents[4]

    month_parts = list(iter_month_partitions(layer2_path))
    if not month_parts:
        raise FileNotFoundError(
            f"No hay parquets dentro de {layer2_path} con estructura year=YYYY/month=MM"
        )

    stats.months_detected = len(month_parts)
    stats.files_detected = sum(len(files) for _, _, files in month_parts)

    info = Table(show_header=True, header_style="bold white", title="Entrada STRESS ZONE TLC")
    info.add_column("Metrica", style="bold cyan")
    info.add_column("Valor", justify="right")
    info.add_row("Meses detectados", f"{stats.months_detected:,}")
    info.add_row("Parquets detectados", f"{stats.files_detected:,}")
    console.print(info)

    min_dt = max(pd.to_datetime(min_date), ALLOWED_MIN_DATE)
    max_dt = min(pd.to_datetime(max_date), ALLOWED_MAX_DATE) + pd.Timedelta(hours=23)

    zones = load_zone_lookup(project_root=project_root, zone_lookup_path=zone_lookup_path)
    if not zones.empty:
        zone_ids = sorted(pd.to_numeric(zones["pu_location_id"], errors="coerce").dropna().astype(int).unique().tolist())
    else:
        zone_ids = []
    stats.unique_zones = len(zone_ids)

    df_meteo = load_meteo_features(meteo_base, min_date=min_date, max_date=max_date)
    df_events = load_event_features(events_base, min_date=min_date, max_date=max_date)
    df_restaurants_zone = load_restaurants_zone_features_yearly(
        restaurants_base=restaurants_base,
        min_date=min_date,
        max_date=max_date,
    )
    df_rent_zone = load_rent_zone_features_yearly(
        rent_base=rent_base,
        min_date=min_date,
        max_date=max_date,
    )
    stats.meteo_rows = len(df_meteo)
    stats.events_rows = len(df_events)
    stats.restaurants_rows = len(df_restaurants_zone)
    stats.rent_rows = len(df_rent_zone)

    model_out_dir = out_base / output_model_dataset_name
    panel_out_dir = out_base / output_panel_dataset_name
    tmp_base_dir = out_base / "_tmp_stress_base"
    tmp_model_dir = out_base / "_tmp_stress_model_nobin"

    for d in [tmp_base_dir, tmp_model_dir]:
        if d.exists():
            safe_remove_dir(d)
        d.mkdir(parents=True, exist_ok=True)

    month_summary_rows: List[Tuple[str, int, int, int]] = []
    month_keys_written: list[tuple[int, int]] = []
    seen_ts: set[pd.Timestamp] = set()
    # Configuración anti-leakage fija y alineada con el split del modelo.
    split_train_frac = 0.70
    split_val_frac = 0.15
    split_gap_steps = 0

    try:
        # Pass 1: construir base mensual en disco.
        for mi, (year, month, files) in enumerate(month_parts, start=1):
            if _month_time_bounds(year, month, min_dt=min_dt, max_dt=max_dt) is None:
                month_summary_rows.append((f"{year}-{month:02d}", len(files), 0, 0))
                continue

            with console.status(
                f"[cyan]Pass 1/3 - Base mensual {mi}/{stats.months_detected}: {year}-{month:02d}[/cyan]"
            ):
                # Ruta principal: DuckDB hace agregacion mensual directamente desde parquet,
                # reduciendo el pico de RAM en meses grandes.
                try:
                    df_base_month, month_rows_input, month_rows_after = _aggregate_tlc_month_duckdb(
                        files=files,
                        min_date=min_date,
                        max_date=max_date,
                    )
                    stats.rows_input += month_rows_input
                    stats.rows_after_filter += month_rows_after
                    stats.rows_base_aggregated += len(df_base_month)
                except Exception as exc:
                    console.print(
                        "[yellow]Aviso:[/yellow] DuckDB no disponible/fallo en agregacion mensual "
                        f"({year}-{month:02d}). Se usa fallback pandas. Detalle: {exc}"
                    )
                    dfs_month: List[pd.DataFrame] = []
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
                    else:
                        df_base_month = pd.DataFrame()

                if not zone_ids and not df_base_month.empty:
                    zone_ids = sorted(
                        pd.to_numeric(df_base_month["pu_location_id"], errors="coerce")
                        .dropna()
                        .astype(int)
                        .unique()
                        .tolist()
                    )
                    stats.unique_zones = len(zone_ids)

                df_month_grid = build_month_grid(
                    df_base_month=df_base_month,
                    zone_ids=zone_ids,
                    year=year,
                    month=month,
                    min_dt=min_dt,
                    max_dt=max_dt,
                )

                month_summary_rows.append(
                    (
                        f"{year}-{month:02d}",
                        len(files),
                        int(len(df_base_month)),
                        int(len(df_month_grid)),
                    )
                )

                if df_month_grid.empty:
                    continue

                stats.rows_base_completed += len(df_month_grid)
                seen_ts.update(pd.to_datetime(df_month_grid["timestamp_hour"], errors="coerce").dropna().unique().tolist())

                write_partitioned_dataset(
                    df_month_grid,
                    tmp_base_dir,
                    partition_cols=["year", "month"],
                )
                month_keys_written.append((year, month))

        stats.unique_timestamps = len(seen_ts)
        if not seen_ts:
            raise ValueError("No hay timestamps en la base completada para construir el dataset de stress.")

        split_bounds = _compute_temporal_split_bounds(
            timestamps=list(seen_ts),
            train_frac=split_train_frac,
            val_frac=split_val_frac,
            gap_steps=split_gap_steps,
        )
        train_end_ts = split_bounds["train_end"]
        if train_end_ts is None:
            raise ValueError("No se pudo calcular train_end para el ajuste sin leakage.")

        console.print(
            "[cyan]Ajuste anti-leakage[/cyan] "
            f"| train_end={train_end_ts} | val_start={split_bounds['val_start']} | "
            f"val_end={split_bounds['val_end']} | test_start={split_bounds['test_start']}"
        )

        # Recalcular momentos SOLO con filas de train para evitar leakage temporal.
        moment_var = RunningMoments()
        moment_log = RunningMoments()
        for year, month in sorted(month_keys_written):
            df_month_grid = _read_month_base(tmp_base_dir, year, month)
            if df_month_grid.empty:
                continue
            ts = pd.to_datetime(df_month_grid["timestamp_hour"], errors="coerce")
            train_mask = ts <= train_end_ts
            if not bool(train_mask.any()):
                continue

            train_slice = df_month_grid.loc[train_mask]
            arr_var = pd.to_numeric(train_slice["price_variability"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            arr_log = np.log1p(pd.to_numeric(train_slice["n_trips"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
            moment_var.update(arr_var)
            moment_log.update(arr_log)

        if moment_var.n <= 0 or moment_log.n <= 0:
            raise ValueError(
                "No hay filas de train suficientes para calcular z-score sin leakage. "
                "Revisa el rango de fechas."
            )

        mean_var, std_var = moment_var.mean_std()
        mean_log, std_log = moment_log.mean_std()

        # Pass 2: features + stress + targets multi-horizonte (sin etiquetas binarias)
        history_tail = pd.DataFrame(
            columns=["timestamp_hour", "pu_location_id", "n_trips", "avg_price", "price_variability"]
        )
        pending_rows = pd.DataFrame(columns=MODEL_PRELABEL_COLS)

        all_stress_parts: list[np.ndarray] = []
        all_target_parts: dict[int, list[np.ndarray]] = {h: [] for h in TARGET_HORIZONS}

        for mi, (year, month) in enumerate(sorted(month_keys_written), start=1):
            with console.status(
                f"[cyan]Pass 2/3 - Features/target {mi}/{len(month_keys_written)}: {year}-{month:02d}[/cyan]"
            ):
                df_month_grid = _read_month_base(tmp_base_dir, year, month)
                if df_month_grid.empty:
                    continue

                df_month_grid["__is_current"] = 1
                if history_tail.empty:
                    hist = pd.DataFrame(columns=df_month_grid.columns)
                else:
                    hist = history_tail.copy()
                    hist["__is_current"] = 0

                augmented = pd.concat([hist, df_month_grid], ignore_index=True)
                augmented = augmented.sort_values(["pu_location_id", "timestamp_hour"]).reset_index(drop=True)

                grp = augmented.groupby("pu_location_id", sort=False)
                augmented["lag_1h_trips"] = grp["n_trips"].shift(1)
                augmented["lag_24h_trips"] = grp["n_trips"].shift(24)
                augmented["lag_168h_trips"] = grp["n_trips"].shift(168)
                augmented["roll_3h_trips"] = _rolling_from_shifted(
                    augmented, "pu_location_id", "n_trips", shift_n=1, window=3, min_periods=3
                )
                augmented["roll_24h_trips"] = _rolling_from_shifted(
                    augmented, "pu_location_id", "n_trips", shift_n=1, window=24, min_periods=24
                )

                augmented["lag_1h_price_variability"] = grp["price_variability"].shift(1)
                augmented["lag_24h_price_variability"] = grp["price_variability"].shift(24)
                augmented["roll_3h_price_variability"] = _rolling_from_shifted(
                    augmented, "pu_location_id", "price_variability", shift_n=1, window=3, min_periods=3
                )
                augmented["roll_24h_price_variability"] = _rolling_from_shifted(
                    augmented, "pu_location_id", "price_variability", shift_n=1, window=24, min_periods=24
                )

                augmented["lag_1h_avg_price"] = grp["avg_price"].shift(1)
                augmented["lag_24h_avg_price"] = grp["avg_price"].shift(24)
                augmented["roll_24h_avg_price"] = _rolling_from_shifted(
                    augmented, "pu_location_id", "avg_price", shift_n=1, window=24, min_periods=24
                )

                current = augmented[augmented["__is_current"] == 1].copy()
                current = current.drop(columns=["__is_current"])
                current["date"] = pd.to_datetime(current["date"], errors="coerce").dt.floor("D")
                current["hour"] = pd.to_numeric(current["hour"], errors="coerce").astype("Int32")

                if df_meteo is not None and not df_meteo.empty:
                    current = current.merge(df_meteo, on=["date", "hour"], how="left")
                else:
                    current["temp_c"] = pd.NA
                    current["precip_mm"] = pd.NA

                if df_events is not None and not df_events.empty:
                    current = current.merge(df_events, on=["date", "hour"], how="left")
                else:
                    current["city_n_events"] = 0.0
                    current["city_has_event"] = 0

                if not zones.empty:
                    current = current.merge(zones, on="pu_location_id", how="left")
                else:
                    current["borough"] = pd.NA

                if df_restaurants_zone is not None and not df_restaurants_zone.empty:
                    current = current.merge(df_restaurants_zone, on=["year", "pu_location_id"], how="left")
                else:
                    current["n_restaurants_zone"] = pd.NA
                    current["n_cuisines_zone"] = pd.NA

                if df_rent_zone is not None and not df_rent_zone.empty:
                    current = current.merge(df_rent_zone, on=["year", "pu_location_id"], how="left")
                else:
                    current["rent_price_zone"] = pd.NA

                current["hour_block_3h"] = (pd.to_numeric(current["hour"], errors="coerce") // 3).astype("Int8")

                if "city_n_events" not in current.columns:
                    current["city_n_events"] = 0.0
                current["city_n_events"] = pd.to_numeric(current["city_n_events"], errors="coerce").fillna(0.0)

                if "city_has_event" in current.columns:
                    city_has_event = pd.to_numeric(current["city_has_event"], errors="coerce")
                    city_has_event = city_has_event.fillna((current["city_n_events"] > 0).astype("Int8"))
                    current["city_has_event"] = city_has_event
                else:
                    current["city_has_event"] = (current["city_n_events"] > 0).astype("Int8")

                for c in ["n_restaurants_zone", "n_cuisines_zone"]:
                    if c not in current.columns:
                        current[c] = 0.0
                    current[c] = pd.to_numeric(current[c], errors="coerce").fillna(0.0)

                if "rent_price_zone" not in current.columns:
                    current["rent_price_zone"] = pd.NA
                current["rent_price_zone"] = pd.to_numeric(current["rent_price_zone"], errors="coerce")

                current["z_log1p_num_trips"] = _zscore_from_params(
                    np.log1p(pd.to_numeric(current["n_trips"], errors="coerce").fillna(0.0)),
                    mean=mean_log,
                    std=std_log,
                )
                current["z_price_variability"] = _zscore_from_params(
                    pd.to_numeric(current["price_variability"], errors="coerce").fillna(0.0),
                    mean=mean_var,
                    std=std_var,
                )
                current["stress_score"] = (
                    pd.to_numeric(current["z_price_variability"], errors="coerce").fillna(0.0)
                    + pd.to_numeric(current["z_log1p_num_trips"], errors="coerce").fillna(0.0)
                ).astype("float32")

                for c in MODEL_PRELABEL_COLS:
                    if c not in current.columns:
                        current[c] = pd.NA
                current = current[MODEL_PRELABEL_COLS].copy()

                combined = pd.concat([pending_rows, current], ignore_index=True)
                combined = combined.sort_values(["pu_location_id", "timestamp_hour"]).reset_index(drop=True)
                grp_combined = combined.groupby("pu_location_id", sort=False)
                for h in TARGET_HORIZONS:
                    col = f"target_stress_t{h}"
                    combined[col] = grp_combined["stress_score"].shift(-h).astype("float32")

                max_h = max(TARGET_HORIZONS)
                max_col = f"target_stress_t{max_h}"
                emit = combined[combined[max_col].notna()].copy()
                pending_rows = combined[combined[max_col].isna()].copy()

                if drop_na_history:
                    needed_history = [
                        "lag_1h_trips",
                        "lag_24h_trips",
                        "lag_168h_trips",
                        "roll_3h_trips",
                        "roll_24h_trips",
                        "lag_1h_price_variability",
                        "lag_24h_price_variability",
                        "roll_3h_price_variability",
                        "roll_24h_price_variability",
                        "lag_1h_avg_price",
                        "lag_24h_avg_price",
                        "roll_24h_avg_price",
                    ]
                    emit = emit.dropna(subset=[c for c in needed_history if c in emit.columns]).copy()

                if drop_na_targets:
                    emit = emit.dropna(subset=TARGET_STRESS_COLS).copy()

                if not emit.empty:
                    # Umbrales de clasificación sin leakage:
                    # se ajustan solo con filas de train (según timestamp de la fila).
                    emit_ts = pd.to_datetime(emit["timestamp_hour"], errors="coerce")
                    train_emit = emit[emit_ts <= train_end_ts].copy()
                    if not train_emit.empty:
                        all_stress_parts.append(
                            pd.to_numeric(train_emit["stress_score"], errors="coerce").dropna().to_numpy(dtype=float)
                        )

                        for h in TARGET_HORIZONS:
                            target_col = f"target_stress_t{h}"
                            # target_stress_tH es t+H: solo se usa si el valor objetivo cae en train.
                            train_target_mask = (
                                emit_ts.loc[train_emit.index] + pd.Timedelta(hours=h)
                            ) <= train_end_ts
                            train_emit_target = train_emit.loc[train_target_mask]
                            if not train_emit_target.empty:
                                all_target_parts[h].append(
                                    pd.to_numeric(train_emit_target[target_col], errors="coerce")
                                    .dropna()
                                    .to_numpy(dtype=float)
                                )

                    write_partitioned_dataset(
                        emit,
                        tmp_model_dir,
                        partition_cols=["year", "month"],
                    )

                hist_base = augmented[
                    ["timestamp_hour", "pu_location_id", "n_trips", "avg_price", "price_variability"]
                ].copy()
                hist_base = hist_base.sort_values(["pu_location_id", "timestamp_hour"])
                history_tail = hist_base.groupby("pu_location_id", group_keys=False).tail(HISTORY_HOURS).reset_index(
                    drop=True
                )

        if all_stress_parts:
            stress_all = np.concatenate(all_stress_parts)
            threshold_now = float(np.quantile(stress_all, stress_quantile))
        else:
            threshold_now = 0.0

        target_thresholds: dict[int, float] = {}
        for h in TARGET_HORIZONS:
            if all_target_parts[h]:
                target_all = np.concatenate(all_target_parts[h])
                target_thresholds[h] = float(np.quantile(target_all, stress_quantile))
            else:
                target_thresholds[h] = 0.0

        stats.threshold_now = threshold_now
        stats.threshold_target = float(target_thresholds[1])
        stats.threshold_target_t1 = float(target_thresholds[1])
        stats.threshold_target_t3 = float(target_thresholds[3])
        stats.threshold_target_t24 = float(target_thresholds[24])

        # Pass 3: etiquetado binario final + salida model-ready en disco
        model_files = list_all_parquets(tmp_model_dir)
        for i, fp in enumerate(model_files, start=1):
            with console.status(f"[cyan]Pass 3/3 - Etiquetado final {i}/{len(model_files)}[/cyan]"):
                df = pd.read_parquet(fp)
                if df.empty:
                    continue

                df["is_stress_now"] = (
                    pd.to_numeric(df["stress_score"], errors="coerce").fillna(0.0) >= threshold_now
                ).astype("Int8")
                for h in TARGET_HORIZONS:
                    target_col = f"target_stress_t{h}"
                    target_is_col = f"target_is_stress_t{h}"
                    df[target_is_col] = (
                        pd.to_numeric(df[target_col], errors="coerce").fillna(-np.inf) >= target_thresholds[h]
                    ).astype("Int8")

                float_cols = [
                    "avg_price",
                    "price_variability",
                    "lag_1h_trips",
                    "lag_24h_trips",
                    "lag_168h_trips",
                    "roll_3h_trips",
                    "roll_24h_trips",
                    "lag_1h_price_variability",
                    "lag_24h_price_variability",
                    "roll_3h_price_variability",
                    "roll_24h_price_variability",
                    "lag_1h_avg_price",
                    "lag_24h_avg_price",
                    "roll_24h_avg_price",
                    "temp_c",
                    "precip_mm",
                    "city_n_events",
                    "n_restaurants_zone",
                    "n_cuisines_zone",
                    "rent_price_zone",
                    "z_price_variability",
                    "z_log1p_num_trips",
                    "stress_score",
                    *TARGET_STRESS_COLS,
                ]
                for c in float_cols:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

                int_cols = ["year", "month", "hour", "day_of_week", "pu_location_id", "n_trips"]
                for c in int_cols:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int32")

                int8_cols = [
                    "is_weekend",
                    "hour_block_3h",
                    "city_has_event",
                    "is_stress_now",
                    *TARGET_IS_STRESS_COLS,
                ]
                for c in int8_cols:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int8")

                if "borough" in df.columns:
                    df["borough"] = df["borough"].astype("string")

                for c in MODEL_COLS:
                    if c not in df.columns:
                        df[c] = pd.NA
                df = df[MODEL_COLS].copy()

                write_partitioned_dataset(
                    df,
                    model_out_dir,
                    partition_cols=["year", "month"],
                )
                stats.rows_model_out += len(df)

        # Panel agregado a partir del modelo final
        panel_df: pd.DataFrame
        try:
            panel_df = _build_panel_with_duckdb(model_out_dir)
        except Exception:
            panel_df = _build_panel_with_pandas(model_out_dir)

        if not panel_df.empty:
            panel_df["n_obs"] = pd.to_numeric(panel_df["n_obs"], errors="coerce").fillna(0).astype("Int32")
            for c in [
                "stress_rate",
                "stress_score_mean",
                "stress_score_p90",
                "n_trips_mean",
                "n_trips_p90",
                "price_variability_mean",
                "price_variability_p90",
                "avg_price_mean",
                "temp_c_mean",
                "precip_mm_mean",
                "city_n_events_mean",
                "city_has_event_rate",
            ]:
                if c in panel_df.columns:
                    panel_df[c] = pd.to_numeric(panel_df[c], errors="coerce").astype("float32")

            panel_df["pu_location_id"] = pd.to_numeric(panel_df["pu_location_id"], errors="coerce").astype("Int32")
            panel_df["day_of_week"] = pd.to_numeric(panel_df["day_of_week"], errors="coerce").astype("Int32")
            panel_df["hour"] = pd.to_numeric(panel_df["hour"], errors="coerce").astype("Int32")
            panel_df["hour_block_3h"] = pd.to_numeric(panel_df["hour_block_3h"], errors="coerce").astype("Int8")
            panel_df["borough"] = panel_df["borough"].astype("string")

            for c in PANEL_COLS:
                if c not in panel_df.columns:
                    panel_df[c] = pd.NA
            panel_df = panel_df[PANEL_COLS].copy()

            write_partitioned_dataset(
                panel_df,
                panel_out_dir,
                partition_cols=["day_of_week"],
            )
            stats.rows_panel_out = len(panel_df)

        month_summary = Table(show_header=True, header_style="bold magenta", title="Resumen mensual STRESS ZONE")
        month_summary.add_column("Mes", style="bold white")
        month_summary.add_column("Parquets", justify="right")
        month_summary.add_column("Filas base agg", justify="right")
        month_summary.add_column("Filas grid mes", justify="right")
        for mes, n_files, n_base, n_grid in month_summary_rows:
            month_summary.add_row(mes, f"{n_files:,}", f"{n_base:,}", f"{n_grid:,}")
        console.print(month_summary)

        outputs = Table(show_header=True, header_style="bold magenta", title="Salida STRESS ZONE")
        outputs.add_column("Dataset", style="bold white")
        outputs.add_column("Rows", justify="right")
        outputs.add_column("Path")
        outputs.add_row(output_model_dataset_name, f"{stats.rows_model_out:,}", str(model_out_dir))
        outputs.add_row(output_panel_dataset_name, f"{stats.rows_panel_out:,}", str(panel_out_dir))
        outputs.add_row("stress_threshold_now", f"{stats.threshold_now:.4f}", "-")
        outputs.add_row("target_threshold_t1", f"{stats.threshold_target_t1:.4f}", "-")
        outputs.add_row("target_threshold_t3", f"{stats.threshold_target_t3:.4f}", "-")
        outputs.add_row("target_threshold_t24", f"{stats.threshold_target_t24:.4f}", "-")
        outputs.add_row("anti_leakage_train_end", str(train_end_ts), "-")
        outputs.add_row("anti_leakage_val_start", str(split_bounds["val_start"]), "-")
        outputs.add_row("anti_leakage_test_start", str(split_bounds["test_start"]), "-")
        console.print(outputs)

        return stats

    finally:
        for d in [tmp_base_dir, tmp_model_dir]:
            if d.exists():
                safe_remove_dir(d)
