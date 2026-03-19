from __future__ import annotations

import argparse
import json
import re
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    from config.settings import obtener_ruta  # type: ignore
except Exception:
    def obtener_ruta(p: str) -> Path:
        return Path(p)


console = Console()

DEBUG = False
MIN_YEAR = 2023
MAX_YEAR = 2025
SERVICES = ("yellow", "green", "fhvhv")

# ---------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------
MIN_GROUP_SIZE = 30
BATCH_SIZE_DEFAULT = 500_000

# variables sometidas a outlier contextual
OUTLIER_VARS = [
    "trip_duration_min",
    "trip_distance",
    "total_amount_std",
    "price_per_mile",
]

# guardarraíles globales previos al detector contextual
MAX_SPEED_MPH_HARD = 120.0
MAX_TIP_PCT = 3.0
MAX_ABS_NEG_AMOUNT = -5.0

# columnas necesarias para leer
NEEDED_COLS = [
    "tpep_pickup_datetime",
    "lpep_pickup_datetime",
    "pickup_datetime",
    "tpep_dropoff_datetime",
    "lpep_dropoff_datetime",
    "dropoff_datetime",
    "PULocationID",
    "pu_location_id",
    "DOLocationID",
    "do_location_id",
    "total_amount",
    "fare_amount",
    "tip_amount",
    "tips",
    "tolls_amount",
    "tolls",
    "airport_fee",
    "Airport_fee",
    "congestion_surcharge",
    "base_passenger_fare",
    "trip_distance",
    "trip_miles",
    "VendorID",
    "passenger_count",
    "RatecodeID",
    "payment_type",
]

# definición conservadora de duplicado exacto
DEDUP_COLS = [
    "service_type",
    "pickup_datetime",
    "dropoff_datetime",
    "pu_location_id",
    "do_location_id",
    "trip_distance",
    "total_amount_std",
    "fare_amount",
    "tip_amount",
    "VendorID",
    "passenger_count",
    "RatecodeID",
    "payment_type",
]

# columnas finales persistidas
FINAL_COLS = [
    "service_type",
    "pickup_datetime",
    "dropoff_datetime",
    "date",
    "year",
    "month",
    "hour",
    "day_of_week",
    "is_weekend",
    "week_of_year",
    "pu_location_id",
    "do_location_id",
    "trip_distance",
    "trip_duration_min",
    "total_amount_std",
    "total_amount",
    "fare_amount",
    "tip_amount",
    "tips",
    "tip_pct",
    "price_per_mile",
    "tolls_amount",
    "tolls",
    "congestion_surcharge",
    "airport_fee",
    "base_passenger_fare",
    "trip_miles",
    "VendorID",
    "passenger_count",
    "RatecodeID",
    "payment_type",
    "is_valid_for_demand",
    "is_valid_for_price",
    "is_valid_for_duration",
    "is_valid_for_distance",
    "is_valid_for_tip",
]


@dataclass
class FileStats:
    file_name: str = ""
    service: str = ""
    file_year: Optional[int] = None
    file_month: Optional[int] = None

    rows_in: int = 0
    rows_after_standardize: int = 0
    rows_removed_missing_core: int = 0
    rows_removed_wrong_month: int = 0
    duplicated_exact_found: int = 0
    rows_removed_duplicates_exact: int = 0
    rows_out: int = 0

    invalidated: Dict[str, int] = field(default_factory=dict)
    winsorized: Dict[str, int] = field(default_factory=dict)

    final_valid_flags_sum: Dict[str, int] = field(default_factory=dict)
    final_non_null_sum: Dict[str, int] = field(default_factory=dict)

    context_mode: str = "zone_hour"

    def __post_init__(self):
        if not self.invalidated:
            self.invalidated = {v: 0 for v in OUTLIER_VARS}
            self.invalidated["speed_incoherence_hard"] = 0
            self.invalidated["tip_pct_rule"] = 0
            self.invalidated["negative_total_amount_std"] = 0
        if not self.winsorized:
            self.winsorized = {v: 0 for v in OUTLIER_VARS}
        if not self.final_valid_flags_sum:
            self.final_valid_flags_sum = {
                "is_valid_for_demand": 0,
                "is_valid_for_price": 0,
                "is_valid_for_duration": 0,
                "is_valid_for_distance": 0,
                "is_valid_for_tip": 0,
            }
        if not self.final_non_null_sum:
            self.final_non_null_sum = {
                "trip_duration_min": 0,
                "trip_distance": 0,
                "total_amount_std": 0,
                "tip_pct": 0,
                "price_per_mile": 0,
            }


# ---------------------------------------------------------------------
# Utilidades de lectura
# ---------------------------------------------------------------------
def _list_parquets(folder: Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(folder.glob("*.parquet"))


def iter_validated_tlc_files(
    validated_base: Path,
    services: Iterable[str] = SERVICES,
) -> Iterator[Tuple[str, Path]]:
    validated_base = Path(validated_base).resolve()
    if not validated_base.exists():
        console.print(f"[bold yellow]WARNING[/bold yellow] No existe validated base: {validated_base}")
        return

    for service in services:
        folder = validated_base / service
        files = _list_parquets(folder)

        if not files:
            console.print(f"[yellow]Aviso:[/yellow] No hay parquets en {folder}. Se omite {service}.")
            continue

        console.print(f"[cyan]{service}[/cyan]: {len(files)} parquets encontrados")
        for fp in files:
            yield service, fp


def get_existing_columns(fp: Path) -> List[str]:
    schema = pq.read_schema(fp)
    names = set(schema.names)
    return [c for c in NEEDED_COLS if c in names]


def iter_parquet_batches(fp: Path, batch_size: int = BATCH_SIZE_DEFAULT):
    cols = get_existing_columns(fp)
    pf = pq.ParquetFile(fp)
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        yield batch.to_pandas()


def _coalesce_cols(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series([pd.NA] * len(df), index=df.index)


def _parse_year_month_from_filename(fp: Path) -> Tuple[Optional[int], Optional[int]]:
    m = re.search(r"(20\d{2})-(\d{2})", fp.name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def output_partition_exists(out_dir: Path, service: str, year: Optional[int], month: Optional[int]) -> bool:
    if year is None or month is None:
        return False
    part_dir = out_dir / f"service_type={service}" / f"year={int(year)}" / f"month={int(month)}"
    return part_dir.exists() and any(part_dir.glob("*.parquet"))


# ---------------------------------------------------------------------
# Estandarización base
# ---------------------------------------------------------------------
def standardize_tlc(df: pd.DataFrame, service: str) -> pd.DataFrame:
    """
    Capa 2: armonización analítica.
    No repite validaciones estructurales duras de Capa 1.
    """
    if df.empty:
        return df

    df["service_type"] = service

    df["pickup_datetime"] = _coalesce_cols(
        df, ["tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime"]
    )
    df["dropoff_datetime"] = _coalesce_cols(
        df, ["tpep_dropoff_datetime", "lpep_dropoff_datetime", "dropoff_datetime"]
    )

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"], errors="coerce")

    df["pu_location_id"] = pd.to_numeric(
        _coalesce_cols(df, ["PULocationID", "pu_location_id"]), errors="coerce"
    )
    df["do_location_id"] = pd.to_numeric(
        _coalesce_cols(df, ["DOLocationID", "do_location_id"]), errors="coerce"
    )

    numeric_cols = [
        "total_amount",
        "fare_amount",
        "tip_amount",
        "tips",
        "tolls_amount",
        "tolls",
        "airport_fee",
        "Airport_fee",
        "congestion_surcharge",
        "base_passenger_fare",
        "trip_distance",
        "trip_miles",
        "VendorID",
        "passenger_count",
        "RatecodeID",
        "payment_type",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "airport_fee" not in df.columns and "Airport_fee" in df.columns:
        df["airport_fee"] = df["Airport_fee"]
    if "Airport_fee" in df.columns:
        df = df.drop(columns=["Airport_fee"])

    if "trip_distance" not in df.columns and "trip_miles" in df.columns:
        df["trip_distance"] = df["trip_miles"]
    elif "trip_distance" in df.columns and "trip_miles" in df.columns:
        df["trip_distance"] = df["trip_distance"].fillna(df["trip_miles"])

    zero = pd.Series(0.0, index=df.index, dtype="float64")

    airport_fee_any = df.get("airport_fee", zero).fillna(0.0)
    tips_any = df.get("tip_amount", zero).fillna(0.0)
    if "tips" in df.columns:
        tips_any = tips_any + df["tips"].fillna(0.0)

    tolls_any = df.get("tolls_amount", zero).fillna(0.0)
    if "tolls" in df.columns:
        tolls_any = tolls_any + df["tolls"].fillna(0.0)

    congestion_any = df.get("congestion_surcharge", zero).fillna(0.0)
    base_fare = df.get("base_passenger_fare", zero).fillna(0.0)

    fallback_total = base_fare + tips_any + tolls_any + airport_fee_any + congestion_any
    if "total_amount" in df.columns:
        df["total_amount_std"] = df["total_amount"].fillna(fallback_total)
    else:
        df["total_amount_std"] = fallback_total

    df["date"] = df["pickup_datetime"].dt.date
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["hour"] = df["pickup_datetime"].dt.hour

    df = df[(df["year"] >= MIN_YEAR) & (df["year"] <= MAX_YEAR)]

    dow0 = df["pickup_datetime"].dt.dayofweek
    df["day_of_week"] = ((dow0 + 1) % 7) + 1
    df["is_weekend"] = df["day_of_week"].isin([1, 7]).astype("int8")
    df["week_of_year"] = df["pickup_datetime"].dt.isocalendar().week.astype("Int64")

    df["trip_duration_min"] = (
        (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds() / 60.0
    )

    fare_amount = pd.to_numeric(
        df.get("fare_amount", pd.Series(np.nan, index=df.index)), errors="coerce"
    )
    tip_amount = pd.to_numeric(
        df.get("tip_amount", pd.Series(np.nan, index=df.index)), errors="coerce"
    )
    total_amount_std = pd.to_numeric(
        df.get("total_amount_std", pd.Series(np.nan, index=df.index)), errors="coerce"
    )
    trip_distance = pd.to_numeric(
        df.get("trip_distance", pd.Series(np.nan, index=df.index)), errors="coerce"
    )

    df["tip_pct"] = np.where(fare_amount > 0, tip_amount / fare_amount, np.nan)
    df["price_per_mile"] = np.where(trip_distance > 0, total_amount_std / trip_distance, np.nan)

    float_cols = [
        "trip_distance",
        "trip_duration_min",
        "total_amount_std",
        "tip_pct",
        "price_per_mile",
        "total_amount",
        "fare_amount",
        "tip_amount",
        "tips",
        "tolls_amount",
        "tolls",
        "congestion_surcharge",
        "airport_fee",
        "base_passenger_fare",
        "trip_miles",
    ]
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    int_cols = [
        "pu_location_id",
        "do_location_id",
        "VendorID",
        "passenger_count",
        "RatecodeID",
        "payment_type",
    ]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int32")

    for flag_col in [
        "is_valid_for_demand",
        "is_valid_for_price",
        "is_valid_for_duration",
        "is_valid_for_distance",
        "is_valid_for_tip",
    ]:
        if flag_col not in df.columns:
            df[flag_col] = np.int8(1)

    return df[[c for c in FINAL_COLS if c in df.columns]]


# ---------------------------------------------------------------------
# Limpieza base
# ---------------------------------------------------------------------
def remove_rows_missing_core(df: pd.DataFrame, stats: FileStats) -> pd.DataFrame:
    if df.empty:
        return df

    before = len(df)
    mask = (
        df["pickup_datetime"].notna()
        & df["pu_location_id"].notna()
        & df["hour"].between(0, 23, inclusive="both")
    )
    df = df.loc[mask].copy()
    stats.rows_removed_missing_core += before - len(df)
    return df


def enforce_expected_month(df: pd.DataFrame, fp: Path, stats: FileStats) -> pd.DataFrame:
    year, month = _parse_year_month_from_filename(fp)
    if year is None or month is None or df.empty:
        return df

    before = len(df)
    df = df[(df["year"] == year) & (df["month"] == month)].copy()
    stats.rows_removed_wrong_month += before - len(df)
    return df


def remove_exact_duplicates_chunkwise(
    df: pd.DataFrame,
    stats: FileStats,
    seen_hashes: set[int],
) -> pd.DataFrame:
    if df.empty:
        return df

    cols = [c for c in DEDUP_COLS if c in df.columns]
    if not cols:
        return df

    hashed = pd.util.hash_pandas_object(df[cols], index=False).astype("uint64")
    dup_mask = hashed.isin(seen_hashes)

    dup_count = int(dup_mask.sum())
    stats.duplicated_exact_found += dup_count
    stats.rows_removed_duplicates_exact += dup_count

    new_hashes = hashed[~dup_mask].tolist()
    seen_hashes.update(int(x) for x in new_hashes)

    return df.loc[~dup_mask].copy()


def refresh_validity_flags(df: pd.DataFrame) -> None:
    df["is_valid_for_demand"] = (
        df["pickup_datetime"].notna()
        & df["pu_location_id"].notna()
        & df["hour"].between(0, 23, inclusive="both")
    ).astype("int8")

    df["is_valid_for_price"] = df["total_amount_std"].notna().astype("int8")
    df["is_valid_for_duration"] = df["trip_duration_min"].notna().astype("int8")
    df["is_valid_for_distance"] = df["trip_distance"].notna().astype("int8")
    df["is_valid_for_tip"] = df["tip_pct"].notna().astype("int8")


def apply_rule_based_cleaning(df: pd.DataFrame, stats: FileStats) -> pd.DataFrame:
    """
    Guardarraíles suaves y duros antes del detector contextual.
    """
    if df.empty:
        return df

    dur = pd.to_numeric(df["trip_duration_min"], errors="coerce")
    dist = pd.to_numeric(df["trip_distance"], errors="coerce")
    total_std = pd.to_numeric(df["total_amount_std"], errors="coerce")
    tip_pct = pd.to_numeric(df["tip_pct"], errors="coerce")

    bad_dur = dur.notna() & (dur <= 0)
    df.loc[bad_dur, "trip_duration_min"] = np.nan

    bad_dist = dist.notna() & (dist <= 0)
    df.loc[bad_dist, "trip_distance"] = np.nan

    bad_total = total_std.notna() & (total_std < MAX_ABS_NEG_AMOUNT)
    stats.invalidated["negative_total_amount_std"] += int(bad_total.sum())
    df.loc[bad_total, "total_amount_std"] = np.nan
    df.loc[bad_total, "price_per_mile"] = np.nan

    bad_tip_pct = tip_pct.notna() & ((tip_pct < 0) | (tip_pct > MAX_TIP_PCT))
    stats.invalidated["tip_pct_rule"] += int(bad_tip_pct.sum())
    df.loc[bad_tip_pct, "tip_pct"] = np.nan

    dur_h = pd.to_numeric(df["trip_duration_min"], errors="coerce") / 60.0
    dist2 = pd.to_numeric(df["trip_distance"], errors="coerce")
    speed = np.where((dur_h > 0) & dist2.notna(), dist2 / dur_h, np.nan)
    speed_s = pd.Series(speed, index=df.index)

    hard_speed_bad = speed_s.notna() & (speed_s > MAX_SPEED_MPH_HARD)
    stats.invalidated["speed_incoherence_hard"] += int(hard_speed_bad.sum())

    df.loc[hard_speed_bad, "trip_distance"] = np.nan
    df.loc[hard_speed_bad, "trip_duration_min"] = np.nan
    df.loc[hard_speed_bad, "price_per_mile"] = np.nan

    ppm = pd.to_numeric(df["price_per_mile"], errors="coerce")
    bad_ppm = ppm.notna() & ~np.isfinite(ppm)
    df.loc[bad_ppm, "price_per_mile"] = np.nan

    refresh_validity_flags(df)
    return df


# ---------------------------------------------------------------------
# Thresholds contextuales con DuckDB
# ---------------------------------------------------------------------
def get_primary_group_cols(context_mode: str) -> List[str]:
    if context_mode == "zone_hour":
        return ["pu_location_id", "hour"]
    if context_mode == "zone_hour_weekend":
        return ["pu_location_id", "hour", "is_weekend"]
    raise ValueError(f"context_mode no soportado: {context_mode}")


def _stage_glob(stage_files: List[Path]) -> str:
    if not stage_files:
        raise ValueError("No hay stage_files")
    return str(stage_files[0].parent / "stage_*.parquet")


def get_duckdb_conn(threads: int = 4) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={int(threads)}")
    return con


def _quantile_exprs(var: str, suffix: str) -> str:
    return f"""
        quantile_cont({var}, 0.01) AS {var}_p01_{suffix},
        quantile_cont({var}, 0.05) AS {var}_p05_{suffix},
        quantile_cont({var}, 0.95) AS {var}_p95_{suffix},
        quantile_cont({var}, 0.99) AS {var}_p99_{suffix},
        count({var}) AS {var}_n_{suffix}
    """


def duckdb_quantiles_for_scope(
    con: duckdb.DuckDBPyConnection,
    parquet_glob: str,
    group_cols: List[str],
    outlier_vars: List[str],
    suffix: str,
) -> pd.DataFrame:
    select_group = ", ".join(group_cols)
    group_by = ", ".join(group_cols)

    pieces = []
    for var in outlier_vars:
        pieces.append(_quantile_exprs(var, suffix))
    metrics_sql = ",\n".join(pieces)

    sql = f"""
    SELECT
        {select_group},
        {metrics_sql}
    FROM read_parquet('{parquet_glob}')
    GROUP BY {group_by}
    """

    out = con.execute(sql).df()

    for var in outlier_vars:
        ncol = f"{var}_n_{suffix}"
        for qcol in [
            f"{var}_p01_{suffix}",
            f"{var}_p05_{suffix}",
            f"{var}_p95_{suffix}",
            f"{var}_p99_{suffix}",
        ]:
            out.loc[out[ncol] < MIN_GROUP_SIZE, qcol] = np.nan

    return out


def duckdb_global_quantiles(
    con: duckdb.DuckDBPyConnection,
    parquet_glob: str,
    outlier_vars: List[str],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}

    for var in outlier_vars:
        sql = f"""
        SELECT
            quantile_cont({var}, 0.01) AS p01,
            quantile_cont({var}, 0.05) AS p05,
            quantile_cont({var}, 0.95) AS p95,
            quantile_cont({var}, 0.99) AS p99,
            count({var}) AS n
        FROM read_parquet('{parquet_glob}')
        """
        row = con.execute(sql).fetchone()
        n = float(row[4] or 0)

        if n < MIN_GROUP_SIZE:
            out[var] = {
                f"{var}_p01_g": np.nan,
                f"{var}_p05_g": np.nan,
                f"{var}_p95_g": np.nan,
                f"{var}_p99_g": np.nan,
                f"{var}_n_g": n,
            }
        else:
            out[var] = {
                f"{var}_p01_g": float(row[0]) if row[0] is not None else np.nan,
                f"{var}_p05_g": float(row[1]) if row[1] is not None else np.nan,
                f"{var}_p95_g": float(row[2]) if row[2] is not None else np.nan,
                f"{var}_p99_g": float(row[3]) if row[3] is not None else np.nan,
                f"{var}_n_g": n,
            }

    return out


def build_threshold_store_duckdb(
    stage_files: List[Path],
    context_mode: str,
    duckdb_threads: int = 4,
) -> Dict[str, Dict[str, object]]:
    primary_cols = get_primary_group_cols(context_mode)
    parquet_glob = _stage_glob(stage_files)

    con = get_duckdb_conn(threads=duckdb_threads)

    primary_df = duckdb_quantiles_for_scope(
        con=con,
        parquet_glob=parquet_glob,
        group_cols=primary_cols,
        outlier_vars=OUTLIER_VARS,
        suffix="p",
    )

    zone_df = duckdb_quantiles_for_scope(
        con=con,
        parquet_glob=parquet_glob,
        group_cols=["pu_location_id"],
        outlier_vars=OUTLIER_VARS,
        suffix="z",
    )

    hour_df = duckdb_quantiles_for_scope(
        con=con,
        parquet_glob=parquet_glob,
        group_cols=["hour"],
        outlier_vars=OUTLIER_VARS,
        suffix="h",
    )

    global_dict = duckdb_global_quantiles(
        con=con,
        parquet_glob=parquet_glob,
        outlier_vars=OUTLIER_VARS,
    )

    con.close()

    store: Dict[str, Dict[str, object]] = {}
    for var in OUTLIER_VARS:
        store[var] = {
            "primary": primary_df[
                primary_cols + [
                    f"{var}_p01_p",
                    f"{var}_p05_p",
                    f"{var}_p95_p",
                    f"{var}_p99_p",
                    f"{var}_n_p",
                ]
            ].copy(),
            "z": zone_df[
                ["pu_location_id"] + [
                    f"{var}_p01_z",
                    f"{var}_p05_z",
                    f"{var}_p95_z",
                    f"{var}_p99_z",
                    f"{var}_n_z",
                ]
            ].copy(),
            "h": hour_df[
                ["hour"] + [
                    f"{var}_p01_h",
                    f"{var}_p05_h",
                    f"{var}_p95_h",
                    f"{var}_p99_h",
                    f"{var}_n_h",
                ]
            ].copy(),
            "g": global_dict[var],
        }

    return store


def resolve_threshold_series(df: pd.DataFrame, var: str, which: str, gvals: Dict[str, float]) -> pd.Series:
    cols = [
        f"{var}_{which}_p",
        f"{var}_{which}_z",
        f"{var}_{which}_h",
    ]

    s = pd.Series(np.nan, index=df.index, dtype="float64")
    for c in cols:
        if c in df.columns:
            s = s.fillna(df[c])

    gcol = f"{var}_{which}_g"
    if gcol in gvals:
        s = s.fillna(float(gvals[gcol]))
    return s


def invalidate_for_var(df: pd.DataFrame, var: str, mask: pd.Series) -> None:
    if not mask.any():
        return

    if var == "trip_duration_min":
        df.loc[mask, "trip_duration_min"] = np.nan
        df.loc[mask, "price_per_mile"] = np.nan

    elif var == "trip_distance":
        df.loc[mask, "trip_distance"] = np.nan
        df.loc[mask, "price_per_mile"] = np.nan

    elif var == "total_amount_std":
        df.loc[mask, "total_amount_std"] = np.nan
        df.loc[mask, "price_per_mile"] = np.nan

    elif var == "price_per_mile":
        df.loc[mask, "price_per_mile"] = np.nan


def apply_contextual_cleaning_chunk(
    df: pd.DataFrame,
    threshold_store: Dict[str, Dict[str, object]],
    stats: FileStats,
    context_mode: str,
) -> pd.DataFrame:
    if df.empty:
        return df

    primary_cols = get_primary_group_cols(context_mode)

    for var in OUTLIER_VARS:
        if var not in df.columns:
            continue

        pack = threshold_store[var]
        primary = pack["primary"]
        z = pack["z"]
        h = pack["h"]
        g = pack["g"]

        if isinstance(primary, pd.DataFrame) and not primary.empty:
            df = df.merge(primary, on=primary_cols, how="left")
        if isinstance(z, pd.DataFrame) and not z.empty:
            df = df.merge(z, on=["pu_location_id"], how="left")
        if isinstance(h, pd.DataFrame) and not h.empty:
            df = df.merge(h, on=["hour"], how="left")

        s = pd.to_numeric(df[var], errors="coerce")

        low_ext = resolve_threshold_series(df, var, "p01", g)
        low_win = resolve_threshold_series(df, var, "p05", g)
        high_win = resolve_threshold_series(df, var, "p95", g)
        high_ext = resolve_threshold_series(df, var, "p99", g)

        extreme_mask = (
            s.notna()
            & (
                (low_ext.notna() & (s < low_ext))
                | (high_ext.notna() & (s > high_ext))
            )
        )

        moderate_mask = (
            s.notna()
            & ~extreme_mask
            & (
                (low_win.notna() & (s < low_win))
                | (high_win.notna() & (s > high_win))
            )
        )

        stats.invalidated[var] += int(extreme_mask.sum())
        stats.winsorized[var] += int(moderate_mask.sum())

        invalidate_for_var(df, var, extreme_mask)

        low_clip_mask = moderate_mask & low_win.notna() & (s < low_win)
        high_clip_mask = moderate_mask & high_win.notna() & (s > high_win)

        df.loc[low_clip_mask, var] = low_win[low_clip_mask].astype("float32")
        df.loc[high_clip_mask, var] = high_win[high_clip_mask].astype("float32")

        aux_cols = [c for c in df.columns if c.startswith(f"{var}_p") or c.startswith(f"{var}_n_")]
        if aux_cols:
            df = df.drop(columns=aux_cols)

    fare_amount = pd.to_numeric(df.get("fare_amount", pd.Series(np.nan, index=df.index)), errors="coerce")
    tip_amount = pd.to_numeric(df.get("tip_amount", pd.Series(np.nan, index=df.index)), errors="coerce")
    trip_distance = pd.to_numeric(df.get("trip_distance", pd.Series(np.nan, index=df.index)), errors="coerce")
    total_amount_std = pd.to_numeric(df.get("total_amount_std", pd.Series(np.nan, index=df.index)), errors="coerce")

    df["tip_pct"] = np.where(fare_amount > 0, tip_amount / fare_amount, np.nan).astype("float32")
    df.loc[(df["tip_pct"] < 0) | (df["tip_pct"] > MAX_TIP_PCT), "tip_pct"] = np.nan

    df["price_per_mile"] = np.where(
        trip_distance > 0,
        total_amount_std / trip_distance,
        np.nan,
    ).astype("float32")

    refresh_validity_flags(df)
    return df


# ---------------------------------------------------------------------
# Escritura y reporting
# ---------------------------------------------------------------------
def write_partitioned(df: pd.DataFrame, out_dir: Path):
    if df.empty:
        return

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for (svc, y, m), g in df.groupby(["service_type", "year", "month"], dropna=False):
        part_dir = out_dir / f"service_type={svc}" / f"year={int(y)}" / f"month={int(m)}"
        part_dir.mkdir(parents=True, exist_ok=True)
        fname = f"part_{uuid.uuid4().hex}.parquet"
        g.to_parquet(part_dir / fname, index=False, engine="pyarrow")


def finalize_stats_from_output(df: pd.DataFrame, stats: FileStats) -> None:
    stats.rows_out += len(df)

    for col in stats.final_valid_flags_sum:
        if col in df.columns:
            stats.final_valid_flags_sum[col] += int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())

    for col in stats.final_non_null_sum:
        if col in df.columns:
            stats.final_non_null_sum[col] += int(df[col].notna().sum())


def build_file_report_dict(stats: FileStats) -> Dict[str, object]:
    rows_out = max(stats.rows_out, 1)

    report = asdict(stats)
    report["pct_rows_removed_missing_core"] = round(100 * stats.rows_removed_missing_core / max(stats.rows_in, 1), 4)
    report["pct_rows_removed_wrong_month"] = round(100 * stats.rows_removed_wrong_month / max(stats.rows_in, 1), 4)
    report["pct_rows_removed_duplicates_exact"] = round(100 * stats.rows_removed_duplicates_exact / max(stats.rows_in, 1), 4)
    report["pct_rows_out_vs_in"] = round(100 * stats.rows_out / max(stats.rows_in, 1), 4)

    report["final_valid_flags_pct"] = {
        k: round(100 * v / rows_out, 4) for k, v in stats.final_valid_flags_sum.items()
    }
    report["final_non_null_pct"] = {
        k: round(100 * v / rows_out, 4) for k, v in stats.final_non_null_sum.items()
    }

    report["winsorized_pct_over_rows_out"] = {
        k: round(100 * v / rows_out, 4) for k, v in stats.winsorized.items()
    }
    report["invalidated_pct_over_rows_out"] = {
        k: round(100 * v / rows_out, 4) for k, v in stats.invalidated.items()
    }

    return report


def write_report(stats: FileStats, src_fp: Path, reports_dir: Path):
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_fp = reports_dir / f"{src_fp.stem}.json"
    with open(out_fp, "w", encoding="utf-8") as f:
        json.dump(build_file_report_dict(stats), f, ensure_ascii=False, indent=2)


def flatten_stats_for_csv(stats: FileStats) -> Dict[str, object]:
    rows_out = max(stats.rows_out, 1)

    row: Dict[str, object] = {
        "file_name": stats.file_name,
        "service": stats.service,
        "file_year": stats.file_year,
        "file_month": stats.file_month,
        "context_mode": stats.context_mode,
        "rows_in": stats.rows_in,
        "rows_after_standardize": stats.rows_after_standardize,
        "rows_removed_missing_core": stats.rows_removed_missing_core,
        "rows_removed_wrong_month": stats.rows_removed_wrong_month,
        "duplicated_exact_found": stats.duplicated_exact_found,
        "rows_removed_duplicates_exact": stats.rows_removed_duplicates_exact,
        "rows_out": stats.rows_out,
        "pct_rows_out_vs_in": round(100 * stats.rows_out / max(stats.rows_in, 1), 4),
    }

    for k, v in stats.invalidated.items():
        row[f"invalidated__{k}"] = v
        row[f"invalidated_pct__{k}"] = round(100 * v / rows_out, 4)

    for k, v in stats.winsorized.items():
        row[f"winsorized__{k}"] = v
        row[f"winsorized_pct__{k}"] = round(100 * v / rows_out, 4)

    for k, v in stats.final_valid_flags_sum.items():
        row[f"valid_sum__{k}"] = v
        row[f"valid_pct__{k}"] = round(100 * v / rows_out, 4)

    for k, v in stats.final_non_null_sum.items():
        row[f"non_null_sum__{k}"] = v
        row[f"non_null_pct__{k}"] = round(100 * v / rows_out, 4)

    return row


def print_file_summary(stats: FileStats) -> None:
    t = Table(show_header=True, header_style="bold white", title=f"Resumen fichero: {stats.file_name}")
    t.add_column("Métrica", style="bold cyan")
    t.add_column("Valor", justify="right")

    t.add_row("service", stats.service)
    t.add_row("context_mode", stats.context_mode)
    t.add_row("rows_in", f"{stats.rows_in:,}")
    t.add_row("after_standardize", f"{stats.rows_after_standardize:,}")
    t.add_row("removed_missing_core", f"{stats.rows_removed_missing_core:,}")
    t.add_row("removed_wrong_month", f"{stats.rows_removed_wrong_month:,}")
    t.add_row("dup_exact_removed", f"{stats.rows_removed_duplicates_exact:,}")
    t.add_row("rows_out", f"{stats.rows_out:,}")

    for k, v in stats.invalidated.items():
        t.add_row(f"invalidated::{k}", f"{v:,}")

    for k, v in stats.winsorized.items():
        t.add_row(f"winsorized::{k}", f"{v:,}")

    console.print(t)


def write_global_audit_csv(audit_rows: List[Dict[str, object]], audit_dir: Path):
    audit_dir.mkdir(parents=True, exist_ok=True)
    out_fp = audit_dir / "capa2_audit_summary.csv"
    df = pd.DataFrame(audit_rows)
    df.to_csv(out_fp, index=False, encoding="utf-8")


# ---------------------------------------------------------------------
# Pipeline de fichero
# ---------------------------------------------------------------------
def process_file(
    service: str,
    fp: Path,
    out_dir: Path,
    reports_dir: Path,
    tmp_root: Path,
    batch_size: int = BATCH_SIZE_DEFAULT,
    context_mode: str = "zone_hour_weekend",
    show_file_summary: bool = True,
    duckdb_threads: int = 4,
) -> FileStats:
    """
    Estrategia:
    1) leer por batches, estandarizar y limpiar base
    2) guardar staging temporal
    3) calcular thresholds contextuales desde staging con DuckDB
    4) releer staging y aplicar limpieza contextual por chunks
    """
    file_year, file_month = _parse_year_month_from_filename(fp)

    stats = FileStats(
        file_name=fp.name,
        service=service,
        file_year=file_year,
        file_month=file_month,
        context_mode=context_mode,
    )

    stage_dir = tmp_root / f"{service}_{fp.stem}_{uuid.uuid4().hex}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[int] = set()
    stage_files: List[Path] = []

    for i, batch_df in enumerate(iter_parquet_batches(fp, batch_size=batch_size)):
        stats.rows_in += len(batch_df)

        std = standardize_tlc(batch_df, service)
        stats.rows_after_standardize += len(std)

        std = remove_rows_missing_core(std, stats)
        std = enforce_expected_month(std, fp, stats)
        std = apply_rule_based_cleaning(std, stats)
        std = remove_exact_duplicates_chunkwise(std, stats, seen_hashes)

        if std.empty:
            continue

        stage_fp = stage_dir / f"stage_{i:04d}.parquet"
        std.to_parquet(stage_fp, index=False, engine="pyarrow")
        stage_files.append(stage_fp)

        del std
        del batch_df

    if not stage_files:
        write_report(stats, fp, reports_dir)
        if show_file_summary:
            print_file_summary(stats)
        shutil.rmtree(stage_dir, ignore_errors=True)
        return stats

    threshold_store = build_threshold_store_duckdb(
        stage_files=stage_files,
        context_mode=context_mode,
        duckdb_threads=duckdb_threads,
    )

    for stage_fp in stage_files:
        df = pd.read_parquet(stage_fp)
        if df.empty:
            continue

        df = apply_contextual_cleaning_chunk(
            df=df,
            threshold_store=threshold_store,
            stats=stats,
            context_mode=context_mode,
        )

        final_cols = [c for c in FINAL_COLS if c in df.columns]
        df = df[final_cols]

        finalize_stats_from_output(df, stats)
        write_partitioned(df, out_dir)

        del df

    write_report(stats, fp, reports_dir)
    if show_file_summary:
        print_file_summary(stats)

    shutil.rmtree(stage_dir, ignore_errors=True)
    return stats


def run_one_file(task: Tuple[str, Path, Path, Path, Path, int, str, bool, int]) -> FileStats:
    service, fp, out_dir, reports_dir, tmp_dir, batch_size, context_mode, show_file_summary, duckdb_threads = task
    return process_file(
        service=service,
        fp=fp,
        out_dir=out_dir,
        reports_dir=reports_dir,
        tmp_root=tmp_dir,
        batch_size=batch_size,
        context_mode=context_mode,
        show_file_summary=show_file_summary,
        duckdb_threads=duckdb_threads,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    console.print(Panel.fit("[bold cyan]CAPA 2 - TLC: LIMPIEZA ANALÍTICA + FEATURE ENGINEERING[/bold cyan]"))

    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default=str(obtener_ruta("data/validated")))
    p.add_argument("--out-dir", default=str(obtener_ruta("data/standardized")))
    p.add_argument("--reports-dir", default=str(obtener_ruta("outputs/procesamiento/capa2_reports")))
    p.add_argument("--audit-dir", default=str(obtener_ruta("outputs/procesamiento/capa2_audit")))
    p.add_argument("--tmp-dir", default=str(obtener_ruta("data/tmp/capa2_stage")))
    p.add_argument("--mode", choices=["append", "overwrite"], default="append")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    p.add_argument(
        "--context-mode",
        choices=["zone_hour", "zone_hour_weekend"],
        default="zone_hour_weekend",
        help="Detector contextual principal",
    )
    p.add_argument(
        "--services",
        nargs="+",
        choices=list(SERVICES),
        default=list(SERVICES),
        help="Servicios a procesar",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Saltar meses cuya salida final ya existe",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Número de procesos en paralelo por fichero",
    )
    p.add_argument(
        "--duckdb-threads",
        type=int,
        default=4,
        help="Threads internos para DuckDB",
    )
    p.add_argument("--hide-file-summary", action="store_true")
    args = p.parse_args()

    validated_base = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    reports_dir = Path(args.reports_dir)
    audit_dir = Path(args.audit_dir)
    tmp_dir = Path(args.tmp_dir)

    cfg_table = Table(show_header=True, header_style="bold white", title="Configuración Capa2 TLC")
    cfg_table.add_column("Campo", style="bold cyan")
    cfg_table.add_column("Valor")
    cfg_table.add_row("raw_dir", str(validated_base))
    cfg_table.add_row("out_dir", str(out_dir))
    cfg_table.add_row("reports_dir", str(reports_dir))
    cfg_table.add_row("audit_dir", str(audit_dir))
    cfg_table.add_row("tmp_dir", str(tmp_dir))
    cfg_table.add_row("mode", args.mode)
    cfg_table.add_row("batch_size", f"{args.batch_size:,}")
    cfg_table.add_row("min_group_size", str(MIN_GROUP_SIZE))
    cfg_table.add_row("context_mode", args.context_mode)
    cfg_table.add_row("services", ", ".join(args.services))
    cfg_table.add_row("skip_existing", str(args.skip_existing))
    cfg_table.add_row("max_workers", str(args.max_workers))
    cfg_table.add_row("duckdb_threads", str(args.duckdb_threads))
    console.print(cfg_table)

    if args.mode == "overwrite":
        if out_dir.exists():
            console.print(f"[yellow]Limpiando salida (overwrite):[/yellow] {out_dir}")
            shutil.rmtree(out_dir)
        if reports_dir.exists():
            shutil.rmtree(reports_dir)
        if audit_dir.exists():
            shutil.rmtree(audit_dir)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    tmp_dir.mkdir(parents=True, exist_ok=True)

    service_stats: Dict[str, Dict[str, int]] = {}
    audit_rows: List[Dict[str, object]] = []

    tasks: List[Tuple[str, Path, Path, Path, Path, int, str, bool, int]] = []
    skipped = 0

    for service, fp in iter_validated_tlc_files(validated_base, services=args.services):
        file_year, file_month = _parse_year_month_from_filename(fp)

        if args.skip_existing and output_partition_exists(out_dir, service, file_year, file_month):
            console.print(
                f"[yellow]Saltando {fp.name} porque ya existe salida para {service}-{file_year}-{file_month:02d}[/yellow]"
            )
            skipped += 1
            continue

        tasks.append((
            service,
            fp,
            out_dir,
            reports_dir,
            tmp_dir,
            args.batch_size,
            args.context_mode,
            not args.hide_file_summary,
            args.duckdb_threads,
        ))

    results: List[FileStats] = []

    if args.max_workers == 1:
        for task in tasks:
            service, fp, *_ = task
            with console.status(f"[cyan]Procesando {service} -> {fp.name}[/cyan]"):
                results.append(run_one_file(task))
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
            future_to_task = {ex.submit(run_one_file, task): task for task in tasks}
            for fut in as_completed(future_to_task):
                task = future_to_task[fut]
                service, fp, *_ = task
                try:
                    st = fut.result()
                    results.append(st)
                    console.print(f"[green]OK[/green] {service} -> {fp.name}")
                except Exception as e:
                    console.print(f"[bold red]ERROR[/bold red] {service} -> {fp.name}: {e}")
                    raise

    for st in results:
        service_stats.setdefault(st.service, {
            "files": 0,
            "rows_in": 0,
            "rows_after_standardize": 0,
            "rows_removed_missing_core": 0,
            "rows_removed_wrong_month": 0,
            "rows_removed_duplicates_exact": 0,
            "rows_out": 0,
        })
        service_stats[st.service]["files"] += 1
        service_stats[st.service]["rows_in"] += st.rows_in
        service_stats[st.service]["rows_after_standardize"] += st.rows_after_standardize
        service_stats[st.service]["rows_removed_missing_core"] += st.rows_removed_missing_core
        service_stats[st.service]["rows_removed_wrong_month"] += st.rows_removed_wrong_month
        service_stats[st.service]["rows_removed_duplicates_exact"] += st.rows_removed_duplicates_exact
        service_stats[st.service]["rows_out"] += st.rows_out

        audit_rows.append(flatten_stats_for_csv(st))

    write_global_audit_csv(audit_rows, audit_dir)

    summary = Table(show_header=True, header_style="bold magenta", title="Resumen Capa2 TLC por servicio")
    summary.add_column("Servicio", style="bold white")
    summary.add_column("Files", justify="right")
    summary.add_column("Rows in", justify="right")
    summary.add_column("After std", justify="right")
    summary.add_column("Missing core", justify="right")
    summary.add_column("Wrong month", justify="right")
    summary.add_column("Dup exact", justify="right")
    summary.add_column("Rows out", justify="right")

    total = {
        "files": 0,
        "rows_in": 0,
        "rows_after_standardize": 0,
        "rows_removed_missing_core": 0,
        "rows_removed_wrong_month": 0,
        "rows_removed_duplicates_exact": 0,
        "rows_out": 0,
    }

    for service in sorted(service_stats):
        st = service_stats[service]
        for k in total:
            total[k] += st[k]

        summary.add_row(
            service,
            f"{st['files']:,}",
            f"{st['rows_in']:,}",
            f"{st['rows_after_standardize']:,}",
            f"{st['rows_removed_missing_core']:,}",
            f"{st['rows_removed_wrong_month']:,}",
            f"{st['rows_removed_duplicates_exact']:,}",
            f"{st['rows_out']:,}",
        )

    summary.add_row(
        "TOTAL",
        f"{total['files']:,}",
        f"{total['rows_in']:,}",
        f"{total['rows_after_standardize']:,}",
        f"{total['rows_removed_missing_core']:,}",
        f"{total['rows_removed_wrong_month']:,}",
        f"{total['rows_removed_duplicates_exact']:,}",
        f"{total['rows_out']:,}",
    )
    console.print(summary)

    if skipped:
        console.print(f"[yellow]Saltados por skip-existing:[/yellow] {skipped:,}")

    console.print(f"[bold green]OK[/bold green] Capa 2 guardada en: {out_dir}")
    console.print(f"[bold green]OK[/bold green] Reports JSON guardados en: {reports_dir}")
    console.print(f"[bold green]OK[/bold green] Audit CSV guardado en: {audit_dir / 'capa2_audit_summary.csv'}")


if __name__ == "__main__":
    main()