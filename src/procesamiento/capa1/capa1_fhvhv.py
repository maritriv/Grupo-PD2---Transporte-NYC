# src/procesamiento/capa1/capa1_fhvhv.py

from __future__ import annotations

import argparse
import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import obtener_ruta

console = Console()

# ------------------------------------------------------------
# HVFHV / HVFHS Data Dictionary (Mar 18, 2025)
# ------------------------------------------------------------
EXPECTED_COLUMNS: List[str] = [
    "hvfhs_license_num",
    "dispatching_base_num",
    "originating_base_num",
    "request_datetime",
    "on_scene_datetime",
    "pickup_datetime",
    "dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_miles",
    "trip_time",
    "base_passenger_fare",
    "tolls",
    "bcf",
    "sales_tax",
    "congestion_surcharge",
    "airport_fee",
    "tips",
    "driver_pay",
    "shared_request_flag",
    "shared_match_flag",
    "access_a_ride_flag",
    "wav_request_flag",
    "wav_match_flag",
    "cbd_congestion_fee",
]

KNOWN_HVFHS = {"HV0002", "HV0003", "HV0004", "HV0005"}

HVFHS_PATTERN = re.compile(r"^HV\d{4}$")
BASE_NUM_PATTERN = re.compile(r"^[A-Z]\d{5}$")  # típico TLC: B00013

YN_FLAGS = {
    "shared_request_flag",
    "shared_match_flag",
    "access_a_ride_flag",
    "wav_request_flag",
    "wav_match_flag",
}
ALLOWED_YN = {"Y", "N"}

FLOAT_COLS = [
    "trip_miles",
    "base_passenger_fare",
    "tolls",
    "bcf",
    "sales_tax",
    "congestion_surcharge",
    "airport_fee",
    "tips",
    "driver_pay",
    "cbd_congestion_fee",
]

# -----------------------------
# Reglas de plausibilidad fuerte
# -----------------------------
MAX_TRIP_DURATION_MIN = 360.0   # 6 horas
MAX_REASONABLE_SPEED_MPH = 100.0

# Warning, no invalid directo
EXTREME_BASE_FARE = 400.0
EXTREME_DRIVER_PAY = 400.0

# Tolerancia entre trip_time (segundos reportados por dataset)
# y duración derivada de pickup/dropoff.
TRIP_TIME_TOLERANCE_SEC = 300  # 5 minutos


# -----------------------------
# Helpers
# -----------------------------
def _lower_map(cols: List[str]) -> Dict[str, str]:
    return {c.lower(): c for c in cols}


def _resolve_col(df: pd.DataFrame, canonical: str, aliases: Optional[List[str]] = None) -> Optional[str]:
    """Devuelve el nombre real de la columna (case-insensitive), o None si no existe."""
    aliases = aliases or []
    if canonical in df.columns:
        return canonical

    lm = _lower_map(list(df.columns))
    if canonical.lower() in lm:
        return lm[canonical.lower()]

    for a in aliases:
        if a in df.columns:
            return a
        if a.lower() in lm:
            return lm[a.lower()]

    return None


def standardize_fhvhv_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Renombra columnas encontradas a los nombres canónicos de EXPECTED_COLUMNS.
    Si falta una columna esperada, se crea como NA para mantener esquema consistente.
    """
    rename_map: Dict[str, str] = {}
    missing: List[str] = []

    aliases_by_canon = {
        "PULocationID": ["PUlocationID", "pulocationid"],
        "DOLocationID": ["DOlocationID", "dolocationid"],
        "pickup_datetime": ["pickup_datetime", "Pickup_datetime"],
        "dropoff_datetime": ["dropoff_datetime", "Dropoff_datetime"],
        "request_datetime": ["request_datetime", "Request_datetime"],
        "on_scene_datetime": ["on_scene_datetime", "On_scene_datetime"],
        "hvfhs_license_num": ["hvfhs_license_num", "HVFHS_license_num"],
        "dispatching_base_num": ["dispatching_base_num", "Dispatching_base_num"],
        "originating_base_num": ["originating_base_num", "Originating_base_num"],
        "trip_miles": ["trip_miles", "Trip_miles"],
        "trip_time": ["trip_time", "Trip_time"],
        "base_passenger_fare": ["base_passenger_fare", "Base_passenger_fare"],
        "congestion_surcharge": ["congestion_surcharge", "Congestion_surcharge"],
        "cbd_congestion_fee": ["cbd_congestion_fee", "CBD_congestion_fee"],
    }

    for canon in EXPECTED_COLUMNS:
        real = _resolve_col(df, canon, aliases=aliases_by_canon.get(canon, []))
        if real is None:
            missing.append(canon)
        else:
            if real != canon:
                rename_map[real] = canon

    if rename_map:
        df = df.rename(columns=rename_map)

    for canon in missing:
        df[canon] = pd.NA

    return df, rename_map, missing


def _to_datetime(s: pd.Series) -> Tuple[pd.Series, pd.Series]:
    orig_notna = s.notna()
    coerced = pd.to_datetime(s, errors="coerce")
    invalid = orig_notna & coerced.isna()
    return coerced, invalid


def _to_int_nullable(s: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    orig_notna = s.notna()
    num = pd.to_numeric(s, errors="coerce")
    invalid_type = orig_notna & num.isna()
    invalid_decimal = num.notna() & ((num % 1) != 0)
    num = num.where(~invalid_decimal)
    return num.astype("Int64"), invalid_type, invalid_decimal


def _to_float_nullable(s: pd.Series) -> Tuple[pd.Series, pd.Series]:
    orig_notna = s.notna()
    num = pd.to_numeric(s, errors="coerce")
    invalid_type = orig_notna & num.isna()
    return num.astype("Float64"), invalid_type


def _normalize_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.upper()


def _normalize_base_num(s: pd.Series) -> Tuple[pd.Series, pd.Series]:
    norm = _normalize_str(s)
    invalid = norm.notna() & ~norm.str.match(BASE_NUM_PATTERN)
    return norm, invalid


def _build_reason_column(masks: Dict[str, pd.Series], index: pd.Index) -> pd.Series:
    """
    Construye una columna string con las razones activadas por fila.
    Ejemplo:
        'pickup_datetime__future;trip_miles__negative'
    """
    reasons = pd.Series("", index=index, dtype="string")

    for name, mask in masks.items():
        mask_filled = mask.fillna(False)
        reasons = reasons.where(~mask_filled, reasons + name + ";")

    return reasons.str.rstrip(";").replace("", pd.NA)


def discover_parquet_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted([p for p in input_path.rglob("*.parquet") if p.is_file()])


def extract_expected_year_month_from_filename(path: Path) -> Tuple[Optional[int], Optional[int]]:
    """
    Extrae YYYY-MM del nombre tipo:
        fhvhv_tripdata_2023-07.parquet
    Si no puede extraerlo, devuelve (None, None).
    """
    m = re.search(r"(\d{4})-(\d{2})", path.stem)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def load_valid_location_ids(location_csv: Optional[Path]) -> Optional[Set[int]]:
    """
    Carga IDs válidos de zonas TLC desde un CSV, si se proporciona.
    Espera una columna LocationID.
    Si no hay CSV, devuelve None y no se aplica esta validación.
    """
    if location_csv is None:
        return None
    if not location_csv.exists():
        raise FileNotFoundError(f"No existe el catálogo de zonas TLC: {location_csv}")

    zones = pd.read_csv(location_csv)
    if "LocationID" not in zones.columns:
        raise ValueError(f"El CSV de zonas TLC no contiene columna 'LocationID': {location_csv}")

    ids = pd.to_numeric(zones["LocationID"], errors="coerce").dropna().astype(int)
    return set(ids.tolist())


@dataclass
class ValidationResult:
    df_clean: pd.DataFrame
    df_bad: pd.DataFrame
    report: Dict[str, Any]


def validate_fhvhv_df(
    df: pd.DataFrame,
    strict_columns: bool = False,
    source_file: Optional[Path] = None,
    valid_location_ids: Optional[Set[int]] = None,
) -> ValidationResult:
    report: Dict[str, Any] = {"n_rows": int(len(df))}

    # 1) Estandarizar columnas
    df, renamed, missing = standardize_fhvhv_columns(df)
    report["columns_missing"] = missing
    report["columns_renamed"] = renamed
    report["columns_unexpected"] = [c for c in df.columns if c not in set(EXPECTED_COLUMNS)]

    if strict_columns and missing:
        raise ValueError(f"Faltan columnas esperadas según diccionario HVFHV: {missing}")

    invalid_masks: Dict[str, pd.Series] = {}
    warning_masks: Dict[str, pd.Series] = {}

    # 2) Required mínimo
    for c in ["hvfhs_license_num", "pickup_datetime", "dropoff_datetime", "PULocationID", "DOLocationID"]:
        invalid_masks[f"{c}__missing_required"] = df[c].isna()

    # 3) hvfhs_license_num
    df["hvfhs_license_num"] = _normalize_str(df["hvfhs_license_num"])
    invalid_masks["hvfhs_license_num__invalid_format"] = (
        df["hvfhs_license_num"].notna() & ~df["hvfhs_license_num"].str.match(HVFHS_PATTERN)
    )
    warning_masks["hvfhs_license_num__unknown_code"] = (
        df["hvfhs_license_num"].notna()
        & df["hvfhs_license_num"].str.match(HVFHS_PATTERN)
        & ~df["hvfhs_license_num"].isin(KNOWN_HVFHS)
    )

    # 4) Base numbers (warning)
    df["dispatching_base_num"], inv_disp_fmt = _normalize_base_num(df["dispatching_base_num"])
    df["originating_base_num"], inv_orig_fmt = _normalize_base_num(df["originating_base_num"])
    warning_masks["dispatching_base_num__missing"] = df["dispatching_base_num"].isna()
    warning_masks["originating_base_num__missing"] = df["originating_base_num"].isna()
    warning_masks["dispatching_base_num__invalid_format"] = inv_disp_fmt
    warning_masks["originating_base_num__invalid_format"] = inv_orig_fmt

    # 5) Datetimes
    for c in ["request_datetime", "on_scene_datetime", "pickup_datetime", "dropoff_datetime"]:
        df[c], inv = _to_datetime(df[c])
        if c in ("pickup_datetime", "dropoff_datetime"):
            invalid_masks[f"{c}__invalid_datetime"] = inv
        else:
            warning_masks[f"{c}__invalid_datetime"] = inv

    pickup = df["pickup_datetime"]
    drop = df["dropoff_datetime"]
    req = df["request_datetime"]
    on_scene = df["on_scene_datetime"]

    # 6) Coherencia temporal fuerte
    invalid_masks["datetime__dropoff_before_pickup"] = pickup.notna() & drop.notna() & (drop < pickup)

    # Fechas futuras
    now_ts = pd.Timestamp.now()
    invalid_masks["pickup_datetime__future"] = pickup.notna() & (pickup > now_ts)
    invalid_masks["dropoff_datetime__future"] = drop.notna() & (drop > now_ts)
    warning_masks["request_datetime__future"] = req.notna() & (req > now_ts)
    warning_masks["on_scene_datetime__future"] = on_scene.notna() & (on_scene > now_ts)

    # Validación frente al mes esperado del fichero
    exp_year, exp_month = (None, None)
    if source_file is not None:
        exp_year, exp_month = extract_expected_year_month_from_filename(source_file)

    if exp_year is not None and exp_month is not None:
        invalid_masks["pickup_datetime__outside_expected_file_month"] = (
            pickup.notna()
            & ((pickup.dt.year != exp_year) | (pickup.dt.month != exp_month))
        )

        month_start = pd.Timestamp(year=exp_year, month=exp_month, day=1)
        next_month_start = month_start + pd.offsets.MonthBegin(1)
        month_end_last_instant = next_month_start - pd.Timedelta(seconds=1)
        dropoff_latest_allowed = month_end_last_instant + pd.Timedelta(minutes=MAX_TRIP_DURATION_MIN)

        invalid_masks["dropoff_datetime__too_far_beyond_expected_month"] = (
            drop.notna() & (drop > dropoff_latest_allowed)
        )

    # Warnings temporales suaves
    warning_masks["datetime__request_after_pickup"] = req.notna() & pickup.notna() & (req > pickup)
    warning_masks["datetime__on_scene_after_pickup"] = on_scene.notna() & pickup.notna() & (on_scene > pickup)
    warning_masks["datetime__on_scene_before_request"] = on_scene.notna() & req.notna() & (on_scene < req)

    # 7) Location IDs + trip_time
    df["PULocationID"], inv_pu_type, inv_pu_dec = _to_int_nullable(df["PULocationID"])
    df["DOLocationID"], inv_do_type, inv_do_dec = _to_int_nullable(df["DOLocationID"])
    invalid_masks["PULocationID__invalid_type"] = inv_pu_type
    invalid_masks["PULocationID__invalid_decimal"] = inv_pu_dec
    invalid_masks["DOLocationID__invalid_type"] = inv_do_type
    invalid_masks["DOLocationID__invalid_decimal"] = inv_do_dec

    if valid_location_ids is not None:
        invalid_masks["PULocationID__unknown_location_id"] = (
            df["PULocationID"].notna() & ~df["PULocationID"].isin(valid_location_ids)
        )
        invalid_masks["DOLocationID__unknown_location_id"] = (
            df["DOLocationID"].notna() & ~df["DOLocationID"].isin(valid_location_ids)
        )

    df["trip_time"], inv_tt_type, inv_tt_dec = _to_int_nullable(df["trip_time"])
    invalid_masks["trip_time__invalid_type"] = inv_tt_type
    invalid_masks["trip_time__invalid_decimal"] = inv_tt_dec
    invalid_masks["trip_time__negative"] = df["trip_time"].notna() & (df["trip_time"] < 0)

    # 8) Floats
    for c in FLOAT_COLS:
        df[c], inv = _to_float_nullable(df[c])
        invalid_masks[f"{c}__invalid_type"] = inv

    invalid_masks["trip_miles__negative"] = df["trip_miles"].notna() & (df["trip_miles"] < 0)

    # Importes negativos: warning, no invalid directo
    money_cols = [
        "base_passenger_fare",
        "tolls",
        "bcf",
        "sales_tax",
        "congestion_surcharge",
        "airport_fee",
        "tips",
        "driver_pay",
        "cbd_congestion_fee",
    ]
    for c in money_cols:
        warning_masks[f"{c}__negative"] = df[c].notna() & (df[c] < 0)

    # Warning de importes extremos
    warning_masks["base_passenger_fare__extreme"] = (
        df["base_passenger_fare"].notna() & (df["base_passenger_fare"] > EXTREME_BASE_FARE)
    )
    warning_masks["driver_pay__extreme"] = (
        df["driver_pay"].notna() & (df["driver_pay"] > EXTREME_DRIVER_PAY)
    )

    # 9) Flags Y/N
    for c in YN_FLAGS:
        s = _normalize_str(df[c])
        df[c] = s
        warning_masks[f"{c}__missing"] = s.isna()
        invalid_masks[f"{c}__out_of_domain"] = s.notna() & ~s.isin(ALLOWED_YN)

    # 10) cbd_congestion_fee missing
    warning_masks["cbd_congestion_fee__missing"] = df["cbd_congestion_fee"].isna()

    # 11) Duración y velocidad implícita
    trip_duration_min = (drop - pickup).dt.total_seconds() / 60.0
    df["trip_duration_min"] = trip_duration_min.astype("Float64")

    invalid_masks["trip_duration_min__negative"] = trip_duration_min.notna() & (trip_duration_min < 0)
    invalid_masks["trip_duration_min__too_long"] = trip_duration_min.notna() & (trip_duration_min > MAX_TRIP_DURATION_MIN)
    warning_masks["trip_duration_min__zero"] = trip_duration_min.notna() & (trip_duration_min == 0)

    duration_hours = trip_duration_min / 60.0
    implied_speed_mph = pd.Series(pd.NA, index=df.index, dtype="Float64")
    valid_speed_base = (
        df["trip_miles"].notna()
        & duration_hours.notna()
        & (df["trip_miles"] > 0)
        & (duration_hours > 0)
    )
    implied_speed_mph.loc[valid_speed_base] = (
        df.loc[valid_speed_base, "trip_miles"] / duration_hours.loc[valid_speed_base]
    ).astype("Float64")
    df["implied_speed_mph"] = implied_speed_mph

    invalid_masks["implied_speed_mph__too_high"] = (
        df["implied_speed_mph"].notna() & (df["implied_speed_mph"] > MAX_REASONABLE_SPEED_MPH)
    )

    # 12) Coherencia entre trip_time reportado y duración derivada
    # trip_time suele venir en segundos en este dataset.
    derived_trip_time_sec = (drop - pickup).dt.total_seconds()
    trip_time_diff_sec = (df["trip_time"] - derived_trip_time_sec).abs()
    warning_masks["trip_time__mismatch_vs_datetimes"] = (
        df["trip_time"].notna()
        & derived_trip_time_sec.notna()
        & (trip_time_diff_sec > TRIP_TIME_TOLERANCE_SEC)
    )

    # 13) Duplicados exactos
    invalid_masks["row__duplicate_exact"] = df.duplicated(keep="first")

    # 14) Report + razones
    report["invalid_counts"] = {k: int(v.fillna(False).sum()) for k, v in invalid_masks.items()}
    report["warning_counts"] = {k: int(v.fillna(False).sum()) for k, v in warning_masks.items()}

    any_invalid = pd.Series(False, index=df.index)
    for v in invalid_masks.values():
        any_invalid |= v.fillna(False)

    df = df.copy()
    df["warning_reasons"] = _build_reason_column(warning_masks, df.index)

    df_bad = df.loc[any_invalid].copy()
    df_bad["rejection_reasons"] = _build_reason_column(invalid_masks, df_bad.index)

    df_clean = df.loc[~any_invalid].copy()

    report["n_bad_rows"] = int(len(df_bad))
    report["n_clean_rows"] = int(len(df_clean))
    report["n_warning_rows_in_clean"] = int(df_clean["warning_reasons"].notna().sum())

    # Reordenar columnas a esquema final conservando trazabilidad adicional
    final_cols = EXPECTED_COLUMNS + ["trip_duration_min", "implied_speed_mph", "warning_reasons"]
    bad_cols = EXPECTED_COLUMNS + ["trip_duration_min", "implied_speed_mph", "warning_reasons", "rejection_reasons"]

    df_clean = df_clean.reindex(columns=[c for c in final_cols if c in df_clean.columns])
    df_bad = df_bad.reindex(columns=[c for c in bad_cols if c in df_bad.columns])

    return ValidationResult(df_clean=df_clean, df_bad=df_bad, report=report)


def _accumulate_counts(dst: Dict[str, int], src: Dict[str, int]) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + int(v)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capa1 FHVHV: validación estructural + plausibilidad fuerte según data dictionary HVFHS.")

    p.add_argument("--service-folder", type=str, default="fhvhv", help="Subcarpeta dentro de data/raw. Default: fhvhv")
    p.add_argument("--prefix", type=str, default="fhvhv_tripdata_", help="Prefijo. Default: fhvhv_tripdata_")
    p.add_argument("--input", type=str, default="", help="Ruta parquet o directorio. Si vacío usa data/raw/<service-folder>.")
    p.add_argument("--output", type=str, default="", help="Salida. Si vacío usa data/validated/<service-folder>.")
    p.add_argument("--months", nargs="*", default=[], help="Meses YYYY-MM para construir <prefix><YYYY-MM>.parquet")
    p.add_argument("--strict-columns", action="store_true", help="Falla si faltan columnas del diccionario.")
    p.add_argument("--write-bad", action="store_true", help="Escribe bad_rows.")
    p.add_argument("--limit-files", type=int, default=0, help="Limita nº ficheros (debug).")
    p.add_argument(
        "--taxi-zones-csv",
        type=str,
        default="",
        help="Ruta a taxi_zone_lookup.csv para validar LocationID reales. Si vacío, se omite esta validación.",
    )

    # batch-size
    p.add_argument(
        "--batch-size",
        type=int,
        default=200_000,
        help="Filas por batch (evita reventar RAM). Default: 200000",
    )

    # liberar memoria más agresivo
    p.add_argument(
        "--gc-every",
        type=int,
        default=10,
        help="Ejecuta gc.collect() cada N batches (0 = nunca). Default: 10",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    base_raw = obtener_ruta("data/raw") / args.service_folder
    base_out = obtener_ruta("data/validated") / args.service_folder
    out_reports = obtener_ruta("outputs/procesamiento") / "capa1_fhvhv"

    input_path = Path(args.input) if args.input else base_raw
    output_dir = Path(args.output) if args.output else base_out

    location_csv = Path(args.taxi_zones_csv) if args.taxi_zones_csv else None
    valid_location_ids = load_valid_location_ids(location_csv)

    # Resolver ficheros
    if args.months:
        files: List[Path] = []
        for m in args.months:
            filename = f"{args.prefix}{m}.parquet"
            fp = base_raw / filename
            if fp.exists():
                files.append(fp)
            else:
                console.print(f"[yellow]Aviso:[/yellow] No existe {fp}")
    else:
        files = discover_parquet_files(input_path)

    if args.limit_files and args.limit_files > 0:
        files = files[: args.limit_files]

    if not files:
        raise FileNotFoundError(f"No se encontraron parquet para input={input_path} (service-folder={args.service_folder})")

    console.print(Panel.fit("[bold cyan]CAPA 1 — FHVHV: VALIDACIÓN (STREAMING + PLAUSIBILIDAD)[/bold cyan]"))

    table = Table(title="Capa1 FHVHV — resumen por fichero", header_style="bold magenta")
    table.add_column("Fichero")
    table.add_column("Rows", justify="right")
    table.add_column("Clean", justify="right")
    table.add_column("Bad", justify="right")
    table.add_column("Warn(clean)", justify="right")

    summary: Dict[str, Any] = {
        "service_folder": args.service_folder,
        "prefix": args.prefix,
        "batch_size": args.batch_size,
        "n_files": len(files),
        "files": [],
        "totals": {"n_rows": 0, "n_clean_rows": 0, "n_bad_rows": 0, "n_warning_rows_in_clean": 0},
    }

    for f in files:
        clean_path = output_dir / "clean" / f"{f.stem}.parquet"
        bad_path = output_dir / "bad_rows" / f"{f.stem}.parquet"
        report_path = output_dir / "reports" / f"{f.stem}.validation_report.json"

        clean_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if args.write_bad:
            bad_path.parent.mkdir(parents=True, exist_ok=True)

        if clean_path.exists():
            clean_path.unlink()
        if args.write_bad and bad_path.exists():
            bad_path.unlink()

        file_totals: Dict[str, Any] = {
            "n_rows": 0,
            "n_clean_rows": 0,
            "n_bad_rows": 0,
            "n_warning_rows_in_clean": 0,
            "invalid_counts": {},
            "warning_counts": {},
            "columns_missing": [],
            "columns_renamed": {},
            "columns_unexpected": [],
        }

        clean_writer: Optional[pq.ParquetWriter] = None
        bad_writer: Optional[pq.ParquetWriter] = None

        parquet_file = pq.ParquetFile(f)

        # Leer columnas esperadas si existen y mantener extras fuera.
        # Como luego estandarizamos y añadimos faltantes, esto es suficiente.
        schema_cols = set(parquet_file.schema.names)
        cols_to_read = [c for c in EXPECTED_COLUMNS if c in schema_cols]

        with console.status(f"[cyan]Validando {f.name} (batch_size={args.batch_size})..."):
            for i, batch in enumerate(parquet_file.iter_batches(batch_size=args.batch_size, columns=cols_to_read), start=1):
                df_batch = batch.to_pandas(
                    self_destruct=True,
                    split_blocks=True,
                    strings_to_categorical=True,
                )

                res = validate_fhvhv_df(
                    df_batch,
                    strict_columns=args.strict_columns,
                    source_file=f,
                    valid_location_ids=valid_location_ids,
                )

                file_totals["n_rows"] += res.report["n_rows"]
                file_totals["n_clean_rows"] += res.report["n_clean_rows"]
                file_totals["n_bad_rows"] += res.report["n_bad_rows"]
                file_totals["n_warning_rows_in_clean"] += res.report["n_warning_rows_in_clean"]

                _accumulate_counts(file_totals["invalid_counts"], res.report.get("invalid_counts", {}))
                _accumulate_counts(file_totals["warning_counts"], res.report.get("warning_counts", {}))

                if not file_totals["columns_missing"]:
                    file_totals["columns_missing"] = res.report.get("columns_missing", [])
                if not file_totals["columns_renamed"]:
                    file_totals["columns_renamed"] = res.report.get("columns_renamed", {})
                if not file_totals["columns_unexpected"]:
                    file_totals["columns_unexpected"] = res.report.get("columns_unexpected", [])

                if clean_writer is None and len(res.df_clean) > 0:
                    t0 = pa.Table.from_pandas(res.df_clean, preserve_index=False)
                    clean_writer = pq.ParquetWriter(clean_path.as_posix(), t0.schema)

                if args.write_bad and bad_writer is None and len(res.df_bad) > 0:
                    tb0 = pa.Table.from_pandas(res.df_bad, preserve_index=False)
                    bad_writer = pq.ParquetWriter(bad_path.as_posix(), tb0.schema)

                if clean_writer is not None and len(res.df_clean) > 0:
                    t = pa.Table.from_pandas(res.df_clean, preserve_index=False)
                    clean_writer.write_table(t)

                if args.write_bad and bad_writer is not None and len(res.df_bad) > 0:
                    tb = pa.Table.from_pandas(res.df_bad, preserve_index=False)
                    bad_writer.write_table(tb)

                del df_batch, res, batch
                if args.gc_every and args.gc_every > 0 and (i % args.gc_every == 0):
                    gc.collect()

        if clean_writer is not None:
            clean_writer.close()
        if bad_writer is not None:
            bad_writer.close()

        file_report = {
            "n_rows": int(file_totals["n_rows"]),
            "n_clean_rows": int(file_totals["n_clean_rows"]),
            "n_bad_rows": int(file_totals["n_bad_rows"]),
            "n_warning_rows_in_clean": int(file_totals["n_warning_rows_in_clean"]),
            "invalid_counts": file_totals["invalid_counts"],
            "warning_counts": file_totals["warning_counts"],
            "columns_missing": file_totals["columns_missing"],
            "columns_renamed": file_totals["columns_renamed"],
            "columns_unexpected": file_totals["columns_unexpected"],
        }
        with report_path.open("w", encoding="utf-8") as fp:
            json.dump(file_report, fp, ensure_ascii=False, indent=2)

        table.add_row(
            f.name,
            str(file_report["n_rows"]),
            str(file_report["n_clean_rows"]),
            str(file_report["n_bad_rows"]),
            str(file_report["n_warning_rows_in_clean"]),
        )

        entry = {"file": str(f), **file_report}
        summary["files"].append(entry)
        summary["totals"]["n_rows"] += file_report["n_rows"]
        summary["totals"]["n_clean_rows"] += file_report["n_clean_rows"]
        summary["totals"]["n_bad_rows"] += file_report["n_bad_rows"]
        summary["totals"]["n_warning_rows_in_clean"] += file_report["n_warning_rows_in_clean"]

    console.print(table)

    out_reports.mkdir(parents=True, exist_ok=True)
    summary_path = out_reports / "capa1_fhvhv_validation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    console.print(f"\n[bold green]OK[/bold green] Resumen global: {summary_path}")
    console.print(
        f"[green]Totales[/green] rows={summary['totals']['n_rows']} | clean={summary['totals']['n_clean_rows']} | "
        f"bad={summary['totals']['n_bad_rows']} | warn(clean)={summary['totals']['n_warning_rows_in_clean']}"
    )
    console.print(f"[green]Salida datos[/green] {output_dir}/clean (y bad_rows si activado)")


if __name__ == "__main__":
    main()