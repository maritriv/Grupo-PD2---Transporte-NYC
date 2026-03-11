# src/procesamiento/capa1/capa1_green.py

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import pyarrow.parquet as pq
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import obtener_ruta


console = Console()

# -----------------------------
# Especificación Green (LPEP) basada en el Data Dictionary (Mar 18, 2025)
# -----------------------------
EXPECTED_COLUMNS: List[str] = [
    "VendorID",
    "lpep_pickup_datetime",
    "lpep_dropoff_datetime",
    "store_and_fwd_flag",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "payment_type",
    "trip_type",
    "congestion_surcharge",
    "cbd_congestion_fee",  # puede no existir en meses antiguos
]

OPTIONAL_COLUMNS = {"cbd_congestion_fee"}

ALLOWED_VENDOR_ID = {1, 2, 6}
ALLOWED_RATECODE_ID = {1, 2, 3, 4, 5, 6, 99}
ALLOWED_PAYMENT_TYPE = {0, 1, 2, 3, 4, 5, 6}
ALLOWED_TRIP_TYPE = {1, 2}
ALLOWED_STORE_AND_FWD = {"Y", "N"}

# -----------------------------
# Reglas de plausibilidad fuerte
# -----------------------------
# Igual que en Yellow, pero aplicadas a Green.
MAX_TRIP_DURATION_MIN = 360.0  # 6 horas
MAX_REASONABLE_SPEED_MPH = 100.0

# Warnings de importes extremos
EXTREME_TOTAL_AMOUNT = 500.0
EXTREME_FARE_AMOUNT = 400.0

# Tolerancia para comparar componentes vs total
TOTAL_AMOUNT_TOLERANCE = 3.0


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


def standardize_green_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Renombra columnas encontradas (case-insensitive + aliases) a nombres canónicos.
    Si falta una columna esperada, la crea como NA para mantener esquema consistente.
    """
    rename_map: Dict[str, str] = {}
    missing: List[str] = []

    aliases_by_canon = {
        "VendorID": ["vendorid"],
        "lpep_pickup_datetime": ["pickup_datetime", "lpep_pickup_datetime"],
        "lpep_dropoff_datetime": ["dropoff_datetime", "lpep_dropoff_datetime"],
        "RatecodeID": ["ratecodeid", "RateCodeID"],
        "store_and_fwd_flag": ["store_and_fwd_flag", "store_and_forward", "store_and_fwd"],
        "PULocationID": ["pulocationid", "PULocationID"],
        "DOLocationID": ["dolocationid", "DOLocationID"],
        "payment_type": ["payment_type", "Payment_type"],
        "trip_type": ["trip_type", "Trip_type"],
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


def _to_float_nullable(s: pd.Series) -> Tuple[pd.Series, pd.Series]:
    orig_notna = s.notna()
    num = pd.to_numeric(s, errors="coerce")
    invalid_type = orig_notna & num.isna()
    return num.astype("Float64"), invalid_type


def _to_int_nullable(s: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Devuelve (int_series, invalid_type_mask, invalid_decimal_mask)
    invalid_type: no nulo y no convertible
    invalid_decimal: convertible pero con decimales
    """
    orig_notna = s.notna()
    num = pd.to_numeric(s, errors="coerce")
    invalid_type = orig_notna & num.isna()

    invalid_decimal = num.notna() & ((num % 1) != 0)
    num = num.where(~invalid_decimal)

    return num.astype("Int64"), invalid_type, invalid_decimal


def _build_reason_column(masks: Dict[str, pd.Series], index: pd.Index) -> pd.Series:
    """
    Construye una columna string con las razones activadas por fila.
    Ejemplo:
        'lpep_pickup_datetime__future;trip_distance__negative'
    """
    reasons = pd.Series("", index=index, dtype="string")

    for name, mask in masks.items():
        mask_filled = mask.fillna(False)
        reasons = reasons.where(~mask_filled, reasons + name + ";")

    return reasons.str.rstrip(";").replace("", pd.NA)


def _invalid_out_of_domain(s: pd.Series, allowed: set) -> pd.Series:
    return s.notna() & ~s.isin(allowed)


def discover_parquet_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted([p for p in input_path.rglob("*.parquet") if p.is_file()])


def read_parquet_safely(filepath: Path) -> pd.DataFrame:
    """
    Lee columnas relevantes si existen, manteniendo extras para reportarlos.
    """
    pf = pq.ParquetFile(filepath)
    cols = pf.schema.names
    cols_to_read = [c for c in EXPECTED_COLUMNS if c in cols] + [c for c in cols if c not in EXPECTED_COLUMNS]
    table = pf.read(columns=[c for c in cols_to_read if c in cols])
    return table.to_pandas()


def extract_expected_year_month_from_filename(path: Path) -> Tuple[Optional[int], Optional[int]]:
    """
    Extrae YYYY-MM del nombre tipo:
        green_tripdata_2020-01.parquet
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


def validate_green_df(
    df: pd.DataFrame,
    strict_columns: bool = False,
    source_file: Optional[Path] = None,
    valid_location_ids: Optional[Set[int]] = None,
) -> ValidationResult:
    report: Dict[str, Any] = {"n_rows": int(len(df))}

    # 1) Estandarizar columnas a canónicas
    df, renamed, missing = standardize_green_columns(df)
    report["columns_missing"] = missing
    report["columns_renamed"] = renamed
    report["columns_unexpected"] = [c for c in df.columns if c not in set(EXPECTED_COLUMNS)]

    mandatory_missing = [c for c in missing if c not in OPTIONAL_COLUMNS]
    if strict_columns and mandatory_missing:
        raise ValueError(f"Faltan columnas obligatorias: {mandatory_missing}")

    invalid_masks: Dict[str, pd.Series] = {}
    warning_masks: Dict[str, pd.Series] = {}

    # -----------------------------
    # 2) Required mínimo para análisis
    # -----------------------------
    for c in ["VendorID", "lpep_pickup_datetime", "lpep_dropoff_datetime", "PULocationID", "DOLocationID"]:
        invalid_masks[f"{c}__missing_required"] = df[c].isna()

    # -----------------------------
    # 3) Datetimes + coherencia temporal
    # -----------------------------
    df["lpep_pickup_datetime"], inv_pickup_dt = _to_datetime(df["lpep_pickup_datetime"])
    df["lpep_dropoff_datetime"], inv_dropoff_dt = _to_datetime(df["lpep_dropoff_datetime"])
    invalid_masks["lpep_pickup_datetime__invalid_datetime"] = inv_pickup_dt
    invalid_masks["lpep_dropoff_datetime__invalid_datetime"] = inv_dropoff_dt

    pickup = df["lpep_pickup_datetime"]
    dropoff = df["lpep_dropoff_datetime"]

    invalid_masks["datetime__dropoff_before_pickup"] = pickup.notna() & dropoff.notna() & (dropoff < pickup)

    # Fechas futuras
    now_ts = pd.Timestamp.now()
    invalid_masks["lpep_pickup_datetime__future"] = pickup.notna() & (pickup > now_ts)
    invalid_masks["lpep_dropoff_datetime__future"] = dropoff.notna() & (dropoff > now_ts)

    # Validación frente al mes esperado del fichero
    exp_year, exp_month = (None, None)
    if source_file is not None:
        exp_year, exp_month = extract_expected_year_month_from_filename(source_file)

    if exp_year is not None and exp_month is not None:
        # El pickup sí debe caer dentro del mes exacto del fichero.
        invalid_masks["lpep_pickup_datetime__outside_expected_file_month"] = (
            pickup.notna()
            & ((pickup.dt.year != exp_year) | (pickup.dt.month != exp_month))
        )

        # Último instante del mes del fichero.
        month_start = pd.Timestamp(year=exp_year, month=exp_month, day=1)
        next_month_start = month_start + pd.offsets.MonthBegin(1)
        month_end_last_instant = next_month_start - pd.Timedelta(seconds=1)

        # Permitimos que el dropoff se desborde como mucho la duración máxima.
        dropoff_latest_allowed = month_end_last_instant + pd.Timedelta(minutes=MAX_TRIP_DURATION_MIN)

        invalid_masks["lpep_dropoff_datetime__too_far_beyond_expected_month"] = (
            dropoff.notna() & (dropoff > dropoff_latest_allowed)
        )

    # -----------------------------
    # 4) Integers (tipo) + dominios
    # -----------------------------
    int_cols = [
        "VendorID",
        "RatecodeID",
        "PULocationID",
        "DOLocationID",
        "passenger_count",
        "payment_type",
        "trip_type",
    ]
    for c in int_cols:
        df[c], inv_type, inv_dec = _to_int_nullable(df[c])
        invalid_masks[f"{c}__invalid_type"] = inv_type
        invalid_masks[f"{c}__invalid_decimal"] = inv_dec

    invalid_masks["VendorID__out_of_domain"] = _invalid_out_of_domain(df["VendorID"], ALLOWED_VENDOR_ID)
    invalid_masks["RatecodeID__out_of_domain"] = _invalid_out_of_domain(df["RatecodeID"], ALLOWED_RATECODE_ID)
    invalid_masks["payment_type__out_of_domain"] = _invalid_out_of_domain(df["payment_type"], ALLOWED_PAYMENT_TYPE)
    invalid_masks["trip_type__out_of_domain"] = _invalid_out_of_domain(df["trip_type"], ALLOWED_TRIP_TYPE)

    # passenger_count
    invalid_masks["passenger_count__negative"] = df["passenger_count"].notna() & (df["passenger_count"] < 0)
    warning_masks["passenger_count__zero"] = df["passenger_count"].notna() & (df["passenger_count"] == 0)

    # Location IDs reales si hay catálogo
    if valid_location_ids is not None:
        invalid_masks["PULocationID__unknown_location_id"] = (
            df["PULocationID"].notna() & ~df["PULocationID"].isin(valid_location_ids)
        )
        invalid_masks["DOLocationID__unknown_location_id"] = (
            df["DOLocationID"].notna() & ~df["DOLocationID"].isin(valid_location_ids)
        )

    # -----------------------------
    # 5) store_and_fwd_flag (Y/N)
    # -----------------------------
    df["store_and_fwd_flag"] = df["store_and_fwd_flag"].astype("string").str.strip().str.upper()
    warning_masks["store_and_fwd_flag__missing"] = df["store_and_fwd_flag"].isna()
    invalid_masks["store_and_fwd_flag__out_of_domain"] = (
        df["store_and_fwd_flag"].notna() & ~df["store_and_fwd_flag"].isin(ALLOWED_STORE_AND_FWD)
    )

    # -----------------------------
    # 6) Floats (tipo)
    # -----------------------------
    float_cols = [
        "trip_distance",
        "fare_amount",
        "extra",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "improvement_surcharge",
        "total_amount",
        "congestion_surcharge",
        "cbd_congestion_fee",
    ]
    for c in float_cols:
        df[c], inv = _to_float_nullable(df[c])
        invalid_masks[f"{c}__invalid_numeric"] = inv

    # -----------------------------
    # 7) Plausibilidad fuerte numérica
    # -----------------------------
    invalid_masks["trip_distance__negative"] = df["trip_distance"].notna() & (df["trip_distance"] < 0)

    # Estas tasas/surcharges no deberían ser negativas.
    invalid_masks["mta_tax__negative"] = df["mta_tax"].notna() & (df["mta_tax"] < 0)
    invalid_masks["improvement_surcharge__negative"] = (
        df["improvement_surcharge"].notna() & (df["improvement_surcharge"] < 0)
    )
    invalid_masks["congestion_surcharge__negative"] = (
        df["congestion_surcharge"].notna() & (df["congestion_surcharge"] < 0)
    )
    invalid_masks["cbd_congestion_fee__negative"] = (
        df["cbd_congestion_fee"].notna() & (df["cbd_congestion_fee"] < 0)
    )

    # Warnings suaves
    warning_masks["trip_distance__zero"] = df["trip_distance"].notna() & (df["trip_distance"] == 0)
    warning_masks["fare_amount__extreme"] = df["fare_amount"].notna() & (df["fare_amount"] > EXTREME_FARE_AMOUNT)
    warning_masks["total_amount__extreme"] = df["total_amount"].notna() & (df["total_amount"] > EXTREME_TOTAL_AMOUNT)

    # -----------------------------
    # 8) Duración y velocidad implícita
    # -----------------------------
    trip_duration_min = (dropoff - pickup).dt.total_seconds() / 60.0
    df["trip_duration_min"] = trip_duration_min.astype("Float64")

    invalid_masks["trip_duration_min__negative"] = trip_duration_min.notna() & (trip_duration_min < 0)
    invalid_masks["trip_duration_min__too_long"] = (
        trip_duration_min.notna() & (trip_duration_min > MAX_TRIP_DURATION_MIN)
    )
    warning_masks["trip_duration_min__zero"] = trip_duration_min.notna() & (trip_duration_min == 0)

    duration_hours = trip_duration_min / 60.0
    implied_speed_mph = pd.Series(pd.NA, index=df.index, dtype="Float64")
    valid_speed_base = (
        df["trip_distance"].notna()
        & duration_hours.notna()
        & (df["trip_distance"] > 0)
        & (duration_hours > 0)
    )
    implied_speed_mph.loc[valid_speed_base] = (
        df.loc[valid_speed_base, "trip_distance"] / duration_hours.loc[valid_speed_base]
    ).astype("Float64")
    df["implied_speed_mph"] = implied_speed_mph

    invalid_masks["implied_speed_mph__too_high"] = (
        df["implied_speed_mph"].notna() & (df["implied_speed_mph"] > MAX_REASONABLE_SPEED_MPH)
    )

    # -----------------------------
    # 9) Coherencia contable suave
    # -----------------------------
    total_components = (
        df["fare_amount"].fillna(0)
        + df["extra"].fillna(0)
        + df["mta_tax"].fillna(0)
        + df["tip_amount"].fillna(0)
        + df["tolls_amount"].fillna(0)
        + df["improvement_surcharge"].fillna(0)
        + df["congestion_surcharge"].fillna(0)
        + df["cbd_congestion_fee"].fillna(0)
    )
    total_diff = (df["total_amount"] - total_components).abs()

    warning_masks["total_amount__component_mismatch"] = (
        df["total_amount"].notna() & (total_diff > TOTAL_AMOUNT_TOLERANCE)
    )

    # -----------------------------
    # 10) Duplicados exactos
    # -----------------------------
    invalid_masks["row__duplicate_exact"] = df.duplicated(keep="first")

    # -----------------------------
    # 11) Report + razones
    # -----------------------------
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

    return ValidationResult(df_clean=df_clean, df_bad=df_bad, report=report)


def write_outputs(
    result: ValidationResult,
    output_dir: Path,
    input_file: Path,
    write_bad: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_file.stem  # ej: green_tripdata_2020-01
    clean_path = output_dir / "clean" / f"{stem}.parquet"
    report_path = output_dir / "reports" / f"{stem}.validation_report.json"

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    result.df_clean.to_parquet(clean_path, index=False)

    if write_bad:
        bad_path = output_dir / "bad_rows" / f"{stem}.parquet"
        bad_path.parent.mkdir(parents=True, exist_ok=True)
        result.df_bad.to_parquet(bad_path, index=False)

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(result.report, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capa1 Green: validación estructural + plausibilidad fuerte (LPEP)."
    )

    p.add_argument(
        "--input",
        type=str,
        default="",
        help="Ruta a parquet o directorio. Si vacío, usa data/raw/green.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="Directorio salida. Si vacío, usa data/validated/green.",
    )
    p.add_argument(
        "--months",
        nargs="*",
        default=[],
        help="Lista de meses YYYY-MM para construir filenames green_tripdata_YYYY-MM.parquet (opcional).",
    )
    p.add_argument("--strict-columns", action="store_true", help="Falla si faltan columnas obligatorias.")
    p.add_argument("--write-bad", action="store_true", help="Escribe bad_rows.")
    p.add_argument("--limit-files", type=int, default=0, help="Limita nº ficheros (debug).")
    p.add_argument(
        "--taxi-zones-csv",
        type=str,
        default="",
        help="Ruta a taxi_zone_lookup.csv para validar LocationID reales. Si vacío, se omite esta validación.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    base_raw = obtener_ruta("data/raw") / "green"
    base_out = obtener_ruta("data/validated") / "green"
    out_reports = obtener_ruta("outputs/procesamiento") / "capa1_green"

    input_path = Path(args.input) if args.input else base_raw
    output_dir = Path(args.output) if args.output else base_out

    location_csv = Path(args.taxi_zones_csv) if args.taxi_zones_csv else None
    valid_location_ids = load_valid_location_ids(location_csv)

    # Resolver ficheros
    if args.months:
        files = []
        for m in args.months:
            filename = f"green_tripdata_{m}.parquet"
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
        raise FileNotFoundError(f"No se encontraron parquet para input={input_path}")

    console.print(Panel.fit("[bold cyan]CAPA 1 — GREEN: VALIDACIÓN (TIPOS + DOMINIOS + PLAUSIBILIDAD)[/bold cyan]"))

    table = Table(title="Capa1 Green — resumen por fichero", header_style="bold magenta")
    table.add_column("Fichero")
    table.add_column("Rows", justify="right")
    table.add_column("Clean", justify="right")
    table.add_column("Bad", justify="right")
    table.add_column("Warn(clean)", justify="right")

    summary = {
        "n_files": len(files),
        "files": [],
        "totals": {"n_rows": 0, "n_clean_rows": 0, "n_bad_rows": 0, "n_warning_rows_in_clean": 0},
    }

    for f in files:
        with console.status(f"[cyan]Validando {f.name}..."):
            df = read_parquet_safely(f)
            res = validate_green_df(
                df,
                strict_columns=args.strict_columns,
                source_file=f,
                valid_location_ids=valid_location_ids,
            )
            write_outputs(res, output_dir, f, write_bad=args.write_bad)

        table.add_row(
            f.name,
            str(res.report["n_rows"]),
            str(res.report["n_clean_rows"]),
            str(res.report["n_bad_rows"]),
            str(res.report["n_warning_rows_in_clean"]),
        )

        entry = {
            "file": str(f),
            **res.report,
        }
        summary["files"].append(entry)
        summary["totals"]["n_rows"] += res.report["n_rows"]
        summary["totals"]["n_clean_rows"] += res.report["n_clean_rows"]
        summary["totals"]["n_bad_rows"] += res.report["n_bad_rows"]
        summary["totals"]["n_warning_rows_in_clean"] += res.report["n_warning_rows_in_clean"]

    console.print(table)

    out_reports.mkdir(parents=True, exist_ok=True)
    summary_path = out_reports / "capa1_green_validation_summary.json"
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