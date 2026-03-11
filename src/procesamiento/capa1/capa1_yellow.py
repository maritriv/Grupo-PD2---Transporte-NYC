# src/procesamiento/capa1/capa1_yellow.py

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
from pandas.tseries.offsets import MonthEnd

from config.settings import obtener_ruta

console = Console()

# -----------------------------
# Especificación Yellow (TPEP) según Data Dictionary (Mar 18, 2025)
# -----------------------------
EXPECTED_COLUMNS: List[str] = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "airport_fee",
    "cbd_congestion_fee",
]

ALLOWED_VENDOR_ID = {1, 2, 6, 7}
ALLOWED_RATECODE_ID = {1, 2, 3, 4, 5, 6, 99}
ALLOWED_STORE_FLAG = {"Y", "N"}
ALLOWED_PAYMENT_TYPE = {0, 1, 2, 3, 4, 5, 6}

# Columnas "monetarias" o numéricas en float (tipo, no reglas de negocio)
FLOAT_COLS = [
    "trip_distance",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "airport_fee",
    "cbd_congestion_fee",
]

INT_COLS = [
    "VendorID",
    "RatecodeID",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
    "payment_type",
]

# -----------------------------
# Reglas de plausibilidad fuerte
# -----------------------------
# Duración máxima razonable para taxi Yellow urbano (en minutos).
# Si supera este umbral, asumimos que el registro no es fiable.
MAX_TRIP_DURATION_MIN = 360.0  # 6 horas

# Velocidad máxima razonable (millas por hora).
# Por encima de esto, muy probablemente hay error de distancia o tiempo.
MAX_REASONABLE_SPEED_MPH = 100.0

# Importe extremo: no se elimina, pero se marca como warning.
EXTREME_TOTAL_AMOUNT = 500.0
EXTREME_FARE_AMOUNT = 400.0

# Tolerancia para comparar suma de componentes vs total_amount.
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


def standardize_yellow_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Renombra columnas encontradas (case-insensitive + aliases) a nombres canónicos.
    Si falta una columna esperada, la crea como NA para mantener esquema consistente.
    """
    rename_map: Dict[str, str] = {}
    missing: List[str] = []

    # aliases útiles por si hay variaciones de mayúsculas o nombres históricos
    aliases_by_canon = {
        "VendorID": ["vendorid"],
        "tpep_pickup_datetime": ["pickup_datetime", "tpep_pickup_datetime"],
        "tpep_dropoff_datetime": ["dropoff_datetime", "tpep_dropoff_datetime"],
        "RatecodeID": ["ratecodeid", "RateCodeID"],
        "store_and_fwd_flag": ["store_and_fwd_flag", "store_and_forward", "store_and_fwd"],
        "PULocationID": ["pulocationid", "PULocationID"],
        "DOLocationID": ["dolocationid", "DOLocationID"],
        "payment_type": ["payment_type", "Payment_type"],
        "airport_fee": ["airport_fee", "Airport_fee"],
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


def discover_parquet_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted([p for p in input_path.rglob("*.parquet") if p.is_file()])


def extract_expected_year_month_from_filename(path: Path) -> Tuple[Optional[int], Optional[int]]:
    """
    Extrae YYYY-MM del nombre tipo:
        yellow_tripdata_2023-07.parquet
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
    Si no hay CSV, devuelve None y simplemente no se aplica esta validación.
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


def _build_reason_column(masks: Dict[str, pd.Series], index: pd.Index) -> pd.Series:
    """
    Construye una columna string con las razones activadas por fila.
    Ejemplo:
        'pickup__future;trip_distance__negative'
    """
    reasons = pd.Series("", index=index, dtype="string")

    for name, mask in masks.items():
        mask_filled = mask.fillna(False)
        reasons = reasons.where(~mask_filled, reasons + name + ";")

    return reasons.str.rstrip(";").replace("", pd.NA)


@dataclass
class ValidationResult:
    df_clean: pd.DataFrame
    df_bad: pd.DataFrame
    report: Dict[str, Any]


def validate_yellow_df(
    df: pd.DataFrame,
    strict_columns: bool = False,
    source_file: Optional[Path] = None,
    valid_location_ids: Optional[Set[int]] = None,
) -> ValidationResult:
    report: Dict[str, Any] = {"n_rows": int(len(df))}

    # 1) Estandarizar columnas a canónicas
    df, renamed, missing = standardize_yellow_columns(df)
    report["columns_missing"] = missing
    report["columns_renamed"] = renamed
    report["columns_unexpected"] = [c for c in df.columns if c not in set(EXPECTED_COLUMNS)]

    if strict_columns and missing:
        raise ValueError(f"Faltan columnas esperadas según diccionario Yellow: {missing}")

    invalid_masks: Dict[str, pd.Series] = {}
    warning_masks: Dict[str, pd.Series] = {}

    # -----------------------------
    # 2) Required mínimo para análisis
    # -----------------------------
    # Estas columnas son imprescindibles para poder analizar un viaje.
    for c in ["VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID"]:
        invalid_masks[f"{c}__missing_required"] = df[c].isna()

    # -----------------------------
    # 3) Datetimes + coherencia temporal
    # -----------------------------
    df["tpep_pickup_datetime"], inv_pickup_dt = _to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"], inv_drop_dt = _to_datetime(df["tpep_dropoff_datetime"])
    invalid_masks["tpep_pickup_datetime__invalid_datetime"] = inv_pickup_dt
    invalid_masks["tpep_dropoff_datetime__invalid_datetime"] = inv_drop_dt

    pickup = df["tpep_pickup_datetime"]
    drop = df["tpep_dropoff_datetime"]

    invalid_masks["datetime__dropoff_before_pickup"] = pickup.notna() & drop.notna() & (drop < pickup)

    # Fechas futuras: son incompatibles con un dataset histórico cerrado.
    # Se usa normalize() para comparar por fecha y evitar falsos positivos por zona horaria/hora.
    now_ts = pd.Timestamp.now()
    invalid_masks["tpep_pickup_datetime__future"] = pickup.notna() & (pickup > now_ts)
    invalid_masks["tpep_dropoff_datetime__future"] = drop.notna() & (drop > now_ts)

    # Validación contra el mes esperado según nombre del fichero.
    # Si el fichero es yellow_tripdata_2023-07.parquet, esperamos viajes de 2023-07.
    exp_year, exp_month = (None, None)
    if source_file is not None:
        exp_year, exp_month = extract_expected_year_month_from_filename(source_file)

    if exp_year is not None and exp_month is not None:
        # El pickup sí debe caer dentro del mes exacto del fichero.
        invalid_masks["tpep_pickup_datetime__outside_expected_file_month"] = (
            pickup.notna()
            & ((pickup.dt.year != exp_year) | (pickup.dt.month != exp_month))
        )

        # Último instante del mes del fichero.
        month_start = pd.Timestamp(year=exp_year, month=exp_month, day=1)
        next_month_start = month_start + pd.offsets.MonthBegin(1)
        month_end_last_instant = next_month_start - pd.Timedelta(seconds=1)

        # Permitimos que dropoff se desborde como mucho el máximo de duración.
        dropoff_latest_allowed = month_end_last_instant + pd.Timedelta(minutes=MAX_TRIP_DURATION_MIN)

        invalid_masks["tpep_dropoff_datetime__too_far_beyond_expected_month"] = (
            drop.notna() & (drop > dropoff_latest_allowed)
        )

    # -----------------------------
    # 4) Integers (tipo) + dominios
    # -----------------------------
    # VendorID
    df["VendorID"], inv_v_type, inv_v_dec = _to_int_nullable(df["VendorID"])
    invalid_masks["VendorID__invalid_type"] = inv_v_type
    invalid_masks["VendorID__invalid_decimal"] = inv_v_dec
    invalid_masks["VendorID__out_of_domain"] = df["VendorID"].notna() & ~df["VendorID"].isin(ALLOWED_VENDOR_ID)

    # RatecodeID (si viene, validar dominio; missing no es crítico)
    df["RatecodeID"], inv_r_type, inv_r_dec = _to_int_nullable(df["RatecodeID"])
    invalid_masks["RatecodeID__invalid_type"] = inv_r_type
    invalid_masks["RatecodeID__invalid_decimal"] = inv_r_dec
    invalid_masks["RatecodeID__out_of_domain"] = df["RatecodeID"].notna() & ~df["RatecodeID"].isin(ALLOWED_RATECODE_ID)

    # payment_type
    df["payment_type"], inv_p_type, inv_p_dec = _to_int_nullable(df["payment_type"])
    invalid_masks["payment_type__invalid_type"] = inv_p_type
    invalid_masks["payment_type__invalid_decimal"] = inv_p_dec
    invalid_masks["payment_type__out_of_domain"] = df["payment_type"].notna() & ~df["payment_type"].isin(ALLOWED_PAYMENT_TYPE)

    # passenger_count
    # Missing permitido; negativo no tiene sentido.
    # passenger_count = 0 se conserva como warning, no como invalid.
    df["passenger_count"], inv_pc_type, inv_pc_dec = _to_int_nullable(df["passenger_count"])
    invalid_masks["passenger_count__invalid_type"] = inv_pc_type
    invalid_masks["passenger_count__invalid_decimal"] = inv_pc_dec
    invalid_masks["passenger_count__negative"] = df["passenger_count"].notna() & (df["passenger_count"] < 0)
    warning_masks["passenger_count__zero"] = df["passenger_count"].notna() & (df["passenger_count"] == 0)

    # PU/DO Location IDs
    df["PULocationID"], inv_pu_type, inv_pu_dec = _to_int_nullable(df["PULocationID"])
    df["DOLocationID"], inv_do_type, inv_do_dec = _to_int_nullable(df["DOLocationID"])
    invalid_masks["PULocationID__invalid_type"] = inv_pu_type
    invalid_masks["PULocationID__invalid_decimal"] = inv_pu_dec
    invalid_masks["DOLocationID__invalid_type"] = inv_do_type
    invalid_masks["DOLocationID__invalid_decimal"] = inv_do_dec

    # Si tenemos catálogo TLC, comprobamos que el ID exista de verdad.
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
    s = df["store_and_fwd_flag"].astype("string").str.strip().str.upper()
    df["store_and_fwd_flag"] = s
    warning_masks["store_and_fwd_flag__missing"] = df["store_and_fwd_flag"].isna()
    invalid_masks["store_and_fwd_flag__out_of_domain"] = (
        df["store_and_fwd_flag"].notna() & ~df["store_and_fwd_flag"].isin(ALLOWED_STORE_FLAG)
    )

    # -----------------------------
    # 6) Floats (tipo)
    # -----------------------------
    for c in FLOAT_COLS:
        df[c], inv = _to_float_nullable(df[c])
        invalid_masks[f"{c}__invalid_type"] = inv

    # Reglas de plausibilidad fuerte en variables numéricas
    invalid_masks["trip_distance__negative"] = df["trip_distance"].notna() & (df["trip_distance"] < 0)

    # Tasas/surcharges que, por definición operativa, no deberían ser negativas.
    invalid_masks["mta_tax__negative"] = df["mta_tax"].notna() & (df["mta_tax"] < 0)
    invalid_masks["improvement_surcharge__negative"] = (
        df["improvement_surcharge"].notna() & (df["improvement_surcharge"] < 0)
    )
    invalid_masks["congestion_surcharge__negative"] = (
        df["congestion_surcharge"].notna() & (df["congestion_surcharge"] < 0)
    )
    invalid_masks["airport_fee__negative"] = df["airport_fee"].notna() & (df["airport_fee"] < 0)
    invalid_masks["cbd_congestion_fee__negative"] = (
        df["cbd_congestion_fee"].notna() & (df["cbd_congestion_fee"] < 0)
    )

    # Warnings de plausibilidad suave
    warning_masks["trip_distance__zero"] = df["trip_distance"].notna() & (df["trip_distance"] == 0)
    warning_masks["fare_amount__extreme"] = df["fare_amount"].notna() & (df["fare_amount"] > EXTREME_FARE_AMOUNT)
    warning_masks["total_amount__extreme"] = df["total_amount"].notna() & (df["total_amount"] > EXTREME_TOTAL_AMOUNT)

    # -----------------------------
    # 7) Duración y velocidad implícita
    # -----------------------------
    trip_duration_min = (drop - pickup).dt.total_seconds() / 60.0
    df["trip_duration_min"] = trip_duration_min.astype("Float64")

    invalid_masks["trip_duration_min__negative"] = trip_duration_min.notna() & (trip_duration_min < 0)
    invalid_masks["trip_duration_min__too_long"] = trip_duration_min.notna() & (trip_duration_min > MAX_TRIP_DURATION_MIN)
    warning_masks["trip_duration_min__zero"] = trip_duration_min.notna() & (trip_duration_min == 0)

    # Velocidad implícita en mph = distancia / horas.
    # Solo se calcula si duración > 0 y distancia no nula.
    duration_hours = trip_duration_min / 60.0
    implied_speed_mph = pd.Series(pd.NA, index=df.index, dtype="Float64")
    valid_speed_base = (
        df["trip_distance"].notna()
        & duration_hours.notna()
        & (df["trip_distance"] > 0)
        & (duration_hours > 0)
    )
    implied_speed_mph.loc[valid_speed_base] = (df.loc[valid_speed_base, "trip_distance"] / duration_hours.loc[valid_speed_base]).astype("Float64")
    df["implied_speed_mph"] = implied_speed_mph

    invalid_masks["implied_speed_mph__too_high"] = (
        df["implied_speed_mph"].notna() & (df["implied_speed_mph"] > MAX_REASONABLE_SPEED_MPH)
    )

    # -----------------------------
    # 8) Coherencia contable suave (warning)
    # -----------------------------
    # No la usamos como invalid porque puede haber ajustes/redondeos internos.
    total_components = (
        df["fare_amount"].fillna(0)
        + df["extra"].fillna(0)
        + df["mta_tax"].fillna(0)
        + df["tip_amount"].fillna(0)
        + df["tolls_amount"].fillna(0)
        + df["improvement_surcharge"].fillna(0)
        + df["congestion_surcharge"].fillna(0)
        + df["airport_fee"].fillna(0)
        + df["cbd_congestion_fee"].fillna(0)
    )
    total_diff = (df["total_amount"] - total_components).abs()
    warning_masks["total_amount__component_mismatch"] = (
        df["total_amount"].notna() & (total_diff > TOTAL_AMOUNT_TOLERANCE)
    )

    # -----------------------------
    # 9) Duplicados exactos
    # -----------------------------
    # Duplicado exacto completo de fila: asumimos error de ingestión o duplicación.
    invalid_masks["row__duplicate_exact"] = df.duplicated(keep="first")

    # -----------------------------
    # 10) Report + columnas de razones
    # -----------------------------
    report["invalid_counts"] = {k: int(v.fillna(False).sum()) for k, v in invalid_masks.items()}
    report["warning_counts"] = {k: int(v.fillna(False).sum()) for k, v in warning_masks.items()}

    any_invalid = pd.Series(False, index=df.index)
    for v in invalid_masks.values():
        any_invalid |= v.fillna(False)

    any_warning = pd.Series(False, index=df.index)
    for v in warning_masks.values():
        any_warning |= v.fillna(False)

    # Añadimos trazabilidad de razones.
    df = df.copy()
    df["warning_reasons"] = _build_reason_column(warning_masks, df.index)

    df_bad = df.loc[any_invalid].copy()
    df_bad["rejection_reasons"] = _build_reason_column(invalid_masks, df_bad.index)

    df_clean = df.loc[~any_invalid].copy()

    report["n_bad_rows"] = int(len(df_bad))
    report["n_clean_rows"] = int(len(df_clean))
    report["n_warning_rows_in_clean"] = int(df_clean["warning_reasons"].notna().sum())

    return ValidationResult(df_clean=df_clean, df_bad=df_bad, report=report)


def write_outputs(result: ValidationResult, output_dir: Path, input_file: Path, write_bad: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_file.stem  # yellow_tripdata_YYYY-MM
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
        description="Capa1 Yellow: validación estructural + plausibilidad fuerte según data dictionary."
    )

    p.add_argument("--service-folder", type=str, default="yellow", help="Subcarpeta dentro de data/raw. Default: yellow")
    p.add_argument("--prefix", type=str, default="yellow_tripdata_", help="Prefijo. Default: yellow_tripdata_")
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

    return p.parse_args()


def main() -> None:
    args = parse_args()

    base_raw = obtener_ruta("data/raw") / args.service_folder
    base_out = obtener_ruta("data/validated") / args.service_folder
    out_reports = obtener_ruta("outputs/procesamiento") / "capa1_yellow"

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

    console.print(Panel.fit("[bold cyan]CAPA 1 — YELLOW: VALIDACIÓN (TIPOS + DOMINIOS + PLAUSIBILIDAD)[/bold cyan]"))

    table = Table(title="Capa1 Yellow — resumen por fichero", header_style="bold magenta")
    table.add_column("Fichero")
    table.add_column("Rows", justify="right")
    table.add_column("Clean", justify="right")
    table.add_column("Bad", justify="right")
    table.add_column("Warn(clean)", justify="right")

    summary: Dict[str, Any] = {
        "service_folder": args.service_folder,
        "prefix": args.prefix,
        "n_files": len(files),
        "files": [],
        "totals": {"n_rows": 0, "n_clean_rows": 0, "n_bad_rows": 0, "n_warning_rows_in_clean": 0},
    }

    for f in files:
        with console.status(f"[cyan]Validando {f.name}..."):
            df = pq.read_table(f).to_pandas()
            res = validate_yellow_df(
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

        entry = {"file": str(f), **res.report}
        summary["files"].append(entry)
        summary["totals"]["n_rows"] += res.report["n_rows"]
        summary["totals"]["n_clean_rows"] += res.report["n_clean_rows"]
        summary["totals"]["n_bad_rows"] += res.report["n_bad_rows"]
        summary["totals"]["n_warning_rows_in_clean"] += res.report["n_warning_rows_in_clean"]

    console.print(table)

    out_reports.mkdir(parents=True, exist_ok=True)
    summary_path = out_reports / "capa1_yellow_validation_summary.json"
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