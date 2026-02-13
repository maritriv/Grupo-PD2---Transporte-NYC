# src/procesamiento/capa1/capa1_yellow.py

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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


@dataclass
class ValidationResult:
    df_clean: pd.DataFrame
    df_bad: pd.DataFrame
    report: Dict[str, Any]


def validate_yellow_df(df: pd.DataFrame, strict_columns: bool = False) -> ValidationResult:
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

    # 2) Required mínimo para análisis
    # (tpep_pickup_datetime, tpep_dropoff_datetime, PULocationID, DOLocationID y VendorID)
    for c in ["VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID"]:
        invalid_masks[f"{c}__missing_required"] = df[c].isna()

    # 3) Datetimes + coherencia temporal
    df["tpep_pickup_datetime"], inv_pickup_dt = _to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"], inv_drop_dt = _to_datetime(df["tpep_dropoff_datetime"])
    invalid_masks["tpep_pickup_datetime__invalid_datetime"] = inv_pickup_dt
    invalid_masks["tpep_dropoff_datetime__invalid_datetime"] = inv_drop_dt

    pickup = df["tpep_pickup_datetime"]
    drop = df["tpep_dropoff_datetime"]
    invalid_masks["datetime__dropoff_before_pickup"] = pickup.notna() & drop.notna() & (drop < pickup)

    # 4) Integers (tipo) + dominios
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

    # passenger_count (tipo; missing permitido, pero si viene negativo => invalid)
    df["passenger_count"], inv_pc_type, inv_pc_dec = _to_int_nullable(df["passenger_count"])
    invalid_masks["passenger_count__invalid_type"] = inv_pc_type
    invalid_masks["passenger_count__invalid_decimal"] = inv_pc_dec
    invalid_masks["passenger_count__negative"] = df["passenger_count"].notna() & (df["passenger_count"] < 0)

    # PU/DO Location IDs (tipo + missing ya marcado arriba)
    df["PULocationID"], inv_pu_type, inv_pu_dec = _to_int_nullable(df["PULocationID"])
    df["DOLocationID"], inv_do_type, inv_do_dec = _to_int_nullable(df["DOLocationID"])
    invalid_masks["PULocationID__invalid_type"] = inv_pu_type
    invalid_masks["PULocationID__invalid_decimal"] = inv_pu_dec
    invalid_masks["DOLocationID__invalid_type"] = inv_do_type
    invalid_masks["DOLocationID__invalid_decimal"] = inv_do_dec

    # 5) store_and_fwd_flag (Y/N) — si falta, warning; si viene raro, invalid
    s = df["store_and_fwd_flag"].astype("string").str.strip().str.upper()
    df["store_and_fwd_flag"] = s
    warning_masks["store_and_fwd_flag__missing"] = df["store_and_fwd_flag"].isna()
    invalid_masks["store_and_fwd_flag__out_of_domain"] = df["store_and_fwd_flag"].notna() & ~df["store_and_fwd_flag"].isin(ALLOWED_STORE_FLAG)

    # 6) Floats (tipo)
    for c in FLOAT_COLS:
        df[c], inv = _to_float_nullable(df[c])
        invalid_masks[f"{c}__invalid_type"] = inv

    # Opcional: distancia negativa no tiene sentido (lo tratamos como invalid suave)
    invalid_masks["trip_distance__negative"] = df["trip_distance"].notna() & (df["trip_distance"] < 0)

    # 7) Report
    report["invalid_counts"] = {k: int(v.fillna(False).sum()) for k, v in invalid_masks.items()}
    report["warning_counts"] = {k: int(v.fillna(False).sum()) for k, v in warning_masks.items()}

    any_invalid = pd.Series(False, index=df.index)
    for v in invalid_masks.values():
        any_invalid |= v.fillna(False)

    df_bad = df.loc[any_invalid].copy()
    df_clean = df.loc[~any_invalid].copy()

    report["n_bad_rows"] = int(len(df_bad))
    report["n_clean_rows"] = int(len(df_clean))

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
    p = argparse.ArgumentParser(description="Capa1 Yellow: validación estructural según data dictionary.")

    p.add_argument("--service-folder", type=str, default="yellow", help="Subcarpeta dentro de data/raw. Default: yellow")
    p.add_argument("--prefix", type=str, default="yellow_tripdata_", help="Prefijo. Default: yellow_tripdata_")
    p.add_argument("--input", type=str, default="", help="Ruta parquet o directorio. Si vacío usa data/raw/<service-folder>.")
    p.add_argument("--output", type=str, default="", help="Salida. Si vacío usa data/validated/<service-folder>.")
    p.add_argument("--months", nargs="*", default=[], help="Meses YYYY-MM para construir <prefix><YYYY-MM>.parquet")
    p.add_argument("--strict-columns", action="store_true", help="Falla si faltan columnas del diccionario.")
    p.add_argument("--write-bad", action="store_true", help="Escribe bad_rows.")
    p.add_argument("--limit-files", type=int, default=0, help="Limita nº ficheros (debug).")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    base_raw = obtener_ruta("data/raw") / args.service_folder
    base_out = obtener_ruta("data/validated") / args.service_folder
    out_reports = obtener_ruta("outputs/procesamiento") / "capa1_yellow"

    input_path = Path(args.input) if args.input else base_raw
    output_dir = Path(args.output) if args.output else base_out

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

    console.print(Panel.fit("[bold cyan]CAPA 1 — YELLOW: VALIDACIÓN (TIPOS + DOMINIOS + TIEMPO)[/bold cyan]"))

    table = Table(title="Capa1 Yellow — resumen por fichero", header_style="bold magenta")
    table.add_column("Fichero")
    table.add_column("Rows", justify="right")
    table.add_column("Clean", justify="right")
    table.add_column("Bad", justify="right")

    summary: Dict[str, Any] = {
        "service_folder": args.service_folder,
        "prefix": args.prefix,
        "n_files": len(files),
        "files": [],
        "totals": {"n_rows": 0, "n_clean_rows": 0, "n_bad_rows": 0},
    }

    for f in files:
        with console.status(f"[cyan]Validando {f.name}..."):
            df = pq.read_table(f).to_pandas()
            res = validate_yellow_df(df, strict_columns=args.strict_columns)
            write_outputs(res, output_dir, f, write_bad=args.write_bad)

        table.add_row(
            f.name,
            str(res.report["n_rows"]),
            str(res.report["n_clean_rows"]),
            str(res.report["n_bad_rows"]),
        )

        entry = {"file": str(f), **res.report}
        summary["files"].append(entry)
        summary["totals"]["n_rows"] += res.report["n_rows"]
        summary["totals"]["n_clean_rows"] += res.report["n_clean_rows"]
        summary["totals"]["n_bad_rows"] += res.report["n_bad_rows"]

    console.print(table)

    out_reports.mkdir(parents=True, exist_ok=True)
    summary_path = out_reports / "capa1_yellow_validation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    console.print(f"\n[bold green]OK[/bold green] Resumen global: {summary_path}")
    console.print(
        f"[green]Totales[/green] rows={summary['totals']['n_rows']} | clean={summary['totals']['n_clean_rows']} | bad={summary['totals']['n_bad_rows']}"
    )
    console.print(f"[green]Salida datos[/green] {output_dir}/clean (y bad_rows si activado)")


if __name__ == "__main__":
    main()
