# src/procesamiento/capa1/capa1_green.py

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

OPTIONAL_COLUMNS = {"cbd_congestion_fee"}  # no fallar si falta

ALLOWED_VENDOR_ID = {1, 2, 6}
ALLOWED_RATECODE_ID = {1, 2, 3, 4, 5, 6, 99}
ALLOWED_PAYMENT_TYPE = {0, 1, 2, 3, 4, 5, 6}
ALLOWED_TRIP_TYPE = {1, 2}
ALLOWED_STORE_AND_FWD = {"Y", "N"}


# -----------------------------
# Utils de casting + máscaras de inválidos
# -----------------------------
def _to_datetime(s: pd.Series) -> Tuple[pd.Series, pd.Series]:
    orig_notna = s.notna()
    coerced = pd.to_datetime(s, errors="coerce")
    invalid = orig_notna & coerced.isna()
    return coerced, invalid


def _to_float(s: pd.Series) -> Tuple[pd.Series, pd.Series]:
    orig_notna = s.notna()
    coerced = pd.to_numeric(s, errors="coerce")
    invalid = orig_notna & coerced.isna()
    return coerced.astype("float64"), invalid


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


def _normalize_flag(s: pd.Series) -> Tuple[pd.Series, pd.Series]:
    s2 = s.astype("string")
    norm = s2.str.strip().str.upper()
    invalid = norm.notna() & ~norm.isin(ALLOWED_STORE_AND_FWD)
    return norm, invalid


def _invalid_out_of_domain(s: pd.Series, allowed: set) -> pd.Series:
    return s.notna() & ~s.isin(allowed)


def discover_parquet_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted([p for p in input_path.rglob("*.parquet") if p.is_file()])


def read_parquet_safely(filepath: Path) -> pd.DataFrame:
    """
    Lee solo columnas relevantes (si existen) para ahorrar memoria.
    """
    pf = pq.ParquetFile(filepath)
    cols = pf.schema.names
    cols_to_read = [c for c in EXPECTED_COLUMNS if c in cols] + [c for c in cols if c not in EXPECTED_COLUMNS]
    # OJO: aquí mantenemos extras también (para reportar). Si quieres ignorarlas, quita la segunda parte.
    table = pf.read(columns=[c for c in cols_to_read if c in cols])
    return table.to_pandas()


@dataclass
class ValidationResult:
    df_clean: pd.DataFrame
    df_bad: pd.DataFrame
    report: Dict[str, Any]


def validate_green_df(df: pd.DataFrame, strict_columns: bool = False) -> ValidationResult:
    report: Dict[str, Any] = {"n_rows": int(len(df))}

    cols_present = set(df.columns)
    missing = [c for c in EXPECTED_COLUMNS if c not in cols_present]
    unexpected = [c for c in df.columns if c not in set(EXPECTED_COLUMNS)]

    report["columns_missing"] = missing
    report["columns_unexpected"] = unexpected

    mandatory_missing = [c for c in missing if c not in OPTIONAL_COLUMNS]
    if strict_columns and mandatory_missing:
        raise ValueError(f"Faltan columnas obligatorias: {mandatory_missing}")

    # Añadir faltantes como NA para esquema consistente
    for c in missing:
        df[c] = pd.NA

    invalid_masks: Dict[str, pd.Series] = {}

    # --- Ints ---
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

    # --- Datetimes ---
    for c in ["lpep_pickup_datetime", "lpep_dropoff_datetime"]:
        df[c], inv = _to_datetime(df[c])
        invalid_masks[f"{c}__invalid_datetime"] = inv

    # --- store_and_fwd_flag ---
    df["store_and_fwd_flag"], inv_flag = _normalize_flag(df["store_and_fwd_flag"])
    invalid_masks["store_and_fwd_flag__invalid_value"] = inv_flag

    # --- Floats ---
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
        df[c], inv = _to_float(df[c])
        invalid_masks[f"{c}__invalid_numeric"] = inv

    # --- Dominios ---
    invalid_masks["VendorID__out_of_domain"] = _invalid_out_of_domain(df["VendorID"], ALLOWED_VENDOR_ID)
    invalid_masks["RatecodeID__out_of_domain"] = _invalid_out_of_domain(df["RatecodeID"], ALLOWED_RATECODE_ID)
    invalid_masks["payment_type__out_of_domain"] = _invalid_out_of_domain(df["payment_type"], ALLOWED_PAYMENT_TYPE)
    invalid_masks["trip_type__out_of_domain"] = _invalid_out_of_domain(df["trip_type"], ALLOWED_TRIP_TYPE)

    # --- Regla temporal ---
    pickup = df["lpep_pickup_datetime"]
    dropoff = df["lpep_dropoff_datetime"]
    invalid_time_order = pickup.notna() & dropoff.notna() & (dropoff < pickup)
    invalid_masks["datetime__dropoff_before_pickup"] = invalid_time_order

    # Conteos
    report["invalid_counts"] = {k: int(v.sum()) for k, v in invalid_masks.items()}

    # Máscara global
    any_invalid = pd.Series(False, index=df.index)
    for v in invalid_masks.values():
        any_invalid |= v.fillna(False)

    report["n_bad_rows"] = int(any_invalid.sum())
    report["n_clean_rows"] = int(len(df) - report["n_bad_rows"])

    df_bad = df.loc[any_invalid].copy()
    df_clean = df.loc[~any_invalid].copy()

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
    p = argparse.ArgumentParser(description="Capa1 Green: validación de tipos/dominios (LPEP).")

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

    return p.parse_args()


def main() -> None:
    args = parse_args()

    base_raw = obtener_ruta("data/raw") / "green"
    base_out = obtener_ruta("data/validated") / "green"
    out_reports = obtener_ruta("outputs/procesamiento") / "capa1_green"

    input_path = Path(args.input) if args.input else base_raw
    output_dir = Path(args.output) if args.output else base_out

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

    console.print(Panel.fit("[bold cyan]CAPA 1 — GREEN: VALIDACIÓN (TIPOS + DOMINIOS)[/bold cyan]"))

    table = Table(title="Capa1 Green — resumen por fichero", header_style="bold magenta")
    table.add_column("Fichero")
    table.add_column("Rows", justify="right")
    table.add_column("Clean", justify="right")
    table.add_column("Bad", justify="right")

    summary = {
        "n_files": len(files),
        "files": [],
        "totals": {"n_rows": 0, "n_clean_rows": 0, "n_bad_rows": 0},
    }

    for f in files:
        with console.status(f"[cyan]Validando {f.name}..."):
            df = read_parquet_safely(f)
            res = validate_green_df(df, strict_columns=args.strict_columns)
            write_outputs(res, output_dir, f, write_bad=args.write_bad)

        table.add_row(
            f.name,
            str(res.report["n_rows"]),
            str(res.report["n_clean_rows"]),
            str(res.report["n_bad_rows"]),
        )

        entry = {
            "file": str(f),
            **res.report,
        }
        summary["files"].append(entry)
        summary["totals"]["n_rows"] += res.report["n_rows"]
        summary["totals"]["n_clean_rows"] += res.report["n_clean_rows"]
        summary["totals"]["n_bad_rows"] += res.report["n_bad_rows"]

    console.print(table)

    # Guardar resumen global en outputs
    out_reports.mkdir(parents=True, exist_ok=True)
    summary_path = out_reports / "capa1_green_validation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    console.print(f"\n[bold green]OK[/bold green] Resumen global: {summary_path}")
    console.print(
        f"[green]Totales[/green] rows={summary['totals']['n_rows']} | clean={summary['totals']['n_clean_rows']} | bad={summary['totals']['n_bad_rows']}"
    )
    console.print(f"[green]Salida datos[/green] {output_dir}/clean (y bad_rows si activado)")


if __name__ == "__main__":
    main()
