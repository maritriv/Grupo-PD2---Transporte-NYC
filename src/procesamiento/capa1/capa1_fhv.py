# src/procesamiento/capa1/capa1_fhv.py

from __future__ import annotations

import argparse
import json
import re
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
# Especificación FHV (Data Dictionary - Mar 18, 2025)
# -----------------------------
EXPECTED_COLUMNS: List[str] = [
    "dispatching_base_num",
    "pickup_datetime",
    "dropOff_datetime",
    "PUlocationID",
    "DOlocationID",
    "SR_Flag",
    "Affiliated_base_number",
]

ALLOWED_SR_FLAG = {1}
BASE_NUM_PATTERN = re.compile(r"^[A-Z]\d{5}$")  # p.ej. B00013


# -----------------------------
# Helpers
# -----------------------------
def _lower_map(cols: List[str]) -> Dict[str, str]:
    return {c.lower(): c for c in cols}


def _resolve_col(df: pd.DataFrame, canonical: str, aliases: Optional[List[str]] = None) -> Optional[str]:
    """Devuelve el nombre real de la columna en df (case-insensitive), o None si no existe."""
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


def standardize_fhv_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Renombra columnas encontradas (case-insensitive + aliases) a los nombres canónicos de EXPECTED_COLUMNS.
    Si alguna falta, la crea como NA.
    Devuelve: (df, rename_map, missing_columns)
    """
    rename_map: Dict[str, str] = {}
    missing: List[str] = []

    # Resolver con aliases típicos (capitalización)
    mapping_specs = {
        "dispatching_base_num": ["Dispatching_base_num", "dispatch_base_num"],
        "pickup_datetime": ["Pickup_datetime", "pickup_datetime"],
        "dropOff_datetime": ["dropoff_datetime", "Dropoff_datetime", "dropoff_datetime", "dropOff_datetime"],
        "PUlocationID": ["PULocationID", "PUlocationID"],
        "DOlocationID": ["DOLocationID", "DOlocationID"],
        "SR_Flag": ["sr_flag", "SR_flag", "srFlag"],
        "Affiliated_base_number": ["affiliated_base_number", "Affiliated_Base_Number"],
    }

    for canon, aliases in mapping_specs.items():
        real = _resolve_col(df, canon, aliases=aliases)
        if real is None:
            missing.append(canon)
        else:
            if real != canon:
                rename_map[real] = canon

    if rename_map:
        df = df.rename(columns=rename_map)

    # Crear faltantes con NA para esquema consistente
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


def _normalize_base_num(s: pd.Series) -> Tuple[pd.Series, pd.Series]:
    s2 = s.astype("string")
    norm = s2.str.strip().str.upper()
    invalid = norm.notna() & ~norm.str.match(BASE_NUM_PATTERN)
    return norm, invalid


def discover_parquet_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted([p for p in input_path.rglob("*.parquet") if p.is_file()])


@dataclass
class ValidationResult:
    df_clean: pd.DataFrame
    df_bad: pd.DataFrame
    df_geo_ready: pd.DataFrame
    report: Dict[str, Any]


def validate_fhv_df(df: pd.DataFrame, strict_columns: bool = False, require_locations: bool = False) -> ValidationResult:
    report: Dict[str, Any] = {"n_rows": int(len(df))}

    # Estandarizar nombres a canónicos
    df, rename_map, missing_cols = standardize_fhv_columns(df)

    report["columns_missing"] = missing_cols
    report["columns_renamed"] = rename_map
    report["columns_unexpected"] = [c for c in df.columns if c not in set(EXPECTED_COLUMNS)]

    if strict_columns and missing_cols:
        raise ValueError(f"Faltan columnas (según diccionario FHV): {missing_cols}")

    invalid_masks: Dict[str, pd.Series] = {}
    warning_masks: Dict[str, pd.Series] = {}

    # -----------------------------
    # CRÍTICOS (invalid): timestamps + consistencia temporal
    # -----------------------------
    # Missing required: pickup/dropoff siempre
    invalid_masks["pickup_datetime__missing_required"] = df["pickup_datetime"].isna()
    invalid_masks["dropOff_datetime__missing_required"] = df["dropOff_datetime"].isna()

    # Datetimes parseables
    df["pickup_datetime"], inv_pickup_dt = _to_datetime(df["pickup_datetime"])
    df["dropOff_datetime"], inv_drop_dt = _to_datetime(df["dropOff_datetime"])
    invalid_masks["pickup_datetime__invalid_datetime"] = inv_pickup_dt
    invalid_masks["dropOff_datetime__invalid_datetime"] = inv_drop_dt

    # Regla temporal: dropoff >= pickup
    pickup = df["pickup_datetime"]
    drop = df["dropOff_datetime"]
    invalid_masks["datetime__dropoff_before_pickup"] = pickup.notna() & drop.notna() & (drop < pickup)

    # -----------------------------
    # Location IDs
    # - Siempre casteamos y validamos tipo/decimales (eso sí es invalid).
    # - Missing: por defecto WARNING (porque tus datos vienen muy incompletos).
    # - Si --require-locations: missing pasa a INVALID.
    # -----------------------------
    df["PUlocationID"], inv_pu_type, inv_pu_dec = _to_int_nullable(df["PUlocationID"])
    df["DOlocationID"], inv_do_type, inv_do_dec = _to_int_nullable(df["DOlocationID"])

    invalid_masks["PUlocationID__invalid_type"] = inv_pu_type
    invalid_masks["PUlocationID__invalid_decimal"] = inv_pu_dec
    invalid_masks["DOlocationID__invalid_type"] = inv_do_type
    invalid_masks["DOlocationID__invalid_decimal"] = inv_do_dec

    pu_missing = df["PUlocationID"].isna()
    do_missing = df["DOlocationID"].isna()

    if require_locations:
        invalid_masks["PUlocationID__missing_required"] = pu_missing
        invalid_masks["DOlocationID__missing_required"] = do_missing
    else:
        warning_masks["PUlocationID__missing"] = pu_missing
        warning_masks["DOlocationID__missing"] = do_missing

    # -----------------------------
    # WARNINGS (no excluyen): bases + SR_Flag
    # -----------------------------
    df["dispatching_base_num"], inv_dispatch_fmt = _normalize_base_num(df["dispatching_base_num"])
    df["Affiliated_base_number"], inv_aff_fmt = _normalize_base_num(df["Affiliated_base_number"])

    warning_masks["dispatching_base_num__missing"] = df["dispatching_base_num"].isna()
    warning_masks["Affiliated_base_number__missing"] = df["Affiliated_base_number"].isna()
    warning_masks["dispatching_base_num__invalid_format"] = inv_dispatch_fmt
    warning_masks["Affiliated_base_number__invalid_format"] = inv_aff_fmt

    # SR_Flag: NULL o 1. Lo dejamos en warning si viene raro.
    df["SR_Flag"], inv_sr_type, inv_sr_dec = _to_int_nullable(df["SR_Flag"])
    warning_masks["SR_Flag__invalid_type"] = inv_sr_type
    warning_masks["SR_Flag__invalid_decimal"] = inv_sr_dec
    warning_masks["SR_Flag__out_of_domain"] = df["SR_Flag"].notna() & ~df["SR_Flag"].isin(ALLOWED_SR_FLAG)

    # -----------------------------
    # Report
    # -----------------------------
    report["invalid_counts"] = {k: int(v.fillna(False).sum()) for k, v in invalid_masks.items()}
    report["warning_counts"] = {k: int(v.fillna(False).sum()) for k, v in warning_masks.items()}

    # Filas inválidas = cualquier invalid
    any_invalid = pd.Series(False, index=df.index)
    for v in invalid_masks.values():
        any_invalid |= v.fillna(False)

    df_bad = df.loc[any_invalid].copy()
    df_clean = df.loc[~any_invalid].copy()

    report["n_bad_rows"] = int(len(df_bad))
    report["n_clean_rows"] = int(len(df_clean))

    # Subset geo_ready: clean + PU y DO presentes (para mapear a borough)
    geo_mask = df_clean["PUlocationID"].notna() & df_clean["DOlocationID"].notna()
    df_geo_ready = df_clean.loc[geo_mask].copy()
    report["n_geo_ready_rows"] = int(len(df_geo_ready))

    return ValidationResult(df_clean=df_clean, df_bad=df_bad, df_geo_ready=df_geo_ready, report=report)


def write_outputs(
    result: ValidationResult,
    output_dir: Path,
    input_file: Path,
    write_bad: bool,
    write_geo_ready: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_file.stem  # ej: fhv_tripdata_2023-01
    clean_path = output_dir / "clean" / f"{stem}.parquet"
    report_path = output_dir / "reports" / f"{stem}.validation_report.json"

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    result.df_clean.to_parquet(clean_path, index=False)

    if write_bad:
        bad_path = output_dir / "bad_rows" / f"{stem}.parquet"
        bad_path.parent.mkdir(parents=True, exist_ok=True)
        result.df_bad.to_parquet(bad_path, index=False)

    if write_geo_ready:
        geo_path = output_dir / "geo_ready" / f"{stem}.parquet"
        geo_path.parent.mkdir(parents=True, exist_ok=True)
        result.df_geo_ready.to_parquet(geo_path, index=False)

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(result.report, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capa1 FHV: validación (invalid vs warnings) según data dictionary.")

    p.add_argument("--service-folder", type=str, default="fhv", help="Subcarpeta dentro de data/raw. Default: fhv")
    p.add_argument("--prefix", type=str, default="fhv_tripdata_", help="Prefijo de fichero. Default: fhv_tripdata_")
    p.add_argument("--input", type=str, default="", help="Ruta parquet o directorio. Si vacío usa data/raw/<service-folder>.")
    p.add_argument("--output", type=str, default="", help="Salida. Si vacío usa data/validated/<service-folder>.")
    p.add_argument("--months", nargs="*", default=[], help="Meses YYYY-MM para construir <prefix><YYYY-MM>.parquet")

    p.add_argument("--strict-columns", action="store_true", help="Falla si faltan columnas del diccionario.")
    p.add_argument("--require-locations", action="store_true", help="Si se activa, PU/DO missing se considera INVALID (modo estricto).")
    p.add_argument("--write-bad", action="store_true", help="Escribe bad_rows.")
    p.add_argument("--write-geo-ready", action="store_true", help="Escribe geo_ready (clean con PU+DO presentes).")
    p.add_argument("--limit-files", type=int, default=0, help="Limita nº ficheros (debug).")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    base_raw = obtener_ruta("data/raw") / args.service_folder
    base_out = obtener_ruta("data/validated") / args.service_folder
    out_reports = obtener_ruta("outputs/procesamiento") / "capa1_fhv"

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

    console.print(Panel.fit("[bold cyan]CAPA 1 — FHV: VALIDACIÓN (INVALID vs WARNINGS)[/bold cyan]"))

    table = Table(title="Capa1 FHV — resumen por fichero", header_style="bold magenta")
    table.add_column("Fichero")
    table.add_column("Rows", justify="right")
    table.add_column("Clean", justify="right")
    table.add_column("Bad", justify="right")
    table.add_column("GeoReady", justify="right")

    summary: Dict[str, Any] = {
        "service_folder": args.service_folder,
        "prefix": args.prefix,
        "require_locations": args.require_locations,
        "n_files": len(files),
        "files": [],
        "totals": {"n_rows": 0, "n_clean_rows": 0, "n_bad_rows": 0, "n_geo_ready_rows": 0},
    }

    for f in files:
        with console.status(f"[cyan]Validando {f.name}..."):
            df = pq.read_table(f).to_pandas()
            res = validate_fhv_df(df, strict_columns=args.strict_columns, require_locations=args.require_locations)
            write_outputs(res, output_dir, f, write_bad=args.write_bad, write_geo_ready=args.write_geo_ready)

        table.add_row(
            f.name,
            str(res.report["n_rows"]),
            str(res.report["n_clean_rows"]),
            str(res.report["n_bad_rows"]),
            str(res.report["n_geo_ready_rows"]),
        )

        entry = {"file": str(f), **res.report}
        summary["files"].append(entry)
        summary["totals"]["n_rows"] += res.report["n_rows"]
        summary["totals"]["n_clean_rows"] += res.report["n_clean_rows"]
        summary["totals"]["n_bad_rows"] += res.report["n_bad_rows"]
        summary["totals"]["n_geo_ready_rows"] += res.report["n_geo_ready_rows"]

    console.print(table)

    out_reports.mkdir(parents=True, exist_ok=True)
    summary_path = out_reports / "capa1_fhv_validation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    console.print(f"\n[bold green]OK[/bold green] Resumen global: {summary_path}")
    console.print(
        f"[green]Totales[/green] rows={summary['totals']['n_rows']} | clean={summary['totals']['n_clean_rows']} | bad={summary['totals']['n_bad_rows']} | geo_ready={summary['totals']['n_geo_ready_rows']}"
    )
    console.print(f"[green]Salida datos[/green] {output_dir}/clean (y bad_rows/geo_ready si activado)")


if __name__ == "__main__":
    main()
