# src/procesamiento/capa1/capa1_fhv.py

"""
Reglas de Validación Aplicadas (Data Dictionary FHV):
| Campo                | Validación aplicada                          |
|----------------------|----------------------------------------------|
| dispatching_base_num | Formato regex ^[A-Z]\d{5}$                   |
| pickup_datetime      | Conversión a datetime (Requerido)            |
| SR_Flag              | Solo permite valor 1 o null                  |
| PU/DOlocationID      | Conversión a Int64 (Nullable)                |
"""

from __future__ import annotations

import rich_click as click
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
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.STYLE_OPTION = "bold cyan"
click.rich_click.STYLE_SWITCH = "bold green"
click.rich_click.STYLE_METAVAR = "yellow"
click.rich_click.STYLE_HELPTEXT = ""

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
    Renombra columnas encontradas (case-insensitive + aliases) a los nombres canónicos.
    """
    rename_map: Dict[str, str] = {}
    missing: List[str] = []

    mapping_specs = {
        "dispatching_base_num": ["dispatch_base_num"],
        "pickup_datetime": ["pickup_datetime"],
        "dropOff_datetime": ["dropoff_datetime"],
        "PUlocationID": ["pulocationid"],
        "DOlocationID": ["dolocationid"],
        "SR_Flag": ["sr_flag"],
        "Affiliated_base_number": ["affiliated_base_num"],
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

    if strict_columns and missing_cols:
        raise ValueError(f"Faltan columnas (según diccionario FHV): {missing_cols}")

    invalid_masks: Dict[str, pd.Series] = {}
    warning_masks: Dict[str, pd.Series] = {}

    # CRÍTICOS (invalid)
    invalid_masks["pickup_missing"] = df["pickup_datetime"].isna()
    invalid_masks["dropoff_missing"] = df["dropOff_datetime"].isna()

    df["pickup_datetime"], inv_pickup_dt = _to_datetime(df["pickup_datetime"])
    df["dropOff_datetime"], inv_drop_dt = _to_datetime(df["dropOff_datetime"])
    invalid_masks["pickup_inv_dt"] = inv_pickup_dt
    invalid_masks["dropoff_inv_dt"] = inv_drop_dt

    pickup = df["pickup_datetime"]
    drop = df["dropOff_datetime"]
    invalid_masks["temporal_error"] = pickup.notna() & drop.notna() & (drop < pickup)

    # Location IDs
    df["PUlocationID"], inv_pu_type, inv_pu_dec = _to_int_nullable(df["PUlocationID"])
    df["DOlocationID"], inv_do_type, inv_do_dec = _to_int_nullable(df["DOlocationID"])

    invalid_masks["pu_bad_type"] = inv_pu_type | inv_pu_dec
    invalid_masks["do_bad_type"] = inv_do_type | inv_do_dec

    if require_locations:
        invalid_masks["pu_missing_req"] = df["PUlocationID"].isna()
        invalid_masks["do_missing_req"] = df["DOlocationID"].isna()

    # WARNINGS
    df["dispatching_base_num"], inv_dispatch_fmt = _normalize_base_num(df["dispatching_base_num"])
    warning_masks["base_num_fmt"] = inv_dispatch_fmt

    df["SR_Flag"], inv_sr_type, _ = _to_int_nullable(df["SR_Flag"])
    warning_masks["sr_flag_inv"] = (df["SR_Flag"].notna() & ~df["SR_Flag"].isin(ALLOWED_SR_FLAG)) | inv_sr_type

    # --- Construcción del Reporte Compacto ---
    any_invalid = pd.Series(False, index=df.index, dtype="bool")
    for v in invalid_masks.values():
        any_invalid |= v.fillna(False)

    df_bad = df.loc[any_invalid].copy()
    df_clean = df.loc[~any_invalid].copy()

    geo_mask = df_clean["PUlocationID"].notna() & df_clean["DOlocationID"].notna()
    df_geo_ready = df_clean.loc[geo_mask].copy()

    report["stats"] = {
        "clean": int(len(df_clean)),
        "bad": int(len(df_bad)),
        "geo_ready": int(len(df_geo_ready))
    }

    report["errors_found"] = {
        k: int(v.fillna(False).sum())
        for k, v in invalid_masks.items()
        if int(v.fillna(False).sum()) > 0
    }

    report["warnings_found"] = {
        k: int(v.fillna(False).sum())
        for k, v in warning_masks.items()
        if int(v.fillna(False).sum()) > 0
    }

    return ValidationResult(
        df_clean=df_clean,
        df_bad=df_bad,
        df_geo_ready=df_geo_ready,
        report=report
    )


def write_outputs(result: ValidationResult, output_dir: Path, input_file: Path, write_bad: bool, write_geo_ready: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = obtener_ruta("outputs/procesamiento") / "capa1_fhv"
    reports_dir.mkdir(parents=True, exist_ok=True)

    stem = input_file.stem

    clean_p = output_dir / "clean" / f"{stem}.parquet"
    report_p = reports_dir / f"{stem}.json"

    clean_p.parent.mkdir(parents=True, exist_ok=True)

    result.df_clean.to_parquet(clean_p, index=False)

    if write_bad:
        bad_p = output_dir / "bad_rows" / f"{stem}.parquet"
        bad_p.parent.mkdir(parents=True, exist_ok=True)
        result.df_bad.to_parquet(bad_p, index=False)

    if write_geo_ready:
        geo_p = output_dir / "geo_ready" / f"{stem}.parquet"
        geo_p.parent.mkdir(parents=True, exist_ok=True)
        result.df_geo_ready.to_parquet(geo_p, index=False)

    with report_p.open("w", encoding="utf-8") as f:
        json.dump(result.report, f, indent=2, ensure_ascii=False)


@click.command(
    context_settings=dict(help_option_names=["-h", "--help"]),
    help="""
[bold cyan]CAPA 1 — VALIDACIÓN DE DATOS FHV (NYC TLC)[/bold cyan]

Valida archivos Parquet del servicio [bold]For-Hire Vehicles (FHV)[/bold]
aplicando reglas del [italic]Data Dictionary oficial del NYC TLC[/italic].

\n
[bold]El proceso realiza:[/bold]

- Estandarización de nombres de columnas
- Conversión de tipos
- Validación temporal de viajes
- Validación de Location IDs
- Normalización de dispatching_base_num
- Generación de datasets limpios y reportes de calidad

\n
[bold]Outputs generados:[/bold]

- data/validated/fhv/clean
- data/validated/fhv/bad_rows
- data/validated/fhv/geo_ready
- outputs/procesamiento/capa1_fhv
"""
)
@click.option(
    "--service-folder",
    default="fhv",
    show_default=True,
    help="Carpeta del servicio dentro de data/raw."
)
@click.option(
    "--prefix",
    default="fhv_tripdata_",
    show_default=True,
    help="Prefijo de los archivos Parquet a procesar."
)
@click.option(
    "--input",
    "input_arg",
    type=click.Path(path_type=Path),
    default="",
    help="Ruta alternativa de entrada (archivo o carpeta)."
)
@click.option(
    "--output",
    "output_arg",
    type=click.Path(path_type=Path),
    default="",
    help="Ruta alternativa de salida para los datasets validados."
)
@click.option(
    "--months",
    multiple=True,
    metavar="YYYY-MM",
    help="Meses específicos a procesar. Ej: --months 2019-01 --months 2019-02"
)
@click.option(
    "--strict-columns",
    is_flag=True,
    help="Falla si faltan columnas del Data Dictionary."
)
@click.option(
    "--require-locations",
    is_flag=True,
    help="Requiere PUlocationID y DOlocationID no nulos."
)
@click.option(
    "--write-bad",
    is_flag=True,
    help="Guardar filas inválidas en data/validated/.../bad_rows."
)
@click.option(
    "--write-geo-ready",
    is_flag=True,
    help="Guardar subset con LocationIDs válidos para análisis geoespacial."
)
@click.option(
    "--limit-files",
    default=0,
    show_default=True,
    type=int,
    help="Limita el número de archivos a procesar (útil para testing)."
)
def main(service_folder, prefix, input_arg, output_arg, months, strict_columns, require_locations, write_bad, write_geo_ready, limit_files):
    base_raw = obtener_ruta("data/raw") / service_folder
    base_out = obtener_ruta("data/validated") / service_folder

    input_path = Path(input_arg) if input_arg else base_raw
    output_dir = Path(output_arg) if output_arg else base_out

    if months:
        files = [base_raw / f"{prefix}{m}.parquet" for m in months if (base_raw / f"{prefix}{m}.parquet").exists()]
    else:
        files = discover_parquet_files(input_path)

    if limit_files > 0:
        files = files[:limit_files]

    if not files:
        raise click.ClickException("No hay archivos.")

    console.print(Panel.fit("[bold cyan]CAPA 1 — FHV VALIDATION[/bold cyan]"))

    table = Table(title="Resumen FHV", header_style="bold magenta")
    for col in ["Fichero", "Total", "Clean", "Bad", "Geo"]:
        table.add_column(col, justify="right" if col != "Fichero" else "left")

    for f in files:
        with console.status(f"[cyan]Procesando {f.name}..."):
            df = pq.read_table(f).to_pandas()

            res = validate_fhv_df(
                df,
                strict_columns=strict_columns,
                require_locations=require_locations
            )

            write_outputs(
                res,
                output_dir,
                f,
                write_bad,
                write_geo_ready
            )

        table.add_row(
            f.name,
            str(res.report["n_rows"]),
            str(res.report["stats"]["clean"]),
            str(res.report["stats"]["bad"]),
            str(res.report["stats"]["geo_ready"])
        )

    console.print(table)
    console.print("[bold green]Proceso finalizado.[/bold green]")


if __name__ == "__main__":
    main()