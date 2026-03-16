from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import obtener_ruta

console = Console()

REQUIRED_COLUMNS = ["date", "hour", "borough", "event_type", "n_events"]
KNOWN_BOROUGHS = {"Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"}
MIN_YEAR = 2023
MAX_YEAR = 2025


def read_raw_events(in_dir: Path, base_name: str = "events_daily_borough_type") -> pd.DataFrame:
    in_dir = Path(in_dir).resolve()
    if not in_dir.exists():
        raise FileNotFoundError(f"No existe el directorio RAW: {in_dir}")

    parquet_path = in_dir / f"{base_name}.parquet"
    csv_path = in_dir / f"{base_name}.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    files = sorted(in_dir.glob("events_*.parquet"))
    if not files:
        files = sorted(in_dir.glob("events_*.parquet.gz"))
    if not files:
        files = sorted(in_dir.glob("events_*.csv"))
    if not files:
        files = sorted(in_dir.glob("events_*.csv.gz"))

    if not files:
        raise FileNotFoundError(
            f"No encuentro {base_name}.parquet/.csv ni events_*.parquet/.csv en {in_dir}"
        )

    dfs = []
    for fp in files:
        if fp.suffix.startswith(".parquet") or ".parquet" in fp.name:
            dfs.append(pd.read_parquet(fp))
        else:
            dfs.append(pd.read_csv(fp))
    return pd.concat(dfs, ignore_index=True)


def _build_reason_column(masks: Dict[str, pd.Series], index: pd.Index) -> pd.Series:
    reasons = pd.Series("", index=index, dtype="string")
    for name, mask in masks.items():
        reasons = reasons.where(~mask.fillna(False), reasons + name + ";")
    return reasons.str.rstrip(";").replace("", pd.NA)


@dataclass
class ValidationResult:
    df_clean: pd.DataFrame
    df_bad: pd.DataFrame
    report: Dict[str, Any]


def validate_events_df(df: pd.DataFrame) -> ValidationResult:
    df = df.copy()
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA

    report: Dict[str, Any] = {"n_rows": int(len(df))}
    invalid_masks: Dict[str, pd.Series] = {}
    warning_masks: Dict[str, pd.Series] = {}

    # date
    date_ts = pd.to_datetime(df["date"], errors="coerce")
    invalid_masks["date__missing_required"] = df["date"].isna()
    invalid_masks["date__invalid"] = df["date"].notna() & date_ts.isna()
    invalid_masks["date__outside_allowed_year"] = date_ts.notna() & (
        (date_ts.dt.year < MIN_YEAR) | (date_ts.dt.year > MAX_YEAR)
    )
    df["date"] = date_ts.dt.date

    # hour
    hour_num = pd.to_numeric(df["hour"], errors="coerce")
    inv_hour_type = df["hour"].notna() & hour_num.isna()
    inv_hour_decimal = hour_num.notna() & ((hour_num % 1) != 0)
    hour_num = hour_num.where(~inv_hour_decimal)
    hour_int = hour_num.astype("Int64")

    invalid_masks["hour__missing_required"] = df["hour"].isna()
    invalid_masks["hour__invalid_type"] = inv_hour_type
    invalid_masks["hour__invalid_decimal"] = inv_hour_decimal
    invalid_masks["hour__out_of_range"] = hour_int.notna() & ((hour_int < 0) | (hour_int > 23))
    df["hour"] = hour_int

    # n_events
    events_num = pd.to_numeric(df["n_events"], errors="coerce")
    inv_events_type = df["n_events"].notna() & events_num.isna()
    inv_events_decimal = events_num.notna() & ((events_num % 1) != 0)
    events_num = events_num.where(~inv_events_decimal).astype("Int64")

    invalid_masks["n_events__missing_required"] = df["n_events"].isna()
    invalid_masks["n_events__invalid_type"] = inv_events_type
    invalid_masks["n_events__invalid_decimal"] = inv_events_decimal
    invalid_masks["n_events__negative"] = events_num.notna() & (events_num < 0)
    df["n_events"] = events_num

    # strings
    borough = df["borough"].astype("string").str.strip().str.title()
    event_type = df["event_type"].astype("string").str.strip()
    invalid_masks["borough__missing_required"] = borough.isna() | (borough == "")
    invalid_masks["event_type__missing_required"] = event_type.isna() | (event_type == "")
    warning_masks["borough__unknown_value"] = borough.notna() & ~borough.isin(KNOWN_BOROUGHS)
    warning_masks["row__duplicate_exact"] = df.duplicated(keep="first")
    df["borough"] = borough
    df["event_type"] = event_type

    report["invalid_counts"] = {k: int(v.fillna(False).sum()) for k, v in invalid_masks.items()}
    report["warning_counts"] = {k: int(v.fillna(False).sum()) for k, v in warning_masks.items()}

    any_invalid = pd.Series(False, index=df.index)
    for mask in invalid_masks.values():
        any_invalid |= mask.fillna(False)

    df["warning_reasons"] = _build_reason_column(warning_masks, df.index)
    df_bad = df.loc[any_invalid].copy()
    df_bad["rejection_reasons"] = _build_reason_column(invalid_masks, df_bad.index)
    df_clean = df.loc[~any_invalid].copy()

    report["n_clean_rows"] = int(len(df_clean))
    report["n_bad_rows"] = int(len(df_bad))
    report["n_warning_rows_in_clean"] = int(df_clean["warning_reasons"].notna().sum())

    keep_cols_clean = ["date", "hour", "borough", "event_type", "n_events", "warning_reasons"]
    keep_cols_bad = keep_cols_clean + ["rejection_reasons"]
    df_clean = df_clean[[c for c in keep_cols_clean if c in df_clean.columns]]
    df_bad = df_bad[[c for c in keep_cols_bad if c in df_bad.columns]]

    return ValidationResult(df_clean=df_clean, df_bad=df_bad, report=report)


def write_outputs(result: ValidationResult, output_dir: Path, write_bad: bool) -> Path:
    clean_dir = output_dir / "clean"
    bad_dir = output_dir / "bad_rows"
    report_dir = output_dir / "reports"

    for folder in [clean_dir, bad_dir, report_dir]:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

    df_clean = result.df_clean.copy()
    df_clean["year"] = pd.to_datetime(df_clean["date"], errors="coerce").dt.year
    df_clean["month"] = pd.to_datetime(df_clean["date"], errors="coerce").dt.month

    for (year, month), group in df_clean.groupby(["year", "month"], dropna=True):
        out_fp = clean_dir / f"events_{int(year)}_{int(month):02d}.parquet"
        group.drop(columns=["year", "month"]).to_parquet(out_fp, index=False)

    if write_bad and not result.df_bad.empty:
        result.df_bad.to_parquet(bad_dir / "events_bad_rows.parquet", index=False)

    report_path = report_dir / "events.validation_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(result.report, f, ensure_ascii=False, indent=2)

    return report_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capa 1 EVENTOS: validación estructural y limpieza base.")
    p.add_argument("--input", type=str, default="", help="Directorio RAW. Default: data/external/events/raw")
    p.add_argument("--output", type=str, default="", help="Directorio salida. Default: data/external/events/validated")
    p.add_argument("--write-bad", action="store_true", help="Escribir bad_rows.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    in_dir = Path(args.input) if args.input else (obtener_ruta("data/external/events/raw"))
    out_dir = Path(args.output) if args.output else (obtener_ruta("data/external/events/validated"))
    out_reports = obtener_ruta("outputs/procesamiento") / "capa1_eventos"

    console.print(Panel.fit("[bold cyan]CAPA 1 — EVENTOS: VALIDACIÓN Y LIMPIEZA[/bold cyan]"))

    with console.status("[cyan]Leyendo eventos RAW..."):
        df_raw = read_raw_events(in_dir)
    with console.status("[cyan]Validando eventos..."):
        result = validate_events_df(df_raw)

    report_path = write_outputs(result, out_dir, write_bad=args.write_bad)

    table = Table(title="Capa1 Eventos — resumen", header_style="bold magenta")
    table.add_column("Rows", justify="right")
    table.add_column("Clean", justify="right")
    table.add_column("Bad", justify="right")
    table.add_column("Warn(clean)", justify="right")
    table.add_row(
        str(result.report["n_rows"]),
        str(result.report["n_clean_rows"]),
        str(result.report["n_bad_rows"]),
        str(result.report["n_warning_rows_in_clean"]),
    )
    console.print(table)

    out_reports.mkdir(parents=True, exist_ok=True)
    summary_path = out_reports / "capa1_eventos_validation_summary.json"
    summary = {"input": str(in_dir), "output": str(out_dir), **result.report}
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    console.print(f"[bold green]OK[/bold green] Reporte: {report_path}")
    console.print(f"[bold green]OK[/bold green] Resumen global: {summary_path}")


if __name__ == "__main__":
    main()
