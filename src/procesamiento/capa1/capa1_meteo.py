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

REQUIRED_COLUMNS = ["date", "hour"]
NUMERIC_COLUMNS = ["temp_c", "precip_mm", "rain_mm", "snowfall_mm", "wind_kmh"]
MIN_YEAR = 2023
MAX_YEAR = 2025


def safe_remove_dir(path: Path) -> None:
    """Borra una carpeta si existe. Si no existe, no hace nada."""
    path = Path(path).resolve()
    if not path.exists():
        return

    try:
        shutil.rmtree(path)
    except PermissionError as exc:
        raise PermissionError(
            f"No se pudo borrar la carpeta '{path}'. "
            "Probablemente está abierta en VS Code, en el explorador "
            "o bloqueada por otro proceso."
        ) from exc


def read_raw_meteo(in_dir: Path, base_name: str = "meteo_hourly_nyc") -> pd.DataFrame:
    in_dir = Path(in_dir).resolve()
    if not in_dir.exists():
        raise FileNotFoundError(f"No existe el directorio RAW: {in_dir}")

    parquet_path = in_dir / f"{base_name}.parquet"
    csv_path = in_dir / f"{base_name}.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    files = sorted(in_dir.glob("meteo_*.parquet"))
    if not files:
        files = sorted(in_dir.glob("meteo_*.parquet.gz"))
    if not files:
        files = sorted(in_dir.glob("meteo_*.csv"))
    if not files:
        files = sorted(in_dir.glob("meteo_*.csv.gz"))

    if not files:
        raise FileNotFoundError(
            f"No encuentro {base_name}.parquet/.csv ni meteo_*.parquet/.csv en {in_dir}"
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


def _mode_or_na(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return pd.NA
    m = s.mode()
    if m.empty:
        return pd.NA
    return m.iloc[0]


@dataclass
class ValidationResult:
    df_clean: pd.DataFrame
    df_bad: pd.DataFrame
    report: Dict[str, Any]


def validate_meteo_df(df: pd.DataFrame) -> ValidationResult:
    df = df.copy()
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    if "weather_code" not in df.columns:
        df["weather_code"] = pd.NA
    for c in NUMERIC_COLUMNS:
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

    # numéricos
    for c in NUMERIC_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # weather_code
    weather_num = pd.to_numeric(df["weather_code"], errors="coerce")
    inv_wc_type = df["weather_code"].notna() & weather_num.isna()
    inv_wc_decimal = weather_num.notna() & ((weather_num % 1) != 0)
    weather_num = weather_num.where(~inv_wc_decimal).astype("Int64")
    invalid_masks["weather_code__invalid_type"] = inv_wc_type
    invalid_masks["weather_code__invalid_decimal"] = inv_wc_decimal
    df["weather_code"] = weather_num

    # reglas de plausibilidad dura
    invalid_masks["precip_mm__negative"] = df["precip_mm"].notna() & (df["precip_mm"] < 0)
    invalid_masks["rain_mm__negative"] = df["rain_mm"].notna() & (df["rain_mm"] < 0)
    invalid_masks["snowfall_mm__negative"] = df["snowfall_mm"].notna() & (df["snowfall_mm"] < 0)
    invalid_masks["wind_kmh__negative"] = df["wind_kmh"].notna() & (df["wind_kmh"] < 0)

    # warnings de extremos
    warning_masks["temp_c__extreme"] = df["temp_c"].notna() & ((df["temp_c"] < -50) | (df["temp_c"] > 60))
    warning_masks["wind_kmh__extreme"] = df["wind_kmh"].notna() & (df["wind_kmh"] > 180)
    warning_masks["row__duplicate_exact"] = df.duplicated(keep="first")

    report["invalid_counts"] = {k: int(v.fillna(False).sum()) for k, v in invalid_masks.items()}
    report["warning_counts"] = {k: int(v.fillna(False).sum()) for k, v in warning_masks.items()}

    any_invalid = pd.Series(False, index=df.index)
    for mask in invalid_masks.values():
        any_invalid |= mask.fillna(False)

    df["warning_reasons"] = _build_reason_column(warning_masks, df.index)
    df_bad = df.loc[any_invalid].copy()
    df_bad["rejection_reasons"] = _build_reason_column(invalid_masks, df_bad.index)
    df_clean = df.loc[~any_invalid].copy()

    # consolidar duplicados date+hour en clean
    before_rows = len(df_clean)
    if not df_clean.empty:
        agg_map = {
            "temp_c": "mean",
            "precip_mm": "sum",
            "rain_mm": "sum",
            "snowfall_mm": "sum",
            "wind_kmh": "mean",
            "weather_code": _mode_or_na,
        }
        df_clean = (
            df_clean.groupby(["date", "hour"], as_index=False)
            .agg(agg_map)
            .sort_values(["date", "hour"])
            .reset_index(drop=True)
        )
        df_clean["warning_reasons"] = pd.NA

    report["n_clean_rows"] = int(len(df_clean))
    report["n_bad_rows"] = int(len(df_bad))
    report["n_warning_rows_in_clean"] = (
        int(df_clean["warning_reasons"].notna().sum()) if "warning_reasons" in df_clean.columns else 0
    )
    report["n_clean_rows_collapsed_duplicates"] = int(before_rows - len(df_clean))

    keep_cols_clean = [
        "date",
        "hour",
        "temp_c",
        "precip_mm",
        "rain_mm",
        "snowfall_mm",
        "wind_kmh",
        "weather_code",
        "warning_reasons",
    ]
    keep_cols_bad = keep_cols_clean + ["rejection_reasons"]
    df_clean = df_clean[[c for c in keep_cols_clean if c in df_clean.columns]]
    df_bad = df_bad[[c for c in keep_cols_bad if c in df_bad.columns]]

    return ValidationResult(df_clean=df_clean, df_bad=df_bad, report=report)


def write_outputs(
    result: ValidationResult,
    output_dir: Path,
    write_bad: bool,
    overwrite: bool = False,
) -> Path:
    clean_dir = output_dir / "clean"
    bad_dir = output_dir / "bad_rows"
    report_dir = output_dir / "reports"

    for folder in [clean_dir, bad_dir, report_dir]:
        if overwrite and folder.exists():
            safe_remove_dir(folder)
        folder.mkdir(parents=True, exist_ok=True)

    df_clean = result.df_clean.copy()
    df_clean["year"] = pd.to_datetime(df_clean["date"], errors="coerce").dt.year
    df_clean["month"] = pd.to_datetime(df_clean["date"], errors="coerce").dt.month

    for (year, month), group in df_clean.groupby(["year", "month"], dropna=True):
        out_fp = clean_dir / f"meteo_{int(year)}_{int(month):02d}.parquet"
        group.drop(columns=["year", "month"]).to_parquet(out_fp, index=False)

    if write_bad and not result.df_bad.empty:
        result.df_bad.to_parquet(bad_dir / "meteo_bad_rows.parquet", index=False)

    report_path = report_dir / "meteo.validation_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(result.report, f, ensure_ascii=False, indent=2)

    return report_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capa 1 METEO: validación estructural y limpieza base.")
    p.add_argument("--input", type=str, default="", help="Directorio RAW. Default: data/external/meteo/raw")
    p.add_argument("--output", type=str, default="", help="Directorio salida. Default: data/external/meteo/validated")
    p.add_argument("--write-bad", action="store_true", help="Escribir bad_rows.")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Si se activa, borra las salidas previas antes de escribir.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    in_dir = Path(args.input) if args.input else (obtener_ruta("data/external/meteo/raw"))
    out_dir = Path(args.output) if args.output else (obtener_ruta("data/external/meteo/validated"))
    out_reports = obtener_ruta("outputs/procesamiento") / "capa1_meteo"

    console.print(Panel.fit("[bold cyan]CAPA 1 — METEO: VALIDACIÓN Y LIMPIEZA[/bold cyan]"))

    with console.status("[cyan]Leyendo meteo RAW..."):
        df_raw = read_raw_meteo(in_dir)
    with console.status("[cyan]Validando meteo..."):
        result = validate_meteo_df(df_raw)

    if args.overwrite:
        console.print("[yellow]Modo overwrite:[/yellow] se limpiarán las salidas previas de meteo.")
    else:
        console.print("[yellow]Modo append/conservador:[/yellow] no se borrarán salidas previas.")

    report_path = write_outputs(
        result,
        out_dir,
        write_bad=args.write_bad,
        overwrite=args.overwrite,
    )

    table = Table(title="Capa1 Meteo — resumen", header_style="bold magenta")
    table.add_column("Rows", justify="right")
    table.add_column("Clean", justify="right")
    table.add_column("Bad", justify="right")
    table.add_column("Collapsed dup", justify="right")
    table.add_row(
        str(result.report["n_rows"]),
        str(result.report["n_clean_rows"]),
        str(result.report["n_bad_rows"]),
        str(result.report["n_clean_rows_collapsed_duplicates"]),
    )
    console.print(table)

    out_reports.mkdir(parents=True, exist_ok=True)
    summary_path = out_reports / "capa1_meteo_validation_summary.json"
    summary = {"input": str(in_dir), "output": str(out_dir), **result.report}
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    console.print(f"[bold green]OK[/bold green] Reporte: {report_path}")
    console.print(f"[bold green]OK[/bold green] Resumen global: {summary_path}")


if __name__ == "__main__":
    main()