"""
Ejercicio 1d: Mapa coroplético de poder adquisitivo por zona NYC.

Construye un dataset zona-año para pintar un mapa coroplético. El índice se calcula
con variables obligatorias derivadas de taxis y, si existen, se enriquece con
alquileres y restaurantes:

Variables base obligatorias:
- Propina media
- Porcentaje medio de propina
- Pasajeros medios
- Volumen de viajes

Variables opcionales:
- Precio medio de alquiler por zona
- Número de restaurantes por zona
- Número de cocinas distintas por zona

Salida:
  data/aggregated/ex1d/df_zone_socioeconomic/year=YYYY/*.parquet

Columnas principales:
- year
- pu_location_id
- avg_tip_amount
- avg_tip_pct
- avg_passenger_count
- trip_volume
- rent_price_zone
- n_restaurants_zone
- socioeconomic_score
- socioeconomic_label
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

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

from src.procesamiento.capa3.common.io import (
    cleanup_dataset_output,
    iter_month_partitions,
    write_partitioned_dataset,
)
from src.procesamiento.capa3.common.externals import (
    load_rent_zone_features_yearly,
    load_restaurants_zone_features_yearly,
)

console = Console()

TIPS_KEEP_COLS = [
    "year",
    "month",
    "pu_location_id",
    "passenger_count",
    "target_tip_amount",
    "target_tip_pct",
    "has_tip",
]

CORE_WEIGHTS = {
    "tip_amount_score": 0.30,
    "tip_pct_score": 0.20,
    "passenger_score": 0.15,
    "trip_volume_score": 0.35,
}

OPTIONAL_WEIGHTS = {
    "rent_score": 0.15,
    "restaurants_score": 0.07,
    "cuisines_score": 0.03,
}

LABELS = ["Bajo", "Medio-bajo", "Medio", "Medio-alto", "Alto"]


def _existing_columns(parquet_file: Path, wanted: Iterable[str]) -> list[str]:
    schema_cols = set(pq.ParquetFile(parquet_file).schema.names)
    return [c for c in wanted if c in schema_cols]


def read_ex1b_trip_level_tips(base_path: Path) -> pd.DataFrame:
    """
    Lee la salida particionada de EX1(b):
      data/aggregated/ex1b/df_trip_level_tips/year=YYYY/month=MM/*.parquet
    """
    base_path = Path(base_path).resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"No existe el directorio de EX1(b): {base_path}")

    parts: list[pd.DataFrame] = []
    files_seen = 0

    for _year, _month, files in iter_month_partitions(base_path):
        for fp in files:
            files_seen += 1
            cols_to_read = _existing_columns(fp, TIPS_KEEP_COLS)
            if not cols_to_read:
                continue
            parts.append(pd.read_parquet(fp, columns=cols_to_read))

    if not parts:
        raise FileNotFoundError(
            f"No se encontraron parquets válidos de EX1(b) en {base_path}. "
            f"Ficheros inspeccionados: {files_seen}"
        )

    return pd.concat(parts, ignore_index=True)


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_zone_socioeconomic_base(df_tips: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega los viajes de EX1(b) a nivel year + pu_location_id.
    """
    required = [
        "year",
        "pu_location_id",
        "passenger_count",
        "target_tip_amount",
        "target_tip_pct",
        "has_tip",
    ]
    missing = [c for c in required if c not in df_tips.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en EX1(b): {missing}")

    df = _coerce_numeric(df_tips, required)
    df = df.dropna(subset=["year", "pu_location_id"])
    df = df[df["pu_location_id"] > 0]

    # Evitamos que valores claramente anómalos deformen el índice.
    df = df[df["target_tip_amount"].isna() | (df["target_tip_amount"] >= 0)]
    df = df[df["target_tip_pct"].isna() | ((df["target_tip_pct"] >= 0) & (df["target_tip_pct"] <= 100))]
    df = df[df["passenger_count"].isna() | ((df["passenger_count"] >= 0) & (df["passenger_count"] <= 8))]

    agg = (
        df.groupby(["year", "pu_location_id"], as_index=False)
        .agg(
            trip_volume=("pu_location_id", "size"),
            avg_tip_amount=("target_tip_amount", "mean"),
            median_tip_amount=("target_tip_amount", "median"),
            avg_tip_pct=("target_tip_pct", "mean"),
            avg_passenger_count=("passenger_count", "mean"),
            share_has_tip=("has_tip", "mean"),
        )
    )

    agg["year"] = pd.to_numeric(agg["year"], errors="coerce").astype("Int32")
    agg["pu_location_id"] = pd.to_numeric(agg["pu_location_id"], errors="coerce").astype("Int32")
    agg["trip_volume"] = pd.to_numeric(agg["trip_volume"], errors="coerce").astype("Int64")

    float_cols = [
        "avg_tip_amount",
        "median_tip_amount",
        "avg_tip_pct",
        "avg_passenger_count",
        "share_has_tip",
    ]
    for col in float_cols:
        agg[col] = pd.to_numeric(agg[col], errors="coerce").astype("float32")

    return agg.sort_values(["year", "pu_location_id"]).reset_index(drop=True)


def add_minmax_by_year(df: pd.DataFrame, source_col: str, output_col: str) -> pd.DataFrame:
    """
    Normaliza una columna a [0, 1] dentro de cada año.
    Si en un año todos los valores son iguales, asigna 0.5 a las filas no nulas.
    """
    out = df.copy()

    def _scale(series: pd.Series) -> pd.Series:
        x = pd.to_numeric(series, errors="coerce")
        non_null = x.dropna()

        if non_null.empty:
            return pd.Series(np.nan, index=series.index)

        mn = non_null.min()
        mx = non_null.max()

        if mx == mn:
            return pd.Series(np.where(x.notna(), 0.5, np.nan), index=series.index)

        return (x - mn) / (mx - mn)

    out[output_col] = (
        out.groupby("year", group_keys=False)[source_col]
        .apply(_scale)
        .astype("float32")
    )
    return out


def _weighted_score_available_features(
    df: pd.DataFrame,
    weights: dict[str, float],
    output_col: str,
) -> pd.DataFrame:
    """
    Calcula media ponderada por fila usando solo features disponibles en esa fila.

    Esto es mejor que rellenar nulos con 0, porque un barrio sin dato externo no debe
    ser penalizado automáticamente como si tuviera alquiler/restaurantes mínimos.
    """
    out = df.copy()
    numerator = pd.Series(0.0, index=out.index, dtype="float64")
    denominator = pd.Series(0.0, index=out.index, dtype="float64")

    for col, weight in weights.items():
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        mask = values.notna()
        numerator.loc[mask] += values.loc[mask] * weight
        denominator.loc[mask] += weight

    out[output_col] = np.where(denominator > 0, numerator / denominator, np.nan)
    out[output_col] = pd.to_numeric(out[output_col], errors="coerce").astype("float32")
    return out


def _add_score_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def label_score(x: float) -> str:
        if pd.isna(x):
            return "Sin datos"
        if x <= 0.20:
            return "Bajo"
        if x <= 0.40:
            return "Medio-bajo"
        if x <= 0.60:
            return "Medio"
        if x <= 0.80:
            return "Medio-alto"
        return "Alto"

    out["socioeconomic_label"] = out["socioeconomic_score"].apply(label_score)
    return out


def build_socioeconomic_score(
    df_base: pd.DataFrame,
    df_rent: pd.DataFrame | None = None,
    df_restaurants: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Construye el índice compuesto de poder adquisitivo.

    Diseño del índice:
    - Primero calcula un score base obligatorio con taxis.
    - Después calcula un score opcional con alquiler/restaurantes si existen.
    - Si hay variables opcionales disponibles, combina 75% taxi + 25% externas.
    - Finalmente reescala por año para que el mapa sea comparable dentro de cada año.
    """
    df = df_base.copy()

    if df_rent is not None and not df_rent.empty:
        df = df.merge(df_rent, on=["year", "pu_location_id"], how="left")
    else:
        df["rent_price_zone"] = np.nan

    if df_restaurants is not None and not df_restaurants.empty:
        df = df.merge(df_restaurants, on=["year", "pu_location_id"], how="left")
    else:
        df["n_restaurants_zone"] = np.nan
        df["n_cuisines_zone"] = np.nan

    numeric_cols = [
        "avg_tip_amount",
        "avg_tip_pct",
        "avg_passenger_count",
        "trip_volume",
        "rent_price_zone",
        "n_restaurants_zone",
        "n_cuisines_zone",
    ]
    df = _coerce_numeric(df, numeric_cols)

    # Normalizaciones por año.
    df = add_minmax_by_year(df, "avg_tip_amount", "tip_amount_score")
    df = add_minmax_by_year(df, "avg_tip_pct", "tip_pct_score")
    df = add_minmax_by_year(df, "avg_passenger_count", "passenger_score")
    df = add_minmax_by_year(df, "trip_volume", "trip_volume_score")
    df = add_minmax_by_year(df, "rent_price_zone", "rent_score")
    df = add_minmax_by_year(df, "n_restaurants_zone", "restaurants_score")
    df = add_minmax_by_year(df, "n_cuisines_zone", "cuisines_score")

    df = _weighted_score_available_features(df, CORE_WEIGHTS, "core_taxi_score")
    df = _weighted_score_available_features(df, OPTIONAL_WEIGHTS, "external_score")

    has_external = df["external_score"].notna()
    df["socioeconomic_score_raw"] = df["core_taxi_score"]
    df.loc[has_external, "socioeconomic_score_raw"] = (
        0.75 * df.loc[has_external, "core_taxi_score"]
        + 0.25 * df.loc[has_external, "external_score"]
    )

    df = add_minmax_by_year(df, "socioeconomic_score_raw", "socioeconomic_score")
    df = _add_score_labels(df)

    ordered_cols = [
        "year",
        "pu_location_id",
        "avg_tip_amount",
        "median_tip_amount",
        "avg_tip_pct",
        "avg_passenger_count",
        "share_has_tip",
        "trip_volume",
        "rent_price_zone",
        "n_restaurants_zone",
        "n_cuisines_zone",
        "tip_amount_score",
        "tip_pct_score",
        "passenger_score",
        "trip_volume_score",
        "rent_score",
        "restaurants_score",
        "cuisines_score",
        "core_taxi_score",
        "external_score",
        "socioeconomic_score_raw",
        "socioeconomic_score",
        "socioeconomic_label",
    ]
    existing = [c for c in ordered_cols if c in df.columns]
    rest = [c for c in df.columns if c not in existing]

    return df[existing + rest].sort_values(["year", "pu_location_id"]).reset_index(drop=True)


def _print_config(args: argparse.Namespace, tips_dir: Path, rent_dir: Path, restaurants_dir: Path, out_base: Path) -> None:
    table = Table(show_header=True, header_style="bold white", title="Configuración EX1(d)")
    table.add_column("Parámetro", style="bold cyan")
    table.add_column("Valor")
    table.add_row("tips_dir", str(tips_dir))
    table.add_row("rent_dir", str(rent_dir))
    table.add_row("restaurants_dir", str(restaurants_dir))
    table.add_row("out_dir", str(out_base))
    table.add_row("min_date", args.min_date)
    table.add_row("max_date", args.max_date)
    table.add_row("mode", args.mode)
    console.print(table)


def _print_summary(df: pd.DataFrame, out_dir: Path) -> None:
    table = Table(show_header=True, header_style="bold magenta", title="Resumen EX1(d)")
    table.add_column("Métrica", style="bold white")
    table.add_column("Valor", justify="right")
    table.add_row("Filas output", f"{len(df):,}")
    table.add_row("Zonas únicas", f"{df['pu_location_id'].nunique():,}")
    table.add_row("Años únicos", f"{df['year'].nunique():,}")
    table.add_row("Score mínimo", f"{df['socioeconomic_score'].min():.4f}")
    table.add_row("Score máximo", f"{df['socioeconomic_score'].max():.4f}")
    table.add_row("Salida", str(out_dir))
    console.print(table)

    dist = (
        df.groupby(["year", "socioeconomic_label"], dropna=False)
        .size()
        .reset_index(name="n_zonas")
        .sort_values(["year", "socioeconomic_label"])
    )
    label_table = Table(show_header=True, header_style="bold blue", title="Distribución de etiquetas")
    label_table.add_column("Año", justify="right")
    label_table.add_column("Etiqueta")
    label_table.add_column("Zonas", justify="right")
    for _, row in dist.iterrows():
        label_table.add_row(str(row["year"]), str(row["socioeconomic_label"]), str(row["n_zonas"]))
    console.print(label_table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye índice de poder adquisitivo por zona para mapa coroplético."
    )
    parser.add_argument(
        "--tips-dir",
        default=str(obtener_ruta("data/aggregated/ex1b/df_trip_level_tips")),
        help="Ruta a salida EX1(b) trip level tips.",
    )
    parser.add_argument(
        "--rent-dir",
        default=str(obtener_ruta("data/external/rent/aggregated")),
        help="Ruta a renta agregada por zona.",
    )
    parser.add_argument(
        "--restaurants-dir",
        default=str(obtener_ruta("data/external/restaurants/aggregated")),
        help="Ruta a restaurantes agregados por zona.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(obtener_ruta("data/aggregated/ex1d")),
        help="Directorio base de salida.",
    )
    parser.add_argument("--min-date", default="2023-01-01", help="Fecha mínima para externas.")
    parser.add_argument("--max-date", default="2025-12-31", help="Fecha máxima para externas.")
    parser.add_argument(
        "--mode",
        choices=["overwrite", "append"],
        default="overwrite",
        help="overwrite borra salidas previas; append conserva.",
    )
    return parser.parse_args()


def main() -> None:
    console.print(Panel.fit("[bold cyan]CAPA 3 EX1(d) - PODER ADQUISITIVO POR ZONA[/bold cyan]"))

    args = parse_args()
    tips_dir = Path(args.tips_dir).resolve()
    rent_dir = Path(args.rent_dir).resolve()
    restaurants_dir = Path(args.restaurants_dir).resolve()
    out_base = Path(args.out_dir).resolve()
    out_dir = out_base / "df_zone_socioeconomic"

    _print_config(args, tips_dir, rent_dir, restaurants_dir, out_base)

    if args.mode == "overwrite":
        cleanup_dataset_output(out_base, "df_zone_socioeconomic", label="EX1(d)")

    out_dir.mkdir(parents=True, exist_ok=True)

    with console.status("[cyan]Leyendo EX1(b)...[/cyan]"):
        df_tips = read_ex1b_trip_level_tips(tips_dir)
    console.print(f"[cyan]EX1(b) leído: {len(df_tips):,} filas[/cyan]")

    with console.status("[cyan]Agregando viajes por zona-año...[/cyan]"):
        df_base = build_zone_socioeconomic_base(df_tips)
    console.print(f"[cyan]Base zona-año: {len(df_base):,} filas[/cyan]")

    with console.status("[cyan]Leyendo alquileres y restaurantes...[/cyan]"):
        df_rent = load_rent_zone_features_yearly(
            rent_base=rent_dir,
            min_date=args.min_date,
            max_date=args.max_date,
        )
        df_restaurants = load_restaurants_zone_features_yearly(
            restaurants_base=restaurants_dir,
            min_date=args.min_date,
            max_date=args.max_date,
        )
    console.print(f"[cyan]Rent yearly: {len(df_rent):,} filas[/cyan]")
    console.print(f"[cyan]Restaurants yearly: {len(df_restaurants):,} filas[/cyan]")

    with console.status("[cyan]Calculando índice compuesto...[/cyan]"):
        df_final = build_socioeconomic_score(
            df_base=df_base,
            df_rent=df_rent,
            df_restaurants=df_restaurants,
        )

    write_partitioned_dataset(df_final, out_dir, partition_cols=["year"])
    _print_summary(df_final, out_dir)

    console.print("[bold green]✅ OK[/bold green] EX1(d) completado")


if __name__ == "__main__":
    main()
