"""
Ejercicio 1c: Identificar patrones de demanda por zona-hora

Clasifica demanda en niveles según zona y franja horaria.
Dimensiones de clasificación:
  - Nivel de demanda: Bajo / Medio / Alto (terciles POR ZONA)
  - Estabilidad: Baja variabilidad = predecible, Alta variabilidad = volátil

Ejemplo de salida:
  zona=123, hora=14 -> demand_level=Alta, stability=Predecible (baja variabilidad)
  zona=456, hora=22 -> demand_level=Media, stability=Volátil (alta variabilidad)

Genera: data/aggregated/ex1c/df_demand_patterns/
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import obtener_ruta

console = Console()

def read_zone_hour_day_global(base_path: Path) -> pd.DataFrame:
    """
    Lee df_zone_hour_day_global particionado por fecha.
    Asume estructura tipo Spark: date=YYYY-MM-DD/part_*.parquet
    """
    base_path = Path(base_path).resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"No existe: {base_path}")
    
    files = list(base_path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No hay parquets en: {base_path}")
    
    console.print(f"[cyan]Leyendo {len(files)} parquets de df_zone_hour_day_global...[/cyan]")
    dfs = [pd.read_parquet(fp) for fp in files]
    df = pd.concat(dfs, ignore_index=True)
    
    return df


def build_demand_patterns(
    df: pd.DataFrame,
    min_trips_threshold: int = 30,
) -> pd.DataFrame:
    """
    Construye patrones de demanda por zona-hora.
    
    Pasos:
    1. Agrupa por zona y hora (promediando sobre fechas)
    2. Calcula media y std de num_trips
    3. Clasifica demanda en terciles POR ZONA (Baja/Media/Alta)
    4. Clasifica estabilidad según CV (coeficiente de variación)
    5. Retorna tabla final con patrones
    """
    required_cols = ["pu_location_id", "hour", "num_trips", "avg_price", "std_price"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")
    
    # Convertir tipos
    df = df.copy()
    df["pu_location_id"] = pd.to_numeric(df["pu_location_id"], errors="coerce").astype("Int64")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype("Int64")
    df["num_trips"] = pd.to_numeric(df["num_trips"], errors="coerce").astype("Int64")
    df["avg_price"] = pd.to_numeric(df["avg_price"], errors="coerce")
    df["std_price"] = pd.to_numeric(df["std_price"], errors="coerce")
    
    # Filtrar mínimo de trips
    df = df[df["num_trips"] >= min_trips_threshold]
    
    if df.empty:
        raise ValueError(f"DataFrame vacío después de filtrar min_trips >= {min_trips_threshold}")
    
    # =========================================================================
    # Paso 1: Agregar por zona-hora (promediar sobre fechas)
    # =========================================================================
    console.print("[cyan]Agregando por zona-hora...[/cyan]")
    
    agg_data = df.groupby(["pu_location_id", "hour"], as_index=False).agg(
        num_trips_avg=("num_trips", "mean"),
        num_trips_std=("num_trips", "std"),
        num_trips_min=("num_trips", "min"),
        num_trips_max=("num_trips", "max"),
        num_trips_count=("num_trips", "count"),  # cuántas fechas hay para esta zona-hora
        avg_price_mean=("avg_price", "mean"),
        std_price_mean=("std_price", "mean"),
    )
    
    # Llenar NaN en desviación estándar (solo hay un valor en esa zona-hora)
    agg_data["num_trips_std"] = agg_data["num_trips_std"].fillna(0.0)
    
    # =========================================================================
    # Paso 2: Clasificación de DEMANDA por ZONA (terciles locales)
    # =========================================================================
    console.print("[cyan]Clasificando demanda (terciles por zona)...[/cyan]")
    
    # Agrupar por zona para calcular percentiles
    agg_data["demand_level"] = agg_data.groupby("pu_location_id")["num_trips_avg"].transform(
        lambda x: pd.qcut(x, q=3, labels=["Baja", "Media", "Alta"], duplicates="drop")
    )
    
    # =========================================================================
    # Paso 3: Clasificación de ESTABILIDAD (variabilidad)
    # =========================================================================
    console.print("[cyan]Clasificando estabilidad...[/cyan]")
    
    # Coeficiente de variación (CV = std / mean). Si CV es muy bajo, es predecible
    agg_data["cv"] = np.where(
        agg_data["num_trips_avg"] > 0,
        agg_data["num_trips_std"] / agg_data["num_trips_avg"],
        0.0
    )
    
    # Clasificar por percentiles globales de CV
    # Bajo CV = predecible, Alto CV = volátil
    cv_p33 = agg_data["cv"].quantile(0.33)
    cv_p66 = agg_data["cv"].quantile(0.66)
    
    agg_data["stability"] = (
        agg_data["cv"].apply(
            lambda x: "Predecible" if x <= cv_p33 
                      else ("Variable" if x <= cv_p66 else "Volátil")
        )
    )
    
    # =========================================================================
    # Paso 4: Score de "Interés operacional"
    # =========================================================================
    # Alta demanda + baja variabilidad = máxima prioridad (hotspot confiable)
    # Baja demanda + alta variabilidad = mínima prioridad
    
    # Mapear demand_level a número
    demand_score = agg_data["demand_level"].map({"Baja": 1, "Media": 2, "Alta": 3})
    stability_score = agg_data["stability"].map({"Volátil": 1, "Variable": 2, "Predecible": 3})
    
    agg_data["operational_priority"] = demand_score * stability_score
    agg_data["operational_priority_label"] = (
        agg_data["operational_priority"].apply(
            lambda x: "Crítica" if x >= 7 
                      else ("Alta" if x >= 4 else "Baja")
        )
    )
    
    # =========================================================================
    # Paso 5: Columnas finales
    # =========================================================================
    final_cols = [
        "pu_location_id",
        "hour",
        "num_trips_avg",
        "num_trips_std",
        "num_trips_min",
        "num_trips_max",
        "num_trips_count",
        "avg_price_mean",
        "std_price_mean",
        "demand_level",
        "stability",
        "cv",
        "operational_priority",
        "operational_priority_label",
    ]
    
    return agg_data[final_cols]


def save_patterns(
    df: pd.DataFrame,
    out_base: Path,
    mode: str = "overwrite",
) -> None:
    """
    Guarda patrones particionados por zona.
    """
    out_base = Path(out_base).resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    
    dataset_dir = out_base / "df_demand_patterns"
    
    if mode == "overwrite" and dataset_dir.exists():
        import shutil
        console.print(f"[yellow]Borrando {dataset_dir}...[/yellow]")
        shutil.rmtree(dataset_dir)
    
    console.print(f"[cyan]Guardando patrones en {dataset_dir}...[/cyan]")
    
    # Particionar por zona para lectura rápida
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for zone, g in df.groupby("pu_location_id"):
        zone_dir = dataset_dir / f"pu_location_id={int(zone)}"
        zone_dir.mkdir(parents=True, exist_ok=True)
        g.drop(columns=["pu_location_id"]).to_parquet(
            zone_dir / "patterns.parquet",
            index=False,
            engine="pyarrow"
        )
    
    # Resumen estadístico
    summary = Table(show_header=True, header_style="bold magenta", title="Resumen EX1(c) Patrones")
    summary.add_column("Métrica", style="bold white")
    summary.add_column("Valor", justify="right")
    
    summary.add_row("Total registros", f"{len(df):,}")
    summary.add_row("Zonas únicas", f"{df['pu_location_id'].nunique():,}")
    summary.add_row("Horas únicas", f"{df['hour'].nunique():,}")
    summary.add_row("", "")
    
    # Distribución de demand_level
    for level in ["Baja", "Media", "Alta"]:
        count = (df["demand_level"] == level).sum()
        summary.add_row(f"Demanda {level}", f"{count:,} ({count/len(df)*100:.1f}%)")
    
    summary.add_row("", "")
    
    # Distribución de stability
    for stab in ["Predecible", "Variable", "Volátil"]:
        count = (df["stability"] == stab).sum()
        summary.add_row(f"Estabilidad {stab}", f"{count:,} ({count/len(df)*100:.1f}%)")
    
    summary.add_row("", "")
    
    # Prioridad operacional
    for prio in ["Crítica", "Alta", "Baja"]:
        count = (df["operational_priority_label"] == prio).sum()
        summary.add_row(f"Prioridad {prio}", f"{count:,} ({count/len(df)*100:.1f}%)")
    
    console.print(summary)
    console.print(f"[bold green]Patrones guardados en {dataset_dir}[/bold green]")


def main() -> None:
    """
    Ejecuta análisis completo de patrones de demanda por zona-hora.
    """
    console.print(Panel.fit("[bold cyan]CAPA 3 EX1(c) - PATRONES DE DEMANDA ZONA-HORA[/bold cyan]"))
    
    p = argparse.ArgumentParser(
        description="Clasifica demanda en bajo/medio/alto por zona-hora con análisis de estabilidad."
    )
    p.add_argument(
        "--in-dir",
        default=str(obtener_ruta("data/aggregated/df_zone_hour_day_global")),
        help="Path a df_zone_hour_day_global",
    )
    p.add_argument(
        "--out-dir",
        default=str(obtener_ruta("data/aggregated/ex1c")),
        help="Directorio de salida",
    )
    p.add_argument(
        "--min-trips",
        type=int,
        default=30,
        help="Mínimo de trips para considerar una zona-hora (default: 30)",
    )
    p.add_argument(
        "--mode",
        choices=["overwrite", "append"],
        default="overwrite",
        help="overwrite o append",
    )
    args = p.parse_args()
    
    in_base = Path(args.in_dir).resolve()
    out_base = Path(args.out_dir).resolve()
    
    cfg = Table(show_header=True, header_style="bold white", title="Configuración EX1(c)")
    cfg.add_column("Parámetro", style="bold cyan")
    cfg.add_column("Valor")
    cfg.add_row("in_dir", str(in_base))
    cfg.add_row("out_dir", str(out_base))
    cfg.add_row("min_trips", str(args.min_trips))
    cfg.add_row("mode", args.mode)
    console.print(cfg)
    
    # Ejecutar
    with console.status("[cyan]Procesando patrones de demanda...[/cyan]"):
        df = read_zone_hour_day_global(in_base)
        console.print(f"[cyan]Leído: {len(df):,} filas de zona-hora-día[/cyan]")
        
        patterns = build_demand_patterns(df, min_trips_threshold=args.min_trips)
        console.print(f"[cyan]Patrones construidos: {len(patterns):,} combinaciones zona-hora[/cyan]")
    
    save_patterns(patterns, out_base, mode=args.mode)
    console.print("[bold green]OK[/bold green] EX1(c) completado")


if __name__ == "__main__":
    main()
