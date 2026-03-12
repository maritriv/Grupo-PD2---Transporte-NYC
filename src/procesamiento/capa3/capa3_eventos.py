# src/procesamiento/capa3/capa3_eventos.py
from __future__ import annotations

import argparse
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from config.settings import obtener_ruta  # type: ignore
except Exception:
    def obtener_ruta(p: str) -> Path:
        return Path(p)

DEBUG = False  # True para ver previews
MIN_YEAR = 2023
MAX_YEAR = 2025


# -----------------------------------------------------------------------------
# Lectura capa 2 eventos (archivos mensuales: events_YYYY-MM.parquet)
# -----------------------------------------------------------------------------
def read_layer2_events(layer2_path: Path, base_name: str = "events_daily_borough_type") -> pd.DataFrame:
    """
    Lee capa 2 de eventos en dos formatos posibles:

    1) Formato nuevo: archivos mensuales
        - events_YYYY-MM.parquet (y opcionalmente events_YYYY-MM.parquet.gz)

    2) Formato antiguo:
        - {base_name}.parquet o {base_name}.csv
    """
    layer2_path = Path(layer2_path).resolve()
    if not layer2_path.exists():
        raise FileNotFoundError(f"No existe {layer2_path}")

    # 1) Mensual (parquet)
    files = sorted(layer2_path.glob("events_*.parquet"))
    if not files:
        files = sorted(layer2_path.glob("events_*.parquet.gz"))

    if files:
        print(f"[INFO] Leyendo {len(files)} archivos Capa 2 (parquet)...")
        dfs = [pd.read_parquet(fp) for fp in files]
        return pd.concat(dfs, ignore_index=True)

    # 2) Antiguo
    parquet_path = layer2_path / f"{base_name}.parquet"
    csv_path = layer2_path / f"{base_name}.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(
        f"No encuentro events_*.parquet en {layer2_path} "
        f"ni {base_name}.parquet / {base_name}.csv."
    )


def _parse_date(s: str | None) -> pd.Timestamp | None:
    if s is None:
        return None
    return pd.to_datetime(s, format="%Y-%m-%d", errors="raise")


def filter_by_range(df: pd.DataFrame, date_from: str | None, date_to: str | None) -> pd.DataFrame:
    """
    Filtra por rango inclusive [date_from, date_to] usando la columna 'date'.
    """
    if date_from is None and date_to is None:
        return df

    dt = pd.to_datetime(df["date"], errors="coerce")
    out = df.copy()
    out["_date_ts"] = dt

    if date_from is not None:
        dfrom = _parse_date(date_from)
        out = out[out["_date_ts"] >= dfrom]

    if date_to is not None:
        dto = _parse_date(date_to)
        out = out[out["_date_ts"] <= dto]

    return out.drop(columns=["_date_ts"])


# -----------------------------------------------------------------------------
# Construcción capa 3
# -----------------------------------------------------------------------------
def build_layer3_events(df2: pd.DataFrame, min_events_hour_day: int = 1):
    """
    Produce 4 tablas agregadas:
      - df_borough_hour_day: borough x date x hour
      - df_daily_borough: borough x date
      - df_type_daily_borough: borough x date x event_type
      - df_hourly_pattern: borough x hour
    """
    df = df2.copy()

    # Tipos/higiene mínima
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["hour"] = pd.to_numeric(df.get("hour"), errors="coerce").astype("Int64")
    df["borough"] = df.get("borough")
    df["event_type"] = df.get("event_type")
    df["n_events"] = pd.to_numeric(df.get("n_events"), errors="coerce").astype("Int64")

    df = df.dropna(subset=["date", "hour", "borough", "event_type", "n_events"])
    date_ts = pd.to_datetime(df["date"], errors="coerce")
    df = df[(date_ts.dt.year >= MIN_YEAR) & (date_ts.dt.year <= MAX_YEAR)]
    df = df[(df["hour"] >= 0) & (df["hour"] <= 23)]
    df = df[df["n_events"] >= 0]

    # Limpieza strings
    df["borough"] = df["borough"].astype(str).str.strip().str.title()
    df["event_type"] = df["event_type"].astype(str).str.strip()

    # 1) borough + date + hour
    df_borough_hour_day = (
        df.groupby(["borough", "date", "hour"], as_index=False)
          .agg(
              n_events=("n_events", "sum"),
              n_event_types=("event_type", "nunique"),
          )
    )
    df_borough_hour_day = df_borough_hour_day[df_borough_hour_day["n_events"] >= min_events_hour_day]

    # 2) daily por borough
    df_daily_borough = (
        df_borough_hour_day.groupby(["borough", "date"], as_index=False)
          .agg(
              n_events=("n_events", "sum"),
              n_event_types_sum=("n_event_types", "sum"),
              max_events_in_an_hour=("n_events", "max"),
          )
    )

    # 3) ranking tipos por borough + date
    df_type_daily_borough = (
        df.groupby(["borough", "date", "event_type"], as_index=False)
          .agg(n_events=("n_events", "sum"))
    )

    # 4) patrón horario medio por borough
    df_hourly_pattern = (
        df_borough_hour_day.groupby(["borough", "hour"], as_index=False)
          .agg(
              avg_events=("n_events", "mean"),
              std_events=("n_events", "std"),
              n_days=("date", "nunique"),
          )
    )
    df_hourly_pattern["std_events"] = df_hourly_pattern["std_events"].fillna(0.0)

    if DEBUG:
        print(df_borough_hour_day.head(10))
        print(df_daily_borough.head(10))
        print(df_type_daily_borough.head(10))
        print(df_hourly_pattern.head(10))

    return df_borough_hour_day, df_daily_borough, df_type_daily_borough, df_hourly_pattern


# -----------------------------------------------------------------------------
# Guardado
# -----------------------------------------------------------------------------
def _safe_partition_value(v) -> str:
    # Evita caracteres raros en nombres de carpeta
    s = str(v)
    s = s.replace("/", "-").replace("\\", "-")
    s = s.replace(":", "-")
    s = s.strip()
    return s


def write_partitioned_dataset(
    df: pd.DataFrame,
    out_dir: Path,
    partition_cols: Iterable[str],
) -> None:
    """
    Escribe un dataset parquet particionado "tipo Spark":

      out_dir/col1=.../col2=.../part_<uuid>.parquet
    """
    if df.empty:
        return

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    partition_cols = list(partition_cols)

    # Convertimos date a string estable si está en particiones
    if "date" in partition_cols and "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    for keys, g in df.groupby(partition_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        part_dir = out_dir
        for col, val in zip(partition_cols, keys):
            part_dir = part_dir / f"{col}={_safe_partition_value(val)}"
        part_dir.mkdir(parents=True, exist_ok=True)

        fname = f"part_{uuid.uuid4().hex}.parquet"
        g.to_parquet(part_dir / fname, index=False, engine="pyarrow")


def save_layer3_events_spark_style(
    df_borough_hour_day: pd.DataFrame,
    df_daily_borough: pd.DataFrame,
    df_type_daily_borough: pd.DataFrame,
    df_hourly_pattern: pd.DataFrame,
    out_base: Path,
    mode: str = "overwrite",
):
    """
    Guarda:
      - df_borough_hour_day: partitionBy(date, borough)
      - df_daily_borough: partitionBy(borough)
      - df_type_daily_borough: partitionBy(date, borough)
      - df_hourly_pattern: partitionBy(borough)
    """
    out_base = Path(out_base).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    if mode == "overwrite" and out_base.exists():
        print(f"[INFO] Borrando salida (overwrite): {out_base}")
        shutil.rmtree(out_base)
        out_base.mkdir(parents=True, exist_ok=True)

    write_partitioned_dataset(
        df_borough_hour_day,
        out_base / "df_borough_hour_day",
        partition_cols=["date", "borough"],
    )
    write_partitioned_dataset(
        df_daily_borough,
        out_base / "df_daily_borough",
        partition_cols=["borough"],
    )
    write_partitioned_dataset(
        df_type_daily_borough,
        out_base / "df_type_daily_borough",
        partition_cols=["date", "borough"],
    )
    write_partitioned_dataset(
        df_hourly_pattern,
        out_base / "df_hourly_pattern",
        partition_cols=["borough"],
    )

    print("\n[OK] Capa 3 EVENTOS guardada en:", out_base)
    print(" - df_borough_hour_day     ->", out_base / "df_borough_hour_day")
    print(" - df_daily_borough        ->", out_base / "df_daily_borough")
    print(" - df_type_daily_borough   ->", out_base / "df_type_daily_borough")
    print(" - df_hourly_pattern       ->", out_base / "df_hourly_pattern")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Capa 3 Eventos (sin Spark): agregados + salida idéntica a Spark partitionBy.")
    p.add_argument("--from", dest="date_from", default=None, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--to", dest="date_to", default=None, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--min-events", type=int, default=1, help="Mínimo n_events en borough-date-hour (default: 1)")
    p.add_argument("--mode", choices=["overwrite", "append"], default="overwrite", help="overwrite borra y regenera")
    args = p.parse_args()

    if args.date_from is not None:
        datetime.strptime(args.date_from, "%Y-%m-%d")
    if args.date_to is not None:
        datetime.strptime(args.date_to, "%Y-%m-%d")

    project_root = Path(__file__).resolve().parents[3]

    # Capa 2
    layer2_path = (project_root / "data" / "external" / "events" / "standarized").resolve()

    # Path
    out_base = (project_root / "data" / "aggregated" / "events").resolve()

    print("[DEBUG] layer2_path:", layer2_path)
    print("[DEBUG] out_base:", out_base)

    df2 = read_layer2_events(layer2_path)
    df2 = filter_by_range(df2, args.date_from, args.date_to)

    df_bhd, df_db, df_tdb, df_hp = build_layer3_events(df2, min_events_hour_day=args.min_events)
    save_layer3_events_spark_style(df_bhd, df_db, df_tdb, df_hp, out_base, mode=args.mode)


if __name__ == "__main__":
    main()
