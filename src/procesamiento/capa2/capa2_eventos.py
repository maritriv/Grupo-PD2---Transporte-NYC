# src/procesamiento/capa2/capa2_eventos.py
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from config.settings import obtener_ruta, config  # type: ignore
except Exception:
    config = {}

    def obtener_ruta(p: str) -> Path:
        return Path(p)


DEBUG = False  # True para ver previews


def read_raw_events(in_dir: Path, base_name: str = "events_daily_borough_type") -> pd.DataFrame:
    """
    Lee datos RAW de eventos en dos formatos posibles:

    1) Formato antiguo (1 archivo):
        - {base_name}.parquet  o  {base_name}.csv

    2) Formato nuevo (muchos archivos mensuales):
        - events_YYYY_MM.parquet (y opcionalmente .parquet.gz)
        - (si quisieras) events_YYYY_MM.csv (y .csv.gz)
    """
    in_dir = Path(in_dir).resolve()
    if not in_dir.exists():
        raise FileNotFoundError(f"No existe el directorio RAW: {in_dir}")

    # 1) Antiguo
    parquet_path = in_dir / f"{base_name}.parquet"
    csv_path = in_dir / f"{base_name}.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    # 2) Mensual
    files = sorted(in_dir.glob("events_*.parquet"))
    if not files:
        files = sorted(in_dir.glob("events_*.parquet.gz"))

    if files:
        print(f"[INFO] Leyendo {len(files)} archivos RAW mensuales (parquet)...")
        dfs = [pd.read_parquet(fp) for fp in files]
        return pd.concat(dfs, ignore_index=True)

    # Fallback CSV si lo usáis
    files = sorted(in_dir.glob("events_*.csv"))
    if not files:
        files = sorted(in_dir.glob("events_*.csv.gz"))

    if files:
        print(f"[INFO] Leyendo {len(files)} archivos RAW mensuales (csv)...")
        dfs = [pd.read_csv(fp) for fp in files]
        return pd.concat(dfs, ignore_index=True)

    raise FileNotFoundError(
        f"No encuentro {base_name}.parquet ni {base_name}.csv en {in_dir}, "
        f"ni archivos mensuales events_*.parquet / events_*.csv."
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


def build_layer2_events(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()

    # Tipos
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce").dt.date
    df2["hour"] = pd.to_numeric(df2["hour"], errors="coerce").astype("Int64")
    df2["borough"] = df2.get("borough")
    df2["event_type"] = df2.get("event_type")
    df2["n_events"] = pd.to_numeric(df2.get("n_events"), errors="coerce").astype("Int64")

    # Higiene mínima
    df2 = df2.dropna(subset=["date", "hour", "borough", "event_type", "n_events"])
    df2 = df2[(df2["hour"] >= 0) & (df2["hour"] <= 23)]
    df2 = df2[df2["n_events"] >= 0]

    # Limpieza strings
    df2["borough"] = df2["borough"].astype(str).str.strip().str.title()
    df2["event_type"] = df2["event_type"].astype(str).str.strip()

    # Variables temporales
    dt = pd.to_datetime(df2["date"])
    df2["year"] = dt.dt.year.astype("int")
    df2["month"] = dt.dt.month.astype("int")
    # Spark dayofweek: 1=Sunday ... 7=Saturday
    df2["day_of_week"] = ((dt.dt.dayofweek + 1) % 7) + 1
    df2["is_weekend"] = df2["day_of_week"].isin([1, 7]).astype("int")
    df2["week_of_year"] = dt.dt.isocalendar().week.astype("int")

    # Orden columnas
    cols = [
        "date",
        "hour",
        "year",
        "month",
        "day_of_week",
        "is_weekend",
        "week_of_year",
        "borough",
        "event_type",
        "n_events",
    ]
    cols = [c for c in cols if c in df2.columns]
    df2 = df2[cols]

    if DEBUG:
        print(df2.head(30))

    return df2


def save_layer2_events(df: pd.DataFrame, out_dir: Path):
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Un parquet por mes
    for (y, m), g in df.groupby(["year", "month"], dropna=False):
        file_name = f"events_{int(y)}-{int(m):02d}.parquet"
        dest_path = out_dir / file_name
        g.to_parquet(dest_path, index=False, engine="pyarrow")

    print("\n[OK] Capa 2 EVENTOS guardada como archivos individuales en:", out_dir)


def main():
    print(">>> entrando en main capa2_eventos (PANDAS MODE)")

    events_cfg = (config.get("eventos") or {}) if isinstance(config, dict) else {}
    default_raw_name = events_cfg.get("out_name", "events_daily_borough_type")

    p = argparse.ArgumentParser(description="Capa 2 Eventos (sin Spark): tipado + higiene + particionado.")
    p.add_argument("--name", dest="raw_name", default=default_raw_name, help="Nombre base del RAW (formato antiguo)")
    p.add_argument("--from", dest="date_from", default=None, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--to", dest="date_to", default=None, help="YYYY-MM-DD (inclusive)")
    args = p.parse_args()

    if args.date_from is not None:
        datetime.strptime(args.date_from, "%Y-%m-%d")
    if args.date_to is not None:
        datetime.strptime(args.date_to, "%Y-%m-%d")

    project_root = Path(__file__).resolve().parents[3]
    raw_dir = (project_root / "data" / "external" / "events" / "raw").resolve()
    out_dir = (project_root / "data" / "external" / "events" / "standarized").resolve()

    print("[DEBUG] raw_dir:", raw_dir)
    print("[DEBUG] out_dir:", out_dir)

    df_raw = read_raw_events(raw_dir, base_name=args.raw_name)
    df_raw = filter_by_range(df_raw, args.date_from, args.date_to)

    df_l2 = build_layer2_events(df_raw)
    save_layer2_events(df_l2, out_dir)

    print("[OK] rows:", len(df_l2))


if __name__ == "__main__":
    main()