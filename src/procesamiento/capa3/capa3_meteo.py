# src/procesamiento/capa3/capa3_meteo.py
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


def read_layer2_meteo(layer2_path: Path) -> pd.DataFrame:
    layer2_path = Path(layer2_path).resolve()
    if not layer2_path.exists():
        raise FileNotFoundError(f"No existe {layer2_path}")

    # Ahora buscamos directamente los archivos con el patrón que creamos: meteo_YYYY-MM.parquet
    files = sorted(layer2_path.glob("meteo_*.parquet"))
    
    # Si no encuentra "meteo_...", intenta con cualquier parquet por si acaso
    if not files:
        files = sorted(layer2_path.glob("*.parquet"))
        
    if not files:
        raise FileNotFoundError(f"No hay parquets dentro de {layer2_path}")

    print(f"[INFO] Leyendo {len(files)} archivos de Capa 2...")
    dfs = [pd.read_parquet(fp) for fp in files]
    df = pd.concat(dfs, ignore_index=True)
    return df


def _parse_date(s: str | None) -> pd.Timestamp | None:
    if s is None:
        return None
    return pd.to_datetime(s, format="%Y-%m-%d", errors="raise")


def filter_by_range(df: pd.DataFrame, date_from: str | None, date_to: str | None) -> pd.DataFrame:
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


def build_layer3_meteo(df2: pd.DataFrame):
    df = df2.copy()

    # numeric cols
    for c in ["temp_c", "precip_mm", "rain_mm", "snowfall_mm", "wind_kmh"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "weather_code" in df.columns:
        df["weather_code"] = pd.to_numeric(df["weather_code"], errors="coerce").astype("Int64")

    df = df.dropna(subset=["date", "hour"])
    df = df[(df["hour"] >= 0) & (df["hour"] <= 23)]

    # 1) date+hour
    agg_hour_day = {k: v for k, v in {
        "temp_c": "mean",
        "wind_kmh": "mean",
        "precip_mm": "sum",
        "rain_mm": "sum",
        "snowfall_mm": "sum",
    }.items() if k in df.columns}

    df_hour_day = df.groupby(["date", "hour"], as_index=False).agg(agg_hour_day)

    rename_map = {}
    if "temp_c" in df_hour_day.columns: rename_map["temp_c"] = "temp_c_mean"
    if "wind_kmh" in df_hour_day.columns: rename_map["wind_kmh"] = "wind_kmh_mean"
    if "precip_mm" in df_hour_day.columns: rename_map["precip_mm"] = "precip_mm_sum"
    if "rain_mm" in df_hour_day.columns: rename_map["rain_mm"] = "rain_mm_sum"
    if "snowfall_mm" in df_hour_day.columns: rename_map["snowfall_mm"] = "snowfall_mm_sum"
    df_hour_day = df_hour_day.rename(columns=rename_map)

    if "weather_code" in df.columns:
        mode_wc = (
            df.dropna(subset=["weather_code"])
              .groupby(["date", "hour"])["weather_code"]
              .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)
              .reset_index()
              .rename(columns={"weather_code": "weather_code_mode"})
        )
        df_hour_day = df_hour_day.merge(mode_wc, on=["date", "hour"], how="left")

    # 2) daily
    agg_daily = {}
    if "temp_c" in df.columns: agg_daily["temp_c"] = ["mean", "min", "max"]
    if "wind_kmh" in df.columns: agg_daily["wind_kmh"] = ["mean", "max"]
    if "precip_mm" in df.columns: agg_daily["precip_mm"] = "sum"
    if "rain_mm" in df.columns: agg_daily["rain_mm"] = "sum"
    if "snowfall_mm" in df.columns: agg_daily["snowfall_mm"] = "sum"

    df_daily = df.groupby("date").agg(agg_daily)
    df_daily.columns = ["_".join([c for c in col if c]) for col in df_daily.columns.to_flat_index()]
    df_daily = df_daily.reset_index()

    if "weather_code" in df.columns:
        wc_daily = (
            df.dropna(subset=["weather_code"])
              .groupby("date")["weather_code"]
              .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)
              .reset_index()
              .rename(columns={"weather_code": "weather_code_mode"})
        )
        df_daily = df_daily.merge(wc_daily, on="date", how="left")

    # 3) hourly pattern
    hourly_cols = {}
    if "temp_c" in df.columns: hourly_cols["temp_c"] = ["mean", "std"]
    if "wind_kmh" in df.columns: hourly_cols["wind_kmh"] = ["mean", "std"]
    if "precip_mm" in df.columns: hourly_cols["precip_mm"] = ["mean", "std"]

    df_hourly_pattern = df.groupby("hour").agg(hourly_cols)
    df_hourly_pattern.columns = ["_".join([c for c in col if c]) for col in df_hourly_pattern.columns.to_flat_index()]
    df_hourly_pattern = df_hourly_pattern.reset_index()

    # 4) weathercode distribution
    df_weathercode_daily = None
    if "weather_code" in df.columns:
        df_weathercode_daily = (
            df.dropna(subset=["weather_code"])
              .groupby(["date", "weather_code"], as_index=False)
              .size()
              .rename(columns={"size": "n_hours"})
        )

    if DEBUG:
        print(df_hour_day.head())
        print(df_daily.head())
        print(df_hourly_pattern.head())

    return df_hour_day, df_daily, df_hourly_pattern, df_weathercode_daily


def save_layer3_meteo(df_hour_day, df_daily, df_hourly_pattern, df_weathercode_daily, out_base: Path):
    out_base = Path(out_base).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    (out_base / "df_hour_day").mkdir(parents=True, exist_ok=True)
    (out_base / "df_daily").mkdir(parents=True, exist_ok=True)
    (out_base / "df_hourly_pattern").mkdir(parents=True, exist_ok=True)

    df_hour_day.to_parquet(out_base / "df_hour_day" / "data.parquet", index=False, engine="pyarrow")
    df_daily.to_parquet(out_base / "df_daily" / "data.parquet", index=False, engine="pyarrow")
    df_hourly_pattern.to_parquet(out_base / "df_hourly_pattern" / "data.parquet", index=False, engine="pyarrow")

    if df_weathercode_daily is not None:
        (out_base / "df_weathercode_daily").mkdir(parents=True, exist_ok=True)
        df_weathercode_daily.to_parquet(out_base / "df_weathercode_daily" / "data.parquet", index=False, engine="pyarrow")

    print("\nCapa 3 METEO guardada en:", out_base)


def main():
    p = argparse.ArgumentParser(description="Capa 3 Meteo (sin Spark): agregados horarios y diarios.")
    p.add_argument("--from", dest="date_from", default=None, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--to", dest="date_to", default=None, help="YYYY-MM-DD (inclusive)")
    args = p.parse_args()

    if args.date_from is not None:
        datetime.strptime(args.date_from, "%Y-%m-%d")
    if args.date_to is not None:
        datetime.strptime(args.date_to, "%Y-%m-%d")

    project_root = Path(__file__).resolve().parents[3]
    layer2_path = (project_root / "data" / "external" / "meteo" / "standarized").resolve()
    out_base = (project_root / "data" / "external" / "meteo" / "aggregated").resolve()

    print("[DEBUG] layer2_path:", layer2_path)
    print("[DEBUG] out_base:", out_base)

    df2 = read_layer2_meteo(layer2_path)
    df2 = filter_by_range(df2, args.date_from, args.date_to)

    df_hour_day, df_daily, df_hourly_pattern, df_weathercode_daily = build_layer3_meteo(df2)
    save_layer3_meteo(df_hour_day, df_daily, df_hourly_pattern, df_weathercode_daily, out_base)


if __name__ == "__main__":
    main()