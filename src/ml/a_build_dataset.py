from __future__ import annotations

import argparse
import os
import math
from pathlib import Path
import pandas as pd

# Para generar dataset completo
# uv run -m src.ml.a_build_dataset
# Genera: data/ml/dataset_completo.parquet
#
# Para generar un dataset más pequeño para entrenar los modelos más rápidamente (rango por fechas)
# uv run -m src.ml.a_build_dataset --from 2024-01-01 --to 2024-01-14
# Genera: data/ml/dataset_rango_2024-01-01__2024-01-14.parquet
#
# Para generar un dataset más pequeño para entrenar los modelos más rápidamente (muestreo aleatorio)
# uv run -m src.ml.a_build_dataset --sample-frac 0.01
# Genera: data/ml/dataset_completo.parquet pero con 1% de filas (si no pasas --from/--to)


# -----------------------
# Helpers de lectura
# -----------------------
def read_partitioned_parquet_dir(base: Path) -> pd.DataFrame:
    """Lee un directorio con parquets particionados tipo Spark (subcarpetas date=... etc.)."""
    base = Path(base).resolve()
    if not base.exists():
        raise FileNotFoundError(f"No existe: {base}")
    files = list(base.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No hay .parquet dentro de: {base}")
    return pd.concat([pd.read_parquet(fp) for fp in files], ignore_index=True)


def ensure_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"[{name}] Faltan columnas: {miss}")


def _safe_date_for_filename(s: str) -> str:
    # deja YYYY-MM-DD “limpio” para nombre de fichero
    return s.strip().replace("/", "-").replace("\\", "-").replace(":", "-")


# -----------------------
# Build dataset
# -----------------------
def build_dataset(
    # Inputs
    tlc_dir: str = "data/aggregated/df_zone_hour_day_service",
    meteo_path: str = "data/external/meteo/aggregated/df_hour_day/data.parquet",
    events_dir: str = "data/aggregated/events/df_borough_hour_day",
    variability_dir: str = "data/aggregated/df_variability",
    # Output
    out_path: str = "data/ml/dataset_completo.parquet",
    # Rango / sample
    date_from: str | None = None,
    date_to: str | None = None,
    sample_frac: float | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Genera dataset final de modelado:
    - Base: capa3 TLC (zona-hora-fecha-servicio)
    - Join meteo: por date+hour
    - Join eventos: agregados ciudad por date+hour
    - Features lags/rolling por zona+servicio
    - Métrica de negocio: biz_score = std_price * log(1+num_trips)
    - (Opcional) Targets: stress_score (= biz_score) e is_stress (top10%)

    Por defecto (sin date_from/date_to ni sample_frac) => dataset completo.
    """
    project_root = Path(__file__).resolve().parents[2]

    # --- 1) TLC base
    tlc_base = project_root / tlc_dir
    tlc = read_partitioned_parquet_dir(tlc_base)

    ensure_cols(
        tlc,
        ["pu_location_id", "hour", "date", "service_type", "num_trips", "avg_price", "std_price"],
        "TLC",
    )

    # Normalizar tipos
    tlc["date"] = pd.to_datetime(tlc["date"], errors="coerce").dt.date
    tlc["hour"] = pd.to_numeric(tlc["hour"], errors="coerce").astype("Int64")
    tlc["pu_location_id"] = pd.to_numeric(tlc["pu_location_id"], errors="coerce").astype("Int64")
    tlc["num_trips"] = pd.to_numeric(tlc["num_trips"], errors="coerce")
    tlc["avg_price"] = pd.to_numeric(tlc["avg_price"], errors="coerce")
    tlc["std_price"] = pd.to_numeric(tlc["std_price"], errors="coerce")
    tlc["service_type"] = tlc["service_type"].astype(str)

    tlc = tlc.dropna(subset=["date", "hour", "pu_location_id", "service_type", "num_trips", "std_price"])

    # Filtro fechas (rango)
    if date_from is not None:
        tlc = tlc[tlc["date"] >= pd.to_datetime(date_from).date()]
    if date_to is not None:
        tlc = tlc[tlc["date"] <= pd.to_datetime(date_to).date()]

    # Sample alternativo
    if sample_frac is not None and 0 < sample_frac < 1:
        tlc = tlc.sample(frac=sample_frac, random_state=random_state)

    # --- 2) METEO por date+hour
    meteo_fp = project_root / meteo_path
    if meteo_fp.exists():
        meteo = pd.read_parquet(meteo_fp)
        ensure_cols(meteo, ["date", "hour"], "METEO")
        meteo["date"] = pd.to_datetime(meteo["date"], errors="coerce").dt.date
        meteo["hour"] = pd.to_numeric(meteo["hour"], errors="coerce").astype("Int64")

        meteo_cols = ["date", "hour"]
        for c in [
            "rain_mm_sum",
            "temp_c_mean",
            "wind_kmh_mean",
            "precip_mm_sum",
            "snowfall_mm_sum",
            "weather_code_mode",
        ]:
            if c in meteo.columns:
                meteo_cols.append(c)

        meteo = meteo[meteo_cols].dropna(subset=["date", "hour"])
        df = tlc.merge(meteo, on=["date", "hour"], how="left")
    else:
        df = tlc.copy()
        # Mantener esquema estable si falta meteo
        df["rain_mm_sum"] = pd.NA
        df["temp_c_mean"] = pd.NA
        df["wind_kmh_mean"] = pd.NA

    # --- 3) EVENTOS: ciudad por date+hour (suma boroughs)
    events_base = project_root / events_dir
    if events_base.exists():
        ev = read_partitioned_parquet_dir(events_base)
        ensure_cols(ev, ["date", "hour", "n_events"], "EVENTOS")
        ev["date"] = pd.to_datetime(ev["date"], errors="coerce").dt.date
        ev["hour"] = pd.to_numeric(ev["hour"], errors="coerce").astype("Int64")
        ev["n_events"] = pd.to_numeric(ev["n_events"], errors="coerce").fillna(0)

        ev_city = ev.groupby(["date", "hour"], as_index=False).agg(event_count_city=("n_events", "sum"))
        df = df.merge(ev_city, on=["date", "hour"], how="left")
    else:
        df["event_count_city"] = 0

    df["event_count_city"] = pd.to_numeric(df["event_count_city"], errors="coerce").fillna(0).astype(int)

    # --- 4) Features temporales + calendáricas
    df = df.sort_values(["service_type", "pu_location_id", "date", "hour"])
    grp = df.groupby(["service_type", "pu_location_id"], sort=False)

    df["lag_1h_trips"] = grp["num_trips"].shift(1)
    df["lag_24h_trips"] = grp["num_trips"].shift(24)
    df["roll_3h_trips"] = grp["num_trips"].rolling(3, min_periods=1).mean().reset_index(level=[0, 1], drop=True)

    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek.astype(int)
    df["month"] = pd.to_datetime(df["date"]).dt.month.astype(int)

    # Flags útiles
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    df[["lag_1h_trips", "lag_24h_trips"]] = df[["lag_1h_trips", "lag_24h_trips"]].fillna(0)

    # --- 5) Métrica de negocio: biz_score (4 variantes)
    #
    # Variabilidad: usamos IQR de capa3/df_variability si existe,
    # o std_price como proxy si no se ha ejecutado capa3.
    var_base = project_root / variability_dir
    has_iqr = False
    if var_base.exists():
        try:
            var_df = read_partitioned_parquet_dir(var_base)
            if "price_variability" in var_df.columns:
                var_df["pu_location_id"] = pd.to_numeric(var_df["pu_location_id"], errors="coerce").astype("Int64")
                var_df["hour"] = pd.to_numeric(var_df["hour"], errors="coerce").astype("Int64")
                var_df["service_type"] = var_df["service_type"].astype(str)
                iqr_map = var_df[["pu_location_id", "hour", "service_type", "price_variability"]].drop_duplicates()
                df = df.merge(iqr_map, on=["pu_location_id", "hour", "service_type"], how="left")
                has_iqr = True
                print(f"   IQR join: {df['price_variability'].notna().sum():,}/{len(df):,} filas con IQR de capa3")
        except Exception as exc:
            print(f"⚠️  No se pudo leer df_variability: {exc}")

    if has_iqr:
        # Rellenar filas sin IQR con std_price como fallback
        df["price_variability"] = df["price_variability"].fillna(df["std_price"].fillna(0))
        variability = df["price_variability"].astype(float)
    else:
        # Sin capa3: usar std_price como proxy de variabilidad
        variability = df["std_price"].fillna(0).astype(float)
        df["price_variability"] = variability
        print("   IQR no disponible → usando std_price como proxy de variabilidad")

    log_volume = df["num_trips"].fillna(0).astype(float).apply(lambda x: math.log1p(x))

    # Agregador 1 — Solo variabilidad (baseline del índice)
    df["biz_score_iqr"] = variability

    # Agregador 2 — Variabilidad × actividad (recomendado)
    df["biz_score"] = variability * log_volume

    # Agregador 3 — Índice normalizado (z-sum)
    v_mean, v_std = float(variability.mean()), float(variability.std())
    lv_mean, lv_std = float(log_volume.mean()), float(log_volume.std())

    z_var = (variability - v_mean) / v_std if v_std > 0 else 0.0
    z_lv = (log_volume - lv_mean) / lv_std if lv_std > 0 else 0.0

    df["biz_score_zsum"] = z_var + z_lv

    # Agregador 4 — Multiplicación normalizada
    df["biz_score_zproduct"] = z_var * z_lv

    # Target operativo: stress_score = biz_score (Agregador 2)
    df["stress_score"] = df["biz_score"]
    thr = df["stress_score"].quantile(0.90)
    df["is_stress"] = (df["stress_score"] >= thr).astype(int)

    # --- 6) Guardado
    out_fp = (project_root / out_path).resolve()
    os.makedirs(out_fp.parent, exist_ok=True)
    df.to_parquet(out_fp, index=False)
    print(f"✅ Dataset guardado -> {out_fp} | rows={len(df):,} cols={len(df.columns)}")

    return df


def main() -> None:
    p = argparse.ArgumentParser(
        description="Construye dataset de modelado (TLC + Meteo + Eventos). "
        "Por defecto genera el dataset completo."
    )
    p.add_argument(
        "--from",
        dest="date_from",
        default=None,
        help="YYYY-MM-DD (inclusive). Si se indica junto con --to, genera dataset por rango.",
    )
    p.add_argument(
        "--to",
        dest="date_to",
        default=None,
        help="YYYY-MM-DD (inclusive). Si se indica junto con --from, genera dataset por rango.",
    )
    p.add_argument("--sample-frac", type=float, default=None, help="Opcional: muestreo aleatorio (0<frac<1).")
    p.add_argument("--out-dir", default="data/ml", help="Carpeta de salida (default: data/ml).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)

    # Nombre de salida: completo o rango
    if args.date_from and args.date_to:
        f = _safe_date_for_filename(args.date_from)
        t = _safe_date_for_filename(args.date_to)
        out_name = f"dataset_rango_{f}__{t}.parquet"
    else:
        out_name = "dataset_completo.parquet"

    build_dataset(
        out_path=str(out_dir / out_name),
        date_from=args.date_from,
        date_to=args.date_to,
        sample_frac=args.sample_frac,
    )


if __name__ == "__main__":
    main()