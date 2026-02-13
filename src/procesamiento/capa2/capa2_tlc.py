from __future__ import annotations

import argparse
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Tuple

import pandas as pd

try:
    from config.settings import obtener_ruta  # type: ignore
except Exception:
    def obtener_ruta(p: str) -> Path:
        return Path(p)

DEBUG = False


# -----------------------------------------------------------------------------
# Listado de parquets (sin cargar a memoria)
# -----------------------------------------------------------------------------
def _list_parquets(folder: Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(folder.glob("*.parquet"))


def iter_raw_tlc_files(
    raw_base: Path,
    services: Iterable[str] = ("yellow", "green", "fhvhv"),
) -> Iterator[Tuple[str, Path]]:
    raw_base = Path(raw_base).resolve()
    if not raw_base.exists():
        print(f"[WARN] No existe RAW base: {raw_base}")
        return

    for service in services:
        folder = raw_base / service
        files = _list_parquets(folder)

        if not files:
            print(f"[WARN] No hay parquets en {folder}. Se omite {service}.")
            continue

        print(f"[INFO] {service}: {len(files)} parquets encontrados")
        for fp in files:
            yield service, fp


# -----------------------------------------------------------------------------
# Normalización columnas
# -----------------------------------------------------------------------------
def _coalesce_cols(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series([pd.NA] * len(df), index=df.index)


def build_layer2_tlc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df2 = df.copy()

    # 1) pickup/dropoff unificados
    df2["pickup_datetime"] = _coalesce_cols(
        df2,
        ["tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime"],
    )
    df2["dropoff_datetime"] = _coalesce_cols(
        df2,
        ["tpep_dropoff_datetime", "lpep_dropoff_datetime", "dropoff_datetime"],
    )

    df2["pickup_datetime"] = pd.to_datetime(df2["pickup_datetime"], errors="coerce")
    df2["dropoff_datetime"] = pd.to_datetime(df2["dropoff_datetime"], errors="coerce")

    # 2) location ids unificados
    df2["pu_location_id"] = pd.to_numeric(_coalesce_cols(df2, ["PULocationID", "pu_location_id"]), errors="coerce")
    df2["do_location_id"] = pd.to_numeric(_coalesce_cols(df2, ["DOLocationID", "do_location_id"]), errors="coerce")

    # 3) numéricas típicas (si existen)
    num_cols = [
        "total_amount", "fare_amount",
        "tip_amount", "tips",
        "tolls_amount", "tolls",
        "airport_fee", "Airport_fee",
        "congestion_surcharge",
        "base_passenger_fare",
        "trip_distance", "trip_miles",
    ]
    for c in num_cols:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")

    # Unificar Airport_fee -> airport_fee
    if "airport_fee" not in df2.columns and "Airport_fee" in df2.columns:
        df2["airport_fee"] = df2["Airport_fee"]
    if "Airport_fee" in df2.columns:
        df2 = df2.drop(columns=["Airport_fee"])

    # 4) total_amount_std
    zero = pd.Series(0.0, index=df2.index)

    airport_fee_any = df2.get("airport_fee", zero).fillna(0)
    tips_any = df2.get("tip_amount", zero).fillna(0)
    if "tips" in df2.columns:
        tips_any = tips_any + df2["tips"].fillna(0)

    tolls_any = df2.get("tolls_amount", zero).fillna(0)
    if "tolls" in df2.columns:
        tolls_any = tolls_any + df2["tolls"].fillna(0)

    congestion_any = df2.get("congestion_surcharge", zero).fillna(0)
    base_fare = df2.get("base_passenger_fare", zero).fillna(0)

    if "total_amount" in df2.columns:
        df2["total_amount_std"] = df2["total_amount"].fillna(
            base_fare + tips_any + tolls_any + airport_fee_any + congestion_any
        )
    else:
        df2["total_amount_std"] = base_fare + tips_any + tolls_any + airport_fee_any + congestion_any

    df2["total_amount_std"] = pd.to_numeric(df2["total_amount_std"], errors="coerce")

    # 5) features temporales
    df2["date"] = df2["pickup_datetime"].dt.date
    df2["year"] = df2["pickup_datetime"].dt.year
    df2["month"] = df2["pickup_datetime"].dt.month
    df2["hour"] = df2["pickup_datetime"].dt.hour

    # Spark dayofweek: 1=Sunday ... 7=Saturday
    dow0 = df2["pickup_datetime"].dt.dayofweek  # 0=Mon..6=Sun
    df2["day_of_week"] = ((dow0 + 1) % 7) + 1
    df2["is_weekend"] = df2["day_of_week"].isin([1, 7]).astype("int")
    df2["week_of_year"] = df2["pickup_datetime"].dt.isocalendar().week.astype("Int64")

    # 6) duración
    dur = (df2["dropoff_datetime"] - df2["pickup_datetime"]).dt.total_seconds() / 60.0
    df2["trip_duration_min"] = dur
    df2.loc[(df2["trip_duration_min"] < 0) | (df2["trip_duration_min"] > 360), "trip_duration_min"] = pd.NA

    # 7) higiene mínima
    df2 = df2.dropna(subset=["pickup_datetime", "date", "year", "month", "hour"])
    df2 = df2[(df2["hour"] >= 0) & (df2["hour"] <= 23)]

    # columnas finales
    cols = [
        "service_type",
        "pickup_datetime", "dropoff_datetime",
        "date", "year", "month", "hour", "day_of_week", "is_weekend", "week_of_year",
        "trip_duration_min",
        "pu_location_id", "do_location_id",
        "total_amount_std",
        "total_amount", "fare_amount",
        "tip_amount", "tips",
        "tolls_amount", "tolls",
        "congestion_surcharge", "airport_fee",
        "base_passenger_fare",
        "trip_distance", "trip_miles",
        "VendorID", "passenger_count", "RatecodeID", "payment_type",
    ]
    cols = [c for c in cols if c in df2.columns]
    df2 = df2[cols]

    if DEBUG:
        print(df2.head())

    return df2


# -----------------------------------------------------------------------------
# Escritura particionada (sin juntar todo)
# -----------------------------------------------------------------------------
def write_partitioned(df: pd.DataFrame, out_dir: Path):
    """
    Escribe df en:
      out_dir/service_type=.../year=.../month=.../part_<uuid>.parquet
    """
    if df.empty:
        return

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for (svc, y, m), g in df.groupby(["service_type", "year", "month"], dropna=False):
        part_dir = out_dir / f"service_type={svc}" / f"year={int(y)}" / f"month={int(m)}"
        part_dir.mkdir(parents=True, exist_ok=True)
        fname = f"part_{uuid.uuid4().hex}.parquet"
        g.to_parquet(part_dir / fname, index=False, engine="pyarrow")


def main():
    p = argparse.ArgumentParser(description="Capa 2 TLC (Pandas, sin Spark): procesa fichero a fichero (sin reventar RAM).")
    p.add_argument("--raw-dir", default=str(obtener_ruta("data/raw")), help="RAW base (contiene yellow/green/fhvhv)")
    p.add_argument("--out-dir", default=str(obtener_ruta("data/external/tlc/standarized")), help="Salida capa 2 TLC (dataset particionado)")
    p.add_argument("--from", dest="date_from", default=None, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--to", dest="date_to", default=None, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--mode", choices=["append", "overwrite"], default="append", help="append o overwrite (borra salida)")
    args = p.parse_args()

    if args.date_from:
        datetime.strptime(args.date_from, "%Y-%m-%d")
    if args.date_to:
        datetime.strptime(args.date_to, "%Y-%m-%d")

    raw_base = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if args.mode == "overwrite" and out_dir.exists():
        print(f"[INFO] Borrando salida (overwrite): {out_dir}")
        shutil.rmtree(out_dir)

    any_written = False

    for service, fp in iter_raw_tlc_files(raw_base):
        print(f"\n[INFO] Procesando {service} -> {fp.name}")

        df = pd.read_parquet(fp)
        df["service_type"] = service

        df2 = build_layer2_tlc(df)

        # filtro por rango opcional
        if not df2.empty and (args.date_from or args.date_to):
            dt = pd.to_datetime(df2["date"], errors="coerce")
            if args.date_from:
                df2 = df2[dt >= pd.to_datetime(args.date_from)]
            if args.date_to:
                df2 = df2[dt <= pd.to_datetime(args.date_to)]

        write_partitioned(df2, out_dir)
        any_written = any_written or (not df2.empty)

    if not any_written:
        print("[WARN] No se escribió nada (no había datos o todo quedó filtrado).")
    else:
        print("\n[OK] Capa 2 TLC guardada en:", out_dir)


if __name__ == "__main__":
    main()