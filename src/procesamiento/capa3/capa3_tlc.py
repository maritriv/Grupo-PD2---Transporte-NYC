# src/procesamiento/capa3/capa3_tlc.py
from __future__ import annotations

import argparse
import gc
import math
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import pandas as pd

try:
    from config.settings import obtener_ruta  # type: ignore
except Exception:
    def obtener_ruta(p: str) -> Path:
        return Path(p)

DEBUG = False
ALLOWED_MIN_DATE = pd.Timestamp("2023-01-01")
ALLOWED_MAX_DATE = pd.Timestamp("2025-12-31")

NEEDED_COLS = ["date", "hour", "service_type", "pu_location_id", "total_amount_std"]


# -----------------------------------------------------------------------------
# Entrada: iterar parquets de capa 2 agrupados por year/month
# -----------------------------------------------------------------------------
def iter_month_partitions(base: Path) -> Iterator[Tuple[int, int, List[Path]]]:
    """
    Busca parquets en una estructura tipo:
      data/standarized/service_type=.../year=YYYY/month=MM/*.parquet

    Devuelve:
      (year, month, [lista de parquets de ese mes de todos los servicios])
    """
    base = Path(base).resolve()
    if not base.exists():
        raise FileNotFoundError(f"No existe: {base}")

    month_map: Dict[Tuple[int, int], List[Path]] = {}

    for fp in base.rglob("*.parquet"):
        if not fp.is_file():
            continue

        year = None
        month = None

        for part in fp.parts:
            if part.startswith("year="):
                try:
                    year = int(part.split("=", 1)[1])
                except Exception:
                    year = None
            elif part.startswith("month="):
                try:
                    month = int(part.split("=", 1)[1])
                except Exception:
                    month = None

        if year is None or month is None:
            # Si algún parquet no sigue la estructura esperada, lo ignoramos.
            # Así evitamos mezclar cosas raras.
            continue

        month_map.setdefault((year, month), []).append(fp)

    for (year, month) in sorted(month_map.keys()):
        yield year, month, sorted(month_map[(year, month)])


# -----------------------------------------------------------------------------
# Agregación online (n, sum, sumsq) + sets para distinct + sampling para quantiles
# -----------------------------------------------------------------------------
@dataclass
class RunningStats:
    n: int = 0
    s: float = 0.0
    ss: float = 0.0

    def add(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        self.n += int(x.size)
        self.s += float(x.sum())
        self.ss += float((x * x).sum())

    def mean(self) -> float:
        return self.s / self.n if self.n > 0 else float("nan")

    def std_sample(self) -> float:
        if self.n <= 1:
            return float("nan")
        var = (self.ss - (self.s * self.s) / self.n) / (self.n - 1)
        if var < 0:
            var = 0.0
        return float(math.sqrt(var))


class Reservoir:
    def __init__(self, k: int, seed: int = 42):
        self.k = int(k)
        self.rng = np.random.default_rng(seed)
        self.samples: List[float] = []
        self.seen: int = 0

    def add_many(self, x: np.ndarray) -> None:
        for v in x:
            self.seen += 1
            if len(self.samples) < self.k:
                self.samples.append(float(v))
            else:
                j = self.rng.integers(1, self.seen + 1)
                if j <= self.k:
                    self.samples[int(j - 1)] = float(v)

    def percentiles(self, ps: Tuple[float, float]) -> Tuple[float, float]:
        if not self.samples:
            return (float("nan"), float("nan"))
        arr = np.asarray(self.samples, dtype=float)
        return (float(np.percentile(arr, ps[0])), float(np.percentile(arr, ps[1])))


def _safe_partition_value(v) -> str:
    s = str(v)
    s = s.replace("/", "-").replace("\\", "-").replace(":", "-").strip()
    return s


def write_partitioned_dataset(df: pd.DataFrame, out_dir: Path, partition_cols: Iterable[str]) -> None:
    if df.empty:
        return

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    part_cols = list(partition_cols)

    if "date" in part_cols and "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    for keys, g in df.groupby(part_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        part_dir = out_dir
        for col, val in zip(part_cols, keys):
            part_dir = part_dir / f"{col}={_safe_partition_value(val)}"
        part_dir.mkdir(parents=True, exist_ok=True)

        fname = f"part_{uuid.uuid4().hex}.parquet"
        g.to_parquet(part_dir / fname, index=False, engine="pyarrow")


# -----------------------------------------------------------------------------
# Lectura + filtros equivalentes a Spark
# -----------------------------------------------------------------------------
def normalize_and_filter(
    df: pd.DataFrame,
    min_date: str,
    max_date: str,
    cap_max_price: float,
) -> pd.DataFrame:
    for c in NEEDED_COLS:
        if c not in df.columns:
            raise ValueError(f"Falta columna requerida: {c}")

    out = df[NEEDED_COLS].copy()

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["hour"] = pd.to_numeric(out["hour"], errors="coerce").astype("Int64")
    out["service_type"] = out["service_type"].astype(str)
    out["pu_location_id"] = pd.to_numeric(out["pu_location_id"], errors="coerce").astype("Int64")
    out["total_amount_std"] = pd.to_numeric(out["total_amount_std"], errors="coerce")

    out = out.dropna(subset=["date", "hour", "service_type", "pu_location_id", "total_amount_std"])

    dmin = pd.to_datetime(min_date)
    dmax = pd.to_datetime(max_date)
    dmin = max(dmin, ALLOWED_MIN_DATE)
    dmax = min(dmax, ALLOWED_MAX_DATE)
    out = out[(out["date"] >= dmin) & (out["date"] <= dmax)]

    out = out[(out["hour"] >= 0) & (out["hour"] <= 23)]
    out = out[(out["total_amount_std"] > 0) & (out["total_amount_std"] < float(cap_max_price))]

    return out


# -----------------------------------------------------------------------------
# Helpers para convertir acumuladores mensuales en DataFrames
# -----------------------------------------------------------------------------
def monthly_daily_service_to_df(
    daily_stats: Dict[Tuple[pd.Timestamp, str], RunningStats],
    daily_zones: Dict[Tuple[pd.Timestamp, str], set],
) -> pd.DataFrame:
    rows = []
    for (d, svc), st in daily_stats.items():
        rows.append(
            {
                "date": d,
                "service_type": svc,
                "num_trips": st.n,
                "avg_price": st.mean(),
                "std_price": st.std_sample(),
                "unique_zones": len(daily_zones[(d, svc)]),
            }
        )
    return pd.DataFrame(rows)


def monthly_zone_hour_day_global_to_df(
    zhd_global: Dict[Tuple[int, int, pd.Timestamp], RunningStats],
    min_trips_df2: int,
) -> pd.DataFrame:
    rows = []
    for (z, h, d), st in zhd_global.items():
        if st.n >= min_trips_df2:
            rows.append(
                {
                    "pu_location_id": z,
                    "hour": h,
                    "date": d,
                    "num_trips": st.n,
                    "avg_price": st.mean(),
                    "std_price": st.std_sample(),
                }
            )
    return pd.DataFrame(rows)


def monthly_zone_hour_day_service_to_df(
    zhd_service: Dict[Tuple[int, int, pd.Timestamp, str], RunningStats],
    min_trips_df2: int,
) -> pd.DataFrame:
    rows = []
    for (z, h, d, svc), st in zhd_service.items():
        if st.n >= min_trips_df2:
            rows.append(
                {
                    "pu_location_id": z,
                    "hour": h,
                    "date": d,
                    "service_type": svc,
                    "num_trips": st.n,
                    "avg_price": st.mean(),
                    "std_price": st.std_sample(),
                }
            )
    return pd.DataFrame(rows)


def finalize_types(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "hour" in df.columns:
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype("Int64")
    if "pu_location_id" in df.columns:
        df["pu_location_id"] = pd.to_numeric(df["pu_location_id"], errors="coerce").astype("Int64")
    if "num_trips" in df.columns:
        df["num_trips"] = pd.to_numeric(df["num_trips"], errors="coerce").astype("Int64")
    return df


# -----------------------------------------------------------------------------
# Construcción CAPA 3:
# - DF1/DF2a/DF2b por mes, guardando y liberando
# - DF3 global
# -----------------------------------------------------------------------------
def build_layer3_streaming_monthly(
    layer2_path: Path,
    out_base: Path,
    min_date: str,
    max_date: str,
    cap_max_price: float,
    min_trips_df2: int,
    min_trips_df3: int,
    reservoir_k: int = 500,
) -> pd.DataFrame:
    month_parts = list(iter_month_partitions(layer2_path))
    if not month_parts:
        raise FileNotFoundError(
            f"No hay parquets dentro de {layer2_path} con estructura year=YYYY/month=MM"
        )

    total_months = len(month_parts)
    total_files = sum(len(files) for _, _, files in month_parts)
    print(f"[INFO] Meses detectados: {total_months}")
    print(f"[INFO] Parquets detectados en Capa 2: {total_files}")

    # DF3 global
    var_stats: Dict[Tuple[int, int, str], Tuple[RunningStats, Reservoir]] = {}

    processed_files = 0

    for mi, (year, month, files) in enumerate(month_parts, start=1):
        print(f"\n[INFO] Procesando mes {mi}/{total_months}: {year}-{month:02d} ({len(files)} parquets)")

        # Acumuladores SOLO del mes actual: se liberan al final del mes
        daily_stats: Dict[Tuple[pd.Timestamp, str], RunningStats] = {}
        daily_zones: Dict[Tuple[pd.Timestamp, str], set] = {}

        zhd_global: Dict[Tuple[int, int, pd.Timestamp], RunningStats] = {}
        zhd_service: Dict[Tuple[int, int, pd.Timestamp, str], RunningStats] = {}

        for fi, fp in enumerate(files, start=1):
            processed_files += 1

            if fi % 25 == 0 or fi == len(files):
                print(
                    f"[INFO]   Mes {year}-{month:02d}: fichero {fi}/{len(files)} "
                    f"(global {processed_files}/{total_files})"
                )

            df_raw = pd.read_parquet(fp, columns=NEEDED_COLS)
            df = normalize_and_filter(
                df_raw,
                min_date=min_date,
                max_date=max_date,
                cap_max_price=cap_max_price,
            )

            del df_raw

            if df.empty:
                del df
                gc.collect()
                continue

            # DF1 mensual: date + service_type
            for (d, svc), g in df.groupby(["date", "service_type"], dropna=False):
                key = (d, svc)
                st = daily_stats.get(key)
                if st is None:
                    st = RunningStats()
                    daily_stats[key] = st
                    daily_zones[key] = set()

                x = g["total_amount_std"].to_numpy(dtype=float, copy=False)
                st.add(x)
                daily_zones[key].update(g["pu_location_id"].astype(int).unique().tolist())

            # DF2a mensual: pu_location_id + hour + date
            for (z, h, d), g in df.groupby(["pu_location_id", "hour", "date"], dropna=False):
                key = (int(z), int(h), d)
                st = zhd_global.get(key)
                if st is None:
                    st = RunningStats()
                    zhd_global[key] = st
                st.add(g["total_amount_std"].to_numpy(dtype=float, copy=False))

            # DF2b mensual: pu_location_id + hour + date + service_type
            for (z, h, d, svc), g in df.groupby(
                ["pu_location_id", "hour", "date", "service_type"],
                dropna=False,
            ):
                key = (int(z), int(h), d, str(svc))
                st = zhd_service.get(key)
                if st is None:
                    st = RunningStats()
                    zhd_service[key] = st
                st.add(g["total_amount_std"].to_numpy(dtype=float, copy=False))

            # DF3 global: pu_location_id + hour + service_type
            for (z, h, svc), g in df.groupby(
                ["pu_location_id", "hour", "service_type"],
                dropna=False,
            ):
                key = (int(z), int(h), str(svc))
                pair = var_stats.get(key)
                if pair is None:
                    pair = (RunningStats(), Reservoir(k=reservoir_k, seed=42))
                    var_stats[key] = pair

                st, res = pair
                x = g["total_amount_std"].to_numpy(dtype=float, copy=False)
                st.add(x)
                res.add_many(x)

            del df
            gc.collect()

        # ---- construir y guardar SOLO lo mensual
        df_daily_service = finalize_types(monthly_daily_service_to_df(daily_stats, daily_zones))
        df_zone_hour_day_global = finalize_types(monthly_zone_hour_day_global_to_df(zhd_global, min_trips_df2))
        df_zone_hour_day_service = finalize_types(monthly_zone_hour_day_service_to_df(zhd_service, min_trips_df2))

        write_partitioned_dataset(
            df_daily_service,
            out_base / "df_daily_service",
            partition_cols=["service_type"],
        )
        write_partitioned_dataset(
            df_zone_hour_day_global,
            out_base / "df_zone_hour_day_global",
            partition_cols=["date"],
        )
        write_partitioned_dataset(
            df_zone_hour_day_service,
            out_base / "df_zone_hour_day_service",
            partition_cols=["date", "service_type"],
        )

        print(
            f"[INFO]   Guardado mes {year}-{month:02d}: "
            f"df1={len(df_daily_service)} filas, "
            f"df2a={len(df_zone_hour_day_global)} filas, "
            f"df2b={len(df_zone_hour_day_service)} filas"
        )

        # liberar acumuladores mensuales
        del daily_stats, daily_zones, zhd_global, zhd_service
        del df_daily_service, df_zone_hour_day_global, df_zone_hour_day_service
        gc.collect()

    # ---- DF3 global final
    rows3 = []
    for (z, h, svc), (st, res) in var_stats.items():
        if st.n < min_trips_df3:
            continue
        p75, p25 = res.percentiles((75, 25))
        pv = p75 - p25
        rows3.append(
            {
                "pu_location_id": z,
                "hour": h,
                "service_type": svc,
                "num_trips": st.n,
                "avg_price": st.mean(),
                "price_variability": pv,
                "biz_score": pv * math.log1p(st.n),
                "biz_score_iqr": pv,
            }
        )

    df_variability = pd.DataFrame(rows3)

    if not df_variability.empty:
        iqr = df_variability["price_variability"].astype(float)
        log_vol = np.log1p(df_variability["num_trips"].astype(float))

        iqr_mean, iqr_std = float(iqr.mean()), float(iqr.std())
        lv_mean, lv_std = float(log_vol.mean()), float(log_vol.std())

        z_iqr = (iqr - iqr_mean) / iqr_std if iqr_std > 0 else 0.0
        z_lv = (log_vol - lv_mean) / lv_std if lv_std > 0 else 0.0

        df_variability["biz_score_zsum"] = z_iqr + z_lv
        df_variability["biz_score_zproduct"] = z_iqr * z_lv

    df_variability = finalize_types(df_variability)

    return df_variability


# -----------------------------------------------------------------------------
# Guardado DF3 final
# -----------------------------------------------------------------------------
def save_df3_only(
    df_variability: pd.DataFrame,
    out_base: Path,
) -> None:
    write_partitioned_dataset(
        df_variability,
        out_base / "df_variability",
        partition_cols=["service_type"],
    )

    print("\nCapa 3 guardada en:", out_base)
    print(" - df_daily_service           ->", out_base / "df_daily_service")
    print(" - df_zone_hour_day_global    ->", out_base / "df_zone_hour_day_global")
    print(" - df_zone_hour_day_service   ->", out_base / "df_zone_hour_day_service")
    print(" - df_variability             ->", out_base / "df_variability")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Capa 3 (sin Spark): DF1/DF2 por mes y DF3 global, guardado particionado."
    )
    p.add_argument("--in-dir", default=str(obtener_ruta("data/standarized")), help="Ruta capa 2 (parquets)")
    p.add_argument("--out-dir", default=str(obtener_ruta("data/aggregated")), help="Salida capa 3")
    p.add_argument("--min-date", default="2023-01-01", help="YYYY-MM-DD (inclusive)")
    p.add_argument("--max-date", default="2025-12-31", help="YYYY-MM-DD (inclusive)")
    p.add_argument("--cap-max-price", type=float, default=500.0, help="Corte superior de precio (default: 500.0)")
    p.add_argument("--min-trips-df2", type=int, default=30, help="Mínimo num_trips en DF2a/DF2b (default: 30)")
    p.add_argument("--min-trips-df3", type=int, default=100, help="Mínimo num_trips en DF3 (default: 100)")
    p.add_argument("--mode", choices=["overwrite", "append"], default="overwrite", help="overwrite borra salida")
    p.add_argument("--reservoir-k", type=int, default=500, help="Tamaño reservoir para IQR aprox (default: 500)")
    args = p.parse_args()

    datetime.strptime(args.min_date, "%Y-%m-%d")
    datetime.strptime(args.max_date, "%Y-%m-%d")

    layer2_path = Path(args.in_dir)
    out_base = Path(args.out_dir).resolve()

    if args.mode == "overwrite" and out_base.exists():
        print(f"[INFO] Borrando salida (overwrite): {out_base}")
        shutil.rmtree(out_base)

    out_base.mkdir(parents=True, exist_ok=True)

    df3 = build_layer3_streaming_monthly(
        layer2_path=layer2_path,
        out_base=out_base,
        min_date=args.min_date,
        max_date=args.max_date,
        cap_max_price=args.cap_max_price,
        min_trips_df2=args.min_trips_df2,
        min_trips_df3=args.min_trips_df3,
        reservoir_k=args.reservoir_k,
    )

    if DEBUG:
        print(df3.head())

    save_df3_only(df3, out_base=out_base)


if __name__ == "__main__":
    main()
