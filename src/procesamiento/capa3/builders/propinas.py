from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    from config.settings import obtener_ruta  # type: ignore
except Exception:
    def obtener_ruta(p: str) -> Path:
        return Path(p)

console = Console()
MIN_ALLOWED_DATE = pd.Timestamp("2023-01-01").date()
MAX_ALLOWED_DATE = pd.Timestamp("2025-12-31").date()


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def read_partitioned_parquet_dir(base: Path) -> pd.DataFrame:
    base = Path(base).resolve()
    if not base.exists():
        raise FileNotFoundError(f"No existe el directorio parquet: {base}")

    files = sorted(base.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No hay archivos parquet dentro de: {base}")

    return pd.concat([pd.read_parquet(fp) for fp in files], ignore_index=True)


def ensure_cols(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: faltan columnas requeridas: {missing}")


def load_zone_lookup(project_root: Path, zone_lookup_path: str) -> pd.DataFrame | None:
    fp = (project_root / zone_lookup_path).resolve()
    if not fp.exists():
        console.print(f"[yellow]No existe zone lookup:[/yellow] {fp}")
        return None

    df = pd.read_csv(fp)
    rename_map = {}
    if "LocationID" in df.columns:
        rename_map["LocationID"] = "pu_location_id"
    if "Borough" in df.columns:
        rename_map["Borough"] = "borough"
    if "Zone" in df.columns:
        rename_map["Zone"] = "zone_name"
    df = df.rename(columns=rename_map)

    needed = [c for c in ["pu_location_id", "borough", "zone_name"] if c in df.columns]
    if "pu_location_id" not in needed or "borough" not in needed:
        console.print("[yellow]Zone lookup sin columnas esperadas; se omitirá borough.[/yellow]")
        return None

    df["pu_location_id"] = pd.to_numeric(df["pu_location_id"], errors="coerce").astype("Int64")
    df["borough"] = df["borough"].astype(str).str.strip().str.title()
    return df[needed].drop_duplicates()


# -----------------------------------------------------------------------------
# Enrichments locales (sin depender de src/ml/dataset)
# -----------------------------------------------------------------------------
def _add_boroughs(df: pd.DataFrame, zones: pd.DataFrame | None) -> pd.DataFrame:
    if zones is None:
        out = df.copy()
        out["borough"] = pd.NA
        out["borough_dropoff"] = pd.NA
        return out

    zones_pick = zones[["pu_location_id", "borough"]].rename(columns={"borough": "borough"})
    zones_drop = zones[["pu_location_id", "borough"]].rename(
        columns={"pu_location_id": "do_location_id", "borough": "borough_dropoff"}
    )
    out = df.merge(zones_pick, on="pu_location_id", how="left")
    out = out.merge(zones_drop, on="do_location_id", how="left")
    return out


def join_meteo(df: pd.DataFrame, project_root: Path, meteo_path: str) -> pd.DataFrame:
    fp = (project_root / meteo_path).resolve()
    if not fp.exists():
        console.print(f"[yellow]Meteo no encontrado:[/yellow] {fp}")
        return df

    met = pd.read_parquet(fp)
    if "date" not in met.columns or "hour" not in met.columns:
        return df

    met["date"] = pd.to_datetime(met["date"], errors="coerce").dt.date
    met["hour"] = pd.to_numeric(met["hour"], errors="coerce").astype("Int64")
    return df.merge(met, on=["date", "hour"], how="left")


def join_events(df: pd.DataFrame, project_root: Path, events_dir: str) -> pd.DataFrame:
    base = (project_root / events_dir).resolve()
    if not base.exists() or "borough" not in df.columns:
        return df

    ev = read_partitioned_parquet_dir(base)
    needed = [c for c in ["borough", "date", "hour"] if c in ev.columns]
    if len(needed) < 3:
        return df

    ev["date"] = pd.to_datetime(ev["date"], errors="coerce").dt.date
    ev["hour"] = pd.to_numeric(ev["hour"], errors="coerce").astype("Int64")
    ev["borough"] = ev["borough"].astype(str).str.strip().str.title()
    out = df.copy()
    out["borough"] = out["borough"].astype("string").str.strip().str.title()
    out = out.merge(ev, on=["borough", "date", "hour"], how="left")

    for c in ["n_events", "n_event_types"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    return out


def join_restaurants(df: pd.DataFrame, project_root: Path, base_dir: str = "data/aggregated/restaurants") -> pd.DataFrame:
    base = (project_root / base_dir).resolve()
    if not base.exists():
        return df

    out = df.copy()

    city_dir = base / "df_city_day"
    if city_dir.exists():
        city = read_partitioned_parquet_dir(city_dir)
        city["date"] = pd.to_datetime(city["date"], errors="coerce").dt.date
        out = out.merge(city, on=["date"], how="left")

    borough_dir = base / "df_borough_day"
    if borough_dir.exists() and "borough" in out.columns:
        bh = read_partitioned_parquet_dir(borough_dir)
        bh["date"] = pd.to_datetime(bh["date"], errors="coerce").dt.date
        bh["borough"] = bh["borough"].astype(str).str.strip().str.title()
        out["borough"] = out["borough"].astype("string").str.strip().str.title()
        out = out.merge(bh, on=["borough", "date"], how="left")

    borough_hour_dir = base / "df_borough_hour_day"
    if borough_hour_dir.exists() and "borough" in out.columns:
        bhh = read_partitioned_parquet_dir(borough_hour_dir)
        bhh["date"] = pd.to_datetime(bhh["date"], errors="coerce").dt.date
        bhh["hour"] = pd.to_numeric(bhh["hour"], errors="coerce").astype("Int64")
        bhh["borough"] = bhh["borough"].astype(str).str.strip().str.title()
        out = out.merge(bhh, on=["borough", "date", "hour"], how="left")

    return out


def join_rent(df: pd.DataFrame, project_root: Path, base_dir: str = "data/aggregated/rent") -> pd.DataFrame:
    base = (project_root / base_dir).resolve()
    if not base.exists():
        return df

    out = df.copy()

    city_dir = base / "df_city_day"
    if city_dir.exists():
        city = read_partitioned_parquet_dir(city_dir)
        city["date"] = pd.to_datetime(city["date"], errors="coerce").dt.date
        out = out.merge(city, on=["date"], how="left")

    borough_dir = base / "df_borough_day"
    if borough_dir.exists() and "borough" in out.columns:
        bh = read_partitioned_parquet_dir(borough_dir)
        bh["date"] = pd.to_datetime(bh["date"], errors="coerce").dt.date
        bh["borough"] = bh["borough"].astype(str).str.strip().str.title()
        out["borough"] = out["borough"].astype("string").str.strip().str.title()
        out = out.merge(bh, on=["borough", "date"], how="left")

    zone_dir = base / "df_zone_day"
    if zone_dir.exists() and "pu_location_id" in out.columns:
        zone = read_partitioned_parquet_dir(zone_dir)
        zone["date"] = pd.to_datetime(zone["date"], errors="coerce").dt.date
        if "zone_id" in zone.columns:
            zone["zone_id"] = pd.to_numeric(zone["zone_id"], errors="coerce").astype("Int64")
            out = out.merge(
                zone,
                left_on=["pu_location_id", "date"],
                right_on=["zone_id", "date"],
                how="left",
            )
            if "zone_id" in out.columns:
                out = out.drop(columns=["zone_id"])

    return out


# -----------------------------------------------------------------------------
# Builder EX1(b)
# -----------------------------------------------------------------------------
def build_tip_dataset(
    tip_base_dir: str = "data/aggregated/ex1b/df_tip_trip_level",
    meteo_path: str = "data/external/meteo/aggregated/df_hour_day/data.parquet",
    events_dir: str = "data/aggregated/events/df_borough_hour_day",
    zone_lookup_path: str = "data/external/taxi_zone_lookup.csv",
    out_path: str = "data/aggregated/ex1b/df_tip_dataset/data.parquet",
    date_from: str | None = None,
    date_to: str | None = None,
    sample_frac: float | None = None,
    random_state: int = 42,
    strict_apriori: bool = False,
) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[4]
    console.print(Panel.fit("[bold cyan]BUILDER EX1(b) - DATASET DE PROPINAS[/bold cyan]"))

    df = read_partitioned_parquet_dir(project_root / tip_base_dir)
    ensure_cols(df, ["date", "hour", "pu_location_id", "do_location_id", "target_tip_amount", "target_tip_pct"], "EX1B")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype("Int64")
    df["pu_location_id"] = pd.to_numeric(df["pu_location_id"], errors="coerce").astype("Int64")
    df["do_location_id"] = pd.to_numeric(df["do_location_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["date", "hour", "pu_location_id", "target_tip_amount"])
    df = df[(df["date"] >= MIN_ALLOWED_DATE) & (df["date"] <= MAX_ALLOWED_DATE)]

    if date_from is not None:
        df = df[df["date"] >= pd.to_datetime(date_from).date()]
    if date_to is not None:
        df = df[df["date"] <= pd.to_datetime(date_to).date()]
    if sample_frac is not None and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=random_state)

    zones = load_zone_lookup(project_root, zone_lookup_path)
    df = _add_boroughs(df, zones)
    df = join_meteo(df, project_root, meteo_path)
    df = join_events(df, project_root, events_dir)
    df = join_restaurants(df, project_root)
    df = join_rent(df, project_root)

    dt = pd.to_datetime(df["date"])
    df["day_of_week"] = dt.dt.dayofweek.astype(int)
    df["month"] = dt.dt.month.astype(int)
    df["day"] = dt.dt.day.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    df["is_night_hour"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    df["timestamp_hour"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"].fillna(0).astype(int), unit="h")

    # Guardrails para el caracter "a priori" del apartado b.
    if strict_apriori:
        drop_cols = [
            "dropoff_datetime",
            "trip_duration_min",
            "total_amount_std",
            "fare_amount",
            "tip_amount",
            "tip_pct",
        ]
        drop_cols = [c for c in drop_cols if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

    df = df.sort_values(["timestamp_hour", "pu_location_id"], kind="mergesort").reset_index(drop=True)

    out_fp = (project_root / out_path).resolve()
    os.makedirs(out_fp.parent, exist_ok=True)
    df.to_parquet(out_fp, index=False)

    cfg = Table(show_header=True, header_style="bold white", title="Configuracion builder EX1(b)")
    cfg.add_column("Campo", style="bold cyan")
    cfg.add_column("Valor")
    cfg.add_row("tip_base_dir", str(project_root / tip_base_dir))
    cfg.add_row("out_path", str(out_fp))
    cfg.add_row("strict_apriori", str(strict_apriori))
    cfg.add_row("filtro", f"{date_from or '...'} -> {date_to or '...'}")
    cfg.add_row("sample_frac", str(sample_frac))
    console.print(cfg)

    summary = Table(show_header=True, header_style="bold magenta", title="Resumen dataset EX1(b)")
    summary.add_column("Metrica", style="bold white")
    summary.add_column("Valor", justify="right")
    summary.add_row("Rows", f"{len(df):,}")
    summary.add_row("Cols", f"{len(df.columns):,}")
    summary.add_row("Min timestamp", str(df["timestamp_hour"].min() if not df.empty else "NA"))
    summary.add_row("Max timestamp", str(df["timestamp_hour"].max() if not df.empty else "NA"))
    console.print(summary)
    console.print(f"[bold green]OK[/bold green] Dataset EX1(b) guardado en: {out_fp}")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Builder EX1(b): construye dataset de propinas directamente desde capa 3.")
    p.add_argument("--from", dest="date_from", default=None, help="YYYY-MM-DD")
    p.add_argument("--to", dest="date_to", default=None, help="YYYY-MM-DD")
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--strict-apriori", action="store_true")
    p.add_argument("--out", default="data/aggregated/ex1b/df_tip_dataset/data.parquet")
    args = p.parse_args()

    build_tip_dataset(
        out_path=args.out,
        date_from=args.date_from,
        date_to=args.date_to,
        sample_frac=args.sample_frac,
        strict_apriori=args.strict_apriori,
    )


if __name__ == "__main__":
    main()
