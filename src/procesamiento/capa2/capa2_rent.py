from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

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
MIN_YEAR = 2023
MAX_YEAR = 2025


def _load_taxi_zone_lookup(project_root: Path) -> Optional[pd.DataFrame]:
    lookup_path = (project_root / "data" / "external" / "taxi_zone_lookup.csv").resolve()
    if not lookup_path.exists():
        console.print(
            f"[yellow]Aviso[/yellow] No existe {lookup_path}. "
            "No se podrá añadir metadata de taxi zones."
        )
        return None

    df = pd.read_csv(lookup_path)
    required = {"LocationID", "Borough", "Zone", "service_zone"}
    missing = required - set(df.columns)
    if missing:
        console.print(
            "[yellow]Aviso[/yellow] taxi_zone_lookup.csv sin columnas esperadas: "
            + ", ".join(sorted(missing))
        )
        return None

    out = df.rename(
        columns={
            "LocationID": "taxi_zone_id",
            "Borough": "taxi_borough",
            "Zone": "taxi_zone_name",
            "service_zone": "taxi_service_zone",
        }
    ).copy()
    out["taxi_zone_id"] = pd.to_numeric(out["taxi_zone_id"], errors="coerce").astype("Int64")
    for c in ["taxi_borough", "taxi_zone_name", "taxi_service_zone"]:
        out[c] = out[c].astype("string").str.strip()
    out = out.dropna(subset=["taxi_zone_id"]).drop_duplicates(subset=["taxi_zone_id"])
    return out[["taxi_zone_id", "taxi_borough", "taxi_zone_name", "taxi_service_zone"]]


def _map_rent_to_tlc_location_id(df: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    """
    Intenta mapear registros de rent a TLC LocationID (pu_location_id) usando punto
    (longitude, latitude) contra el shapefile oficial de taxi zones.

    Si no hay shapefile o geopandas, deja el dataset intacto.
    """
    out = df.copy()
    out["pu_location_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["taxi_zone_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["taxi_borough"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["taxi_zone_name"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["taxi_service_zone"] = pd.Series(pd.NA, index=out.index, dtype="string")

    if "latitude" not in out.columns or "longitude" not in out.columns:
        return out

    shp_path = (project_root / "data" / "external" / "taxi_zones" / "taxi_zones.shp").resolve()
    if not shp_path.exists():
        console.print(
            f"[yellow]Aviso[/yellow] No existe {shp_path}. "
            "Se conserva zone_id original (tracto censal) y pu_location_id quedará nulo."
        )
        return out

    try:
        import geopandas as gpd  # type: ignore
    except Exception:
        console.print(
            "[yellow]Aviso[/yellow] geopandas no disponible. "
            "No se puede mapear rent a LocationID TLC."
        )
        return out

    lat = pd.to_numeric(out["latitude"], errors="coerce")
    lon = pd.to_numeric(out["longitude"], errors="coerce")
    valid = lat.notna() & lon.notna()

    if not valid.any():
        return out

    zones = gpd.read_file(shp_path)
    if "LocationID" not in zones.columns or "geometry" not in zones.columns:
        console.print(
            "[yellow]Aviso[/yellow] taxi_zones.shp sin columnas esperadas (LocationID, geometry)."
        )
        return out

    zones = zones[["LocationID", "geometry"]].copy()
    zones["LocationID"] = pd.to_numeric(zones["LocationID"], errors="coerce").astype("Int64")
    zones = zones.dropna(subset=["LocationID", "geometry"])
    zones = zones.set_geometry("geometry")
    if zones.crs is None:
        zones = zones.set_crs(epsg=4326, allow_override=True)
    else:
        zones = zones.to_crs(epsg=4326)

    points = gpd.GeoDataFrame(
        out.loc[valid, ["latitude", "longitude"]].copy(),
        geometry=gpd.points_from_xy(lon.loc[valid], lat.loc[valid]),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(points, zones, how="left", predicate="within")
    mapped = pd.to_numeric(joined["LocationID"], errors="coerce").astype("Int64")
    out.loc[joined.index, "pu_location_id"] = mapped
    out["taxi_zone_id"] = out["pu_location_id"].astype("Int64")

    n_ok = int(out["pu_location_id"].notna().sum())
    n_total = int(valid.sum())
    console.print(f"[cyan]Mapeo rent -> TLC LocationID[/cyan] {n_ok}/{n_total} filas con coordenadas")

    lookup = _load_taxi_zone_lookup(project_root)
    if lookup is not None:
        out = out.drop(columns=["taxi_borough", "taxi_zone_name", "taxi_service_zone"], errors="ignore")
        out = out.merge(lookup, on="taxi_zone_id", how="left")

    return out


def _normalize_snapshot_date(series: pd.Series) -> pd.Series:
    """
    Normaliza source_snapshot_date a datetime.
    Soporta fechas ISO y formatos tipo '2024-acs5' (usa el año como fallback).
    """
    s = series.astype("string").str.strip()
    dt = pd.to_datetime(s, errors="coerce")

    # Fallback: extraemos YYYY y usamos YYYY-01-01 para poder particionar.
    year = s.str.extract(r"(?P<year>\d{4})")["year"]
    fallback = pd.to_datetime(year + "-01-01", errors="coerce")
    return dt.fillna(fallback)


def read_raw_rent(in_dir: Path) -> pd.DataFrame:
    in_dir = Path(in_dir).resolve()
    if not in_dir.exists():
        raise FileNotFoundError(f"No existe el directorio RAW: {in_dir}")

    files = sorted(in_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No hay .parquet en {in_dir}")

    console.print(f"[cyan]Lectura RAW[/cyan] {len(files)} snapshots rent")
    return pd.concat([pd.read_parquet(fp) for fp in files], ignore_index=True)


def filter_by_range(df: pd.DataFrame, date_from: str | None, date_to: str | None) -> pd.DataFrame:
    if "source_snapshot_date" not in df.columns or (date_from is None and date_to is None):
        return df

    out = df.copy()
    out["_date_ts"] = _normalize_snapshot_date(out["source_snapshot_date"])
    if date_from is not None:
        out = out[out["_date_ts"] >= pd.to_datetime(date_from)]
    if date_to is not None:
        out = out[out["_date_ts"] <= pd.to_datetime(date_to)]
    return out.drop(columns=["_date_ts"])


def build_layer2_rent(df: pd.DataFrame, project_root: Optional[Path] = None) -> pd.DataFrame:
    df2 = df.copy()
    df2["source_snapshot_date"] = _normalize_snapshot_date(df2["source_snapshot_date"]).dt.date
    for c in ["latitude", "longitude", "price", "price_moe"]:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
    for c in ["id", "accommodates", "minimum_nights", "availability_365"]:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce").astype("Int64")

    if "zone_id" in df2.columns:
        df2["zone_id"] = pd.to_numeric(df2["zone_id"], errors="coerce").astype("Int64")

    for c in ["zone_name", "borough", "neighborhood", "room_type", "property_type"]:
        if c in df2.columns:
            df2[c] = df2[c].astype("string").str.strip()

    # Intentamos homologar a ID de zona TLC (1..265) para facilitar joins con pu_location_id.
    if project_root is not None:
        df2 = _map_rent_to_tlc_location_id(df2, project_root=project_root)
        if "zone_id" in df2.columns:
            df2["source_zone_id"] = df2["zone_id"].astype("string")

        has_coords = pd.to_numeric(df2.get("latitude"), errors="coerce").notna() & pd.to_numeric(
            df2.get("longitude"), errors="coerce"
        ).notna()
        n_total = int(has_coords.sum())
        n_mapped = int(pd.to_numeric(df2.get("taxi_zone_id"), errors="coerce").notna().sum())
        coverage = (n_mapped / n_total) if n_total > 0 else 0.0
        console.print(
            f"[cyan]Cobertura rent -> taxi_zone_id[/cyan] "
            f"{n_mapped}/{n_total} ({coverage:.1%})"
        )

    df2 = df2.dropna(subset=["source_snapshot_date", "borough", "price"])
    df2 = df2[df2["price"] > 0]

    dt = pd.to_datetime(df2["source_snapshot_date"])
    df2["year"] = dt.dt.year.astype("int")
    df2["month"] = dt.dt.month.astype("int")
    df2 = df2[(df2["year"] >= MIN_YEAR) & (df2["year"] <= MAX_YEAR)]

    cols = [
        "id", "zone_id", "source_zone_id", "pu_location_id", "taxi_zone_id",
        "taxi_borough", "taxi_zone_name", "taxi_service_zone",
        "zone_name", "source_snapshot_date",
        "year", "month", "borough", "neighborhood",
        "latitude", "longitude", "room_type", "property_type", "accommodates", "minimum_nights",
        "availability_365", "price", "price_moe",
    ]
    cols = [c for c in cols if c in df2.columns]
    return df2[cols].drop_duplicates()


def save_layer2_rent(df: pd.DataFrame, out_dir: Path) -> int:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files_written = 0
    for (y, m), g in df.groupby(["year", "month"], dropna=False):
        file_name = f"rent_{int(y)}-{int(m):02d}.parquet"
        g.to_parquet(out_dir / file_name, index=False, engine="pyarrow")
        files_written += 1
    return files_written


def main():
    console.print(Panel.fit("[bold cyan]CAPA 2 - RENT: TIPADO + HIGIENE + PARTICIONADO[/bold cyan]"))

    p = argparse.ArgumentParser(description="Capa 2 Rent: tipado + higiene + particionado.")
    p.add_argument("--from", dest="date_from", default=None, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--to", dest="date_to", default=None, help="YYYY-MM-DD (inclusive)")
    args = p.parse_args()

    if args.date_from is not None:
        datetime.strptime(args.date_from, "%Y-%m-%d")
    if args.date_to is not None:
        datetime.strptime(args.date_to, "%Y-%m-%d")

    project_root = Path(__file__).resolve().parents[3]
    validated_dir = (project_root / "data" / "external" / "rent" / "validated").resolve()
    raw_dir = (project_root / "data" / "external" / "rent" / "raw").resolve()
    in_dir = validated_dir if validated_dir.exists() else raw_dir
    out_dir = (project_root / "data" / "external" / "rent" / "standarized").resolve()

    cfg = Table(show_header=True, header_style="bold white", title="Configuracion Capa2 Rent")
    cfg.add_column("Campo", style="bold cyan")
    cfg.add_column("Valor")
    cfg.add_row("in_dir", str(in_dir))
    cfg.add_row("out_dir", str(out_dir))
    cfg.add_row("filtro", f"{args.date_from or '...'} -> {args.date_to or '...'}")
    console.print(cfg)

    with console.status("[cyan]Procesando rent...[/cyan]"):
        df_raw = read_raw_rent(in_dir)
        df_raw = filter_by_range(df_raw, args.date_from, args.date_to)
        df_l2 = build_layer2_rent(df_raw, project_root=project_root)
        files_written = save_layer2_rent(df_l2, out_dir)

    summary = Table(show_header=True, header_style="bold magenta", title="Resumen Capa2 Rent")
    summary.add_column("Metrica", style="bold white")
    summary.add_column("Valor", justify="right")
    summary.add_row("Rows raw", f"{len(df_raw):,}")
    summary.add_row("Rows capa2", f"{len(df_l2):,}")
    summary.add_row("Parquets escritos", f"{files_written:,}")
    summary.add_row("Salida", str(out_dir))
    console.print(summary)
    console.print("[bold green]OK[/bold green] Capa 2 RENT completada")


if __name__ == "__main__":
    main()
