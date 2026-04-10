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


def _load_valid_taxi_location_ids(project_root: Path) -> Optional[set[int]]:
    lookup_path = (project_root / "data" / "external" / "taxi_zone_lookup.csv").resolve()
    if not lookup_path.exists():
        console.print(
            f"[yellow]Aviso[/yellow] No existe {lookup_path}. "
            "No se validarán LocationID contra taxi_zone_lookup."
        )
        return None

    df = pd.read_csv(lookup_path)
    if "LocationID" not in df.columns:
        console.print("[yellow]Aviso[/yellow] taxi_zone_lookup.csv sin columna LocationID.")
        return None

    ids = pd.to_numeric(df["LocationID"], errors="coerce").dropna().astype("int64").unique().tolist()
    return set(ids)


def _map_restaurants_to_tlc_location_id(df: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    """
    Mapea restaurantes a TLC LocationID (pu_location_id) usando (longitude, latitude)
    contra taxi_zones.shp.
    """
    out = df.copy()
    out["pu_location_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")

    if "latitude" not in out.columns or "longitude" not in out.columns:
        return out

    shp_path = (project_root / "data" / "external" / "taxi_zones" / "taxi_zones.shp").resolve()
    if not shp_path.exists():
        console.print(
            f"[yellow]Aviso[/yellow] No existe {shp_path}. "
            "No se podrá mapear restaurants a pu_location_id."
        )
        return out

    try:
        import geopandas as gpd  # type: ignore
    except Exception:
        console.print(
            "[yellow]Aviso[/yellow] geopandas no disponible. "
            "No se puede mapear restaurants a pu_location_id."
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

    valid_ids = _load_valid_taxi_location_ids(project_root)
    if valid_ids is not None:
        as_num = pd.to_numeric(out["pu_location_id"], errors="coerce")
        in_lookup = as_num.isin(valid_ids)
        out.loc[~in_lookup, "pu_location_id"] = pd.NA

    n_ok = int(out["pu_location_id"].notna().sum())
    n_total = int(valid.sum())
    console.print(f"[cyan]Mapeo restaurants -> pu_location_id[/cyan] {n_ok}/{n_total} filas con coordenadas")
    return out


def read_raw_restaurants(in_dir: Path) -> pd.DataFrame:
    in_dir = Path(in_dir).resolve()
    if not in_dir.exists():
        raise FileNotFoundError(f"No existe el directorio RAW: {in_dir}")
    files = sorted(in_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No hay .parquet en {in_dir}")
    console.print(f"[cyan]Lectura RAW[/cyan] {len(files)} snapshots restaurants")
    return pd.concat([pd.read_parquet(fp) for fp in files], ignore_index=True)


def filter_by_range(df: pd.DataFrame, date_from: str | None, date_to: str | None) -> pd.DataFrame:
    if date_from is None and date_to is None:
        return df
    out = df.copy()
    out["_date_ts"] = pd.to_datetime(out["inspection_date"], errors="coerce")
    if date_from is not None:
        out = out[out["_date_ts"] >= pd.to_datetime(date_from)]
    if date_to is not None:
        out = out[out["_date_ts"] <= pd.to_datetime(date_to)]
    return out.drop(columns=["_date_ts"])


def build_layer2_restaurants(df: pd.DataFrame, project_root: Optional[Path] = None) -> pd.DataFrame:
    df2 = df.copy()

    # Compatibilidad entre esquemas:
    # - RAW historico: boro / dba
    # - Capa 1 validada: borough / restaurant_name
    if "boro" not in df2.columns and "borough" in df2.columns:
        df2["boro"] = df2["borough"]
    if "dba" not in df2.columns and "restaurant_name" in df2.columns:
        df2["dba"] = df2["restaurant_name"]

    for c in ["inspection_date", "grade_date", "record_date"]:
        if c in df2.columns:
            df2[c] = pd.to_datetime(df2[c], errors="coerce")
    for c in ["camis", "score"]:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce").astype("Int64")
    for c in ["latitude", "longitude"]:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")

    for c in ["boro", "dba", "cuisine_description", "critical_flag", "grade", "inspection_type"]:
        if c in df2.columns:
            df2[c] = df2[c].astype("string").str.strip()

    if project_root is not None:
        df2 = _map_restaurants_to_tlc_location_id(df2, project_root=project_root)

    missing_required = [c for c in ["inspection_date", "boro", "camis"] if c not in df2.columns]
    if missing_required:
        raise ValueError(
            "No se puede construir capa 2 de restaurants: faltan columnas requeridas "
            f"{missing_required}. Verifica salida de capa1 o esquema RAW."
        )

    df2 = df2.dropna(subset=["inspection_date", "boro", "camis"])
    df2["date"] = df2["inspection_date"].dt.date
    dt = pd.to_datetime(df2["date"])
    df2["year"] = dt.dt.year.astype("int")
    df2["month"] = dt.dt.month.astype("int")
    df2["hour"] = df2["inspection_date"].dt.hour.fillna(12).astype("int")
    df2 = df2[(df2["year"] >= MIN_YEAR) & (df2["year"] <= MAX_YEAR)]
    df2 = df2[(df2["hour"] >= 0) & (df2["hour"] <= 23)]

    cols = [
        "camis", "dba", "boro", "date", "inspection_date", "year", "month", "hour", "cuisine_description",
        "action", "violation_code", "violation_description", "critical_flag", "score", "grade",
        "grade_date", "record_date", "inspection_type", "latitude", "longitude",
        "pu_location_id",
        "community_board", "council_district", "census_tract", "bin", "bbl", "nta",
    ]
    cols = [c for c in cols if c in df2.columns]
    return df2[cols].drop_duplicates()


def save_layer2_restaurants(df: pd.DataFrame, out_dir: Path) -> int:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    files_written = 0
    for (y, m), g in df.groupby(["year", "month"], dropna=False):
        file_name = f"restaurants_{int(y)}-{int(m):02d}.parquet"
        g.to_parquet(out_dir / file_name, index=False, engine="pyarrow")
        files_written += 1
    return files_written


def main():
    console.print(Panel.fit("[bold cyan]CAPA 2 - RESTAURANTS: TIPADO + HIGIENE + PARTICIONADO[/bold cyan]"))
    p = argparse.ArgumentParser(description="Capa 2 Restaurants: tipado + higiene + particionado.")
    p.add_argument("--from", dest="date_from", default=None, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--to", dest="date_to", default=None, help="YYYY-MM-DD (inclusive)")
    args = p.parse_args()

    if args.date_from is not None:
        datetime.strptime(args.date_from, "%Y-%m-%d")
    if args.date_to is not None:
        datetime.strptime(args.date_to, "%Y-%m-%d")

    project_root = Path(__file__).resolve().parents[3]
    validated_dir = (project_root / "data" / "external" / "restaurants" / "validated").resolve()
    raw_dir = (project_root / "data" / "external" / "restaurants" / "raw").resolve()
    in_dir = validated_dir if validated_dir.exists() else raw_dir
    out_dir = (project_root / "data" / "external" / "restaurants" / "standarized").resolve()

    cfg = Table(show_header=True, header_style="bold white", title="Configuracion Capa2 Restaurants")
    cfg.add_column("Campo", style="bold cyan")
    cfg.add_column("Valor")
    cfg.add_row("in_dir", str(in_dir))
    cfg.add_row("out_dir", str(out_dir))
    cfg.add_row("filtro", f"{args.date_from or '...'} -> {args.date_to or '...'}")
    console.print(cfg)

    with console.status("[cyan]Procesando restaurants...[/cyan]"):
        df_raw = read_raw_restaurants(in_dir)
        df_raw = filter_by_range(df_raw, args.date_from, args.date_to)
        df_l2 = build_layer2_restaurants(df_raw, project_root=project_root)
        files_written = save_layer2_restaurants(df_l2, out_dir)

    summary = Table(show_header=True, header_style="bold magenta", title="Resumen Capa2 Restaurants")
    summary.add_column("Metrica", style="bold white")
    summary.add_column("Valor", justify="right")
    summary.add_row("Rows raw", f"{len(df_raw):,}")
    summary.add_row("Rows capa2", f"{len(df_l2):,}")
    summary.add_row("Parquets escritos", f"{files_written:,}")
    summary.add_row("Salida", str(out_dir))
    console.print(summary)
    console.print("[bold green]OK[/bold green] Capa 2 RESTAURANTS completada")


if __name__ == "__main__":
    main()
