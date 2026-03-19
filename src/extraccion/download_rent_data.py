from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import click
import pandas as pd
import requests
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from config.settings import obtener_ruta


GET_THE_DATA_URL = "https://insideairbnb.com/get-the-data/"
TIMEOUT = 120
DEFAULT_OUT_DIR = obtener_ruta("data/external/rent/raw")
DEFAULT_TAG = "rent_insideairbnb"

PRICE_MAX_REASONABLE = 10_000.0
ACS_YEAR = 2024
ACS_API_URL = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"
ACS_RENT_VAR = "B25064_001E"
ACS_RENT_MOE_VAR = "B25064_001M"
TIGER_TRACTS_URL = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/0/query"
)
NYC_COUNTIES = {
    "005": "Bronx",
    "047": "Brooklyn",
    "061": "Manhattan",
    "081": "Queens",
    "085": "Staten Island",
}

console = Console()


@dataclass(frozen=True)
class RentSource:
    url: str
    snapshot_date: str
    dataset_kind: str
    calendar_url: Optional[str] = None


def _download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"[dim]  -> Descargando: {url}[/dim]")

    with requests.get(url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[cyan]{task.fields[mb]} MB"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Descargando", total=total or None, mb=0.0)

            downloaded = 0
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress.update(task, completed=downloaded if total else None, mb=downloaded / 1e6)

    console.print("[dim]  -> Descarga completada[/dim]")


def _read_insideairbnb_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="infer", low_memory=False)


def _coerce_price(series: pd.Series, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(float("nan"), index=index, dtype="float64")

    cleaned = (
        series.astype("string")
        .str.replace(r"[^0-9,\.\-]", "", regex=True)
        .str.replace(",", ".", regex=False)
        .replace({"": pd.NA, "nan": pd.NA, "<NA>": pd.NA, "None": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _extract_stats(df: pd.DataFrame) -> dict[str, Any]:
    price_col = _first_existing_column(df, ["price"])
    borough_col = _first_existing_column(df, ["borough", "neighbourhood_group_cleansed", "neighbourhood_group"])
    lat_col = _first_existing_column(df, ["latitude"])
    lon_col = _first_existing_column(df, ["longitude"])

    prices = _coerce_price(df[price_col], df.index) if price_col else pd.Series(pd.NA, index=df.index)

    return {
        "rows": int(len(df)),
        "price_non_null": int(prices.notna().sum()),
        "price_min": float(prices.min()) if prices.notna().any() else None,
        "price_max": float(prices.max()) if prices.notna().any() else None,
        "borough_non_null": int(df[borough_col].notna().sum()) if borough_col else 0,
        "latitude_non_null": int(df[lat_col].notna().sum()) if lat_col else 0,
        "longitude_non_null": int(df[lon_col].notna().sum()) if lon_col else 0,
    }


def _validate_rent_snapshot(
    df: pd.DataFrame,
    source: RentSource,
    allow_missing_price: bool = False,
) -> dict[str, Any]:
    required_groups = {
        "borough": ["borough", "neighbourhood_group_cleansed", "neighbourhood_group"],
        "neighborhood": ["neighborhood", "neighbourhood_cleansed", "neighbourhood"],
        "latitude": ["latitude"],
        "longitude": ["longitude"],
        "room_type": ["room_type"],
        "price": ["price"],
    }

    missing_groups = [
        logical_name
        for logical_name, candidates in required_groups.items()
        if _first_existing_column(df, candidates) is None
    ]
    if missing_groups:
        raise ValueError(
            "El dataset de alquiler no contiene las columnas esperadas para NYC: "
            + ", ".join(sorted(missing_groups))
        )

    stats = _extract_stats(df)

    if stats["rows"] == 0:
        raise ValueError("El dataset de alquiler se ha descargado vacío.")
    if not allow_missing_price and stats["price_non_null"] == 0:
        raise ValueError(
            "El dataset de alquiler no trae precios válidos. "
            f"Fuente recibida: {source.url}"
        )
    if stats["borough_non_null"] == 0 or stats["latitude_non_null"] == 0 or stats["longitude_non_null"] == 0:
        raise ValueError(
            "El dataset de alquiler no trae la geografía necesaria para análisis por zona."
        )

    return stats


def _discover_latest_insideairbnb_nyc(dataset_kind: str = "summary") -> RentSource:
    dataset_kind = dataset_kind.lower()
    if dataset_kind not in {"summary", "detailed"}:
        raise ValueError(f"dataset_kind no soportado: {dataset_kind}")

    with requests.get(GET_THE_DATA_URL, timeout=TIMEOUT) as r:
        r.raise_for_status()
        html = r.text

    pattern = re.compile(
        r"https://data\.insideairbnb\.com/united-states/ny/new-york-city/"
        r"(?P<snapshot>\d{4}-\d{2}-\d{2})/"
        r"(?P<section>visualisations|data)/"
        r"(?P<file>listings\.csv(?:\.gz)?)"
    )

    candidates: list[RentSource] = []
    for match in pattern.finditer(html):
        section = match.group("section")
        file_name = match.group("file")

        kind = None
        if section == "visualisations" and file_name == "listings.csv":
            kind = "summary"
        elif section == "data" and file_name == "listings.csv.gz":
            kind = "detailed"

        if kind != dataset_kind:
            continue

        candidates.append(
            RentSource(
                url=match.group(0),
                snapshot_date=match.group("snapshot"),
                dataset_kind=kind,
                calendar_url=(
                    "https://data.insideairbnb.com/united-states/ny/new-york-city/"
                    f"{match.group('snapshot')}/data/calendar.csv.gz"
                ),
            )
        )

    if not candidates:
        raise ValueError(
            "No se pudo descubrir un snapshot de alquiler para New York City en Inside Airbnb."
        )

    candidates.sort(key=lambda item: item.snapshot_date, reverse=True)
    return candidates[0]


def _resolve_rent_source(url: Optional[str], dataset_kind: str) -> RentSource:
    if url:
        match = re.search(
            r"/new-york-city/(?P<snapshot>\d{4}-\d{2}-\d{2})/(?P<section>visualisations|data)/",
            url,
        )
        if match:
            inferred_kind = "summary" if match.group("section") == "visualisations" else "detailed"
            return RentSource(
                url=url,
                snapshot_date=match.group("snapshot"),
                dataset_kind=inferred_kind,
                calendar_url=(
                    "https://data.insideairbnb.com/united-states/ny/new-york-city/"
                    f"{match.group('snapshot')}/data/calendar.csv.gz"
                ),
            )
        return RentSource(
            url=url,
            snapshot_date="unknown",
            dataset_kind=dataset_kind,
            calendar_url=None,
        )

    return _discover_latest_insideairbnb_nyc(dataset_kind=dataset_kind)


def _dataset_is_current(final_parquet: Path, source: RentSource) -> bool:
    if not final_parquet.exists():
        return False

    try:
        df_existing = pd.read_parquet(final_parquet, columns=["source_url", "source_snapshot_date", "price"])
    except Exception:
        return False

    if df_existing.empty:
        return False

    current_url = str(df_existing["source_url"].iloc[0]) if "source_url" in df_existing.columns else None
    current_snapshot = (
        str(df_existing["source_snapshot_date"].iloc[0])
        if "source_snapshot_date" in df_existing.columns
        else None
    )
    price_non_null = int(_coerce_price(df_existing["price"], df_existing.index).notna().sum())

    return current_url == source.url and current_snapshot == source.snapshot_date and price_non_null > 0


def _download_acs_rent_snapshot() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for county_code, borough in NYC_COUNTIES.items():
        params = {
            "get": f"NAME,{ACS_RENT_VAR},{ACS_RENT_MOE_VAR}",
            "for": "tract:*",
            "in": f"state:36 county:{county_code}",
        }
        resp = requests.get(ACS_API_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()

        if len(payload) < 2:
            continue

        county_df = pd.DataFrame(payload[1:], columns=payload[0])
        county_df["borough"] = borough
        rows.append(county_df)

    if not rows:
        raise ValueError("La API ACS no devolvió filas para los tractos de NYC.")

    df = pd.concat(rows, ignore_index=True)
    df["zone_id"] = df["state"].astype(str) + df["county"].astype(str) + df["tract"].astype(str)
    df["id"] = pd.to_numeric(df["zone_id"], errors="coerce")
    df["price"] = pd.to_numeric(df[ACS_RENT_VAR], errors="coerce")
    df["price_moe"] = pd.to_numeric(df[ACS_RENT_MOE_VAR], errors="coerce")
    df.loc[df["price"] <= 0, "price"] = pd.NA
    df.loc[df["price_moe"] < 0, "price_moe"] = pd.NA

    centroids = _download_tiger_tract_centroids()
    df = df.merge(centroids, on="zone_id", how="left")

    df["source_snapshot_date"] = f"{ACS_YEAR}-acs5"
    df["neighborhood"] = df["NAME"]
    df["room_type"] = "All Rentals"
    df["property_type"] = "Census Tract"
    df["accommodates"] = pd.NA
    df["minimum_nights"] = pd.NA
    df["availability_365"] = pd.NA

    cols = [
        "id",
        "source_snapshot_date",
        "borough",
        "neighborhood",
        "latitude",
        "longitude",
        "room_type",
        "property_type",
        "accommodates",
        "minimum_nights",
        "availability_365",
        "price",
        "price_moe",
        "zone_id",
    ]
    return df[cols]


def _download_tiger_tract_centroids() -> pd.DataFrame:
    where = "STATE='36' AND COUNTY IN ('005','047','061','081','085')"
    params = {
        "where": where,
        "outFields": "GEOID,INTPTLAT,INTPTLON,COUNTY,NAME",
        "returnGeometry": "false",
        "f": "json",
    }
    resp = requests.get(TIGER_TRACTS_URL, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()

    features = payload.get("features") or []
    if not features:
        raise ValueError("Tigerweb no devolvió centroides de tractos para NYC.")

    rows = [feature.get("attributes", {}) for feature in features]
    df = pd.DataFrame(rows)
    df = df.rename(columns={
        "GEOID": "zone_id",
        "INTPTLAT": "latitude",
        "INTPTLON": "longitude",
    })
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    return df[["zone_id", "latitude", "longitude"]]


def _aggregate_calendar_prices(calendar_path: Path, chunk_size: int = 500_000) -> pd.DataFrame:
    sum_all = pd.Series(dtype="float64")
    count_all = pd.Series(dtype="float64")
    sum_avail = pd.Series(dtype="float64")
    count_avail = pd.Series(dtype="float64")

    for chunk in pd.read_csv(
        calendar_path,
        compression="infer",
        usecols=lambda c: c in {"listing_id", "available", "price", "adjusted_price"},
        chunksize=chunk_size,
        low_memory=False,
    ):
        listing_id = pd.to_numeric(chunk["listing_id"], errors="coerce")
        base_price = chunk["adjusted_price"] if "adjusted_price" in chunk.columns else chunk["price"]
        price = _coerce_price(base_price.fillna(chunk["price"]), chunk.index)
        available = chunk["available"].astype("string").str.strip().str.lower().isin(["t", "true", "1", "yes"])

        tmp = pd.DataFrame({
            "listing_id": listing_id,
            "price": price,
            "available": available,
        }).dropna(subset=["listing_id", "price"])

        if tmp.empty:
            continue

        tmp["listing_id"] = tmp["listing_id"].astype("int64")

        grouped_all = tmp.groupby("listing_id", observed=True)["price"].agg(["sum", "count"])
        sum_all = sum_all.add(grouped_all["sum"], fill_value=0)
        count_all = count_all.add(grouped_all["count"], fill_value=0)

        tmp_avail = tmp.loc[tmp["available"]]
        if not tmp_avail.empty:
            grouped_avail = tmp_avail.groupby("listing_id", observed=True)["price"].agg(["sum", "count"])
            sum_avail = sum_avail.add(grouped_avail["sum"], fill_value=0)
            count_avail = count_avail.add(grouped_avail["count"], fill_value=0)

    if sum_all.empty:
        return pd.DataFrame(columns=["id", "price"])

    avg_all = (sum_all / count_all).rename("price_all")
    avg_avail = (sum_avail / count_avail).rename("price_available")

    merged = pd.concat([avg_avail, avg_all], axis=1)
    merged["price"] = merged["price_available"].fillna(merged["price_all"])
    merged = merged[["price"]].reset_index().rename(columns={"listing_id": "id"})
    return merged


def _fill_prices_from_calendar(
    df: pd.DataFrame,
    source: RentSource,
    calendar_path: Path,
) -> pd.DataFrame:
    if not source.calendar_url:
        raise ValueError(
            "No se puede derivar precio desde calendar.csv.gz porque la URL del calendario no es conocida."
        )

    _download_file(source.calendar_url, calendar_path)
    price_by_listing = _aggregate_calendar_prices(calendar_path)

    if price_by_listing.empty:
        raise ValueError("No se pudieron derivar precios válidos desde calendar.csv.gz.")

    df = df.copy()
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    if "price" in df.columns:
        df["price"] = _coerce_price(df["price"], df.index)
    else:
        df["price"] = pd.Series(float("nan"), index=df.index, dtype="float64")

    df = df.merge(price_by_listing, on="id", how="left", suffixes=("", "_calendar"))
    df["price"] = df["price"].fillna(df["price_calendar"])
    df = df.drop(columns=["price_calendar"])
    return df


def download_rent_snapshot(
    url: Optional[str] = None,
    out_dir: Optional[Path] = None,
    tag: str = DEFAULT_TAG,
    dataset_kind: str = "summary",
    force: bool = False,
    provider: str = "acs",
) -> dict[str, Any]:
    """
    Descarga un snapshot de alquiler de NYC y lo guarda en parquet.

    El proveedor por defecto es ACS a nivel de tracto censal. Inside Airbnb queda
    disponible como opción manual, pero sus snapshots recientes de NYC no están
    publicando precios utilizables.
    """
    out_dir = out_dir or DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    provider = "insideairbnb" if url else provider.lower()
    if provider not in {"acs", "insideairbnb"}:
        raise ValueError(f"provider no soportado: {provider}")

    if provider == "acs":
        source = RentSource(
            url=ACS_API_URL,
            snapshot_date=f"{ACS_YEAR}-acs5",
            dataset_kind="acs_tract",
            calendar_url=None,
        )
    else:
        source = _resolve_rent_source(url=url, dataset_kind=dataset_kind)

    final_parquet = out_dir / f"{tag}.parquet"
    tmp_download = out_dir / f"__tmp_{tag}{Path(source.url.split('?')[0]).suffix}"
    tmp_parquet = out_dir / f"__tmp_{tag}.parquet"
    tmp_calendar = out_dir / f"__tmp_{tag}_calendar.csv.gz"

    if not force and _dataset_is_current(final_parquet, source):
        console.print(
            f"[yellow]SKIP:[/yellow] {final_parquet.name} ya está alineado con "
            f"{source.dataset_kind} {source.snapshot_date}"
        )
        return {
            "successful": 1,
            "total": 1,
            "skipped": 1,
            "failed": 0,
            "snapshot_date": source.snapshot_date,
            "dataset_kind": source.dataset_kind,
            "rows": None,
        }

    try:
        if provider == "acs":
            console.print(
                "[dim]  -> Descargando renta residencial por tracto censal desde ACS/TIGERweb...[/dim]"
            )
            df = _download_acs_rent_snapshot()
            stats = _validate_rent_snapshot(df, source)
        else:
            _download_file(source.url, tmp_download)

            console.print("[dim]  -> Leyendo CSV y generando Parquet...[/dim]")
            df = _read_insideairbnb_csv(tmp_download)
            stats = _validate_rent_snapshot(df, source, allow_missing_price=True)

            if stats["price_non_null"] == 0:
                console.print(
                    "[yellow]Aviso:[/yellow] listings no trae precio. "
                    "Derivando precio medio por listing desde calendar.csv.gz..."
                )
                df = _fill_prices_from_calendar(df, source, tmp_calendar)

            stats = _validate_rent_snapshot(df, source)

        df["source_url"] = source.url
        df["source_snapshot_date"] = source.snapshot_date
        df["source_dataset_kind"] = source.dataset_kind
        df["source_city"] = "new-york-city"

        df.to_parquet(tmp_parquet, engine="pyarrow", index=False)
        tmp_parquet.replace(final_parquet)
    finally:
        for tmp in (tmp_download, tmp_parquet, tmp_calendar):
            try:
                tmp.unlink()
            except Exception:
                pass

    console.print(
        f"[green]OK:[/green] {final_parquet.name} "
        f"({stats['rows']} filas, price_non_null={stats['price_non_null']}, "
        f"snapshot={source.snapshot_date}, kind={source.dataset_kind})"
    )

    return {
        "successful": 1,
        "total": 1,
        "skipped": 0,
        "failed": 0,
        "rows": stats["rows"],
        "price_non_null": stats["price_non_null"],
        "snapshot_date": source.snapshot_date,
        "dataset_kind": source.dataset_kind,
    }


@click.command()
@click.option("--url", default=None, help="URL manual de Inside Airbnb. Si se pasa, fuerza el proveedor `insideairbnb`.")
@click.option("--out-dir", default=None, help="Ruta destino (si no se pasa, usa la de settings)")
@click.option("--tag", default=DEFAULT_TAG, show_default=True, help="Nombre base del parquet final")
@click.option(
    "--provider",
    type=click.Choice(["acs", "insideairbnb"]),
    default="acs",
    show_default=True,
    help="Proveedor de datos de alquiler",
)
@click.option(
    "--dataset-kind",
    type=click.Choice(["summary", "detailed"]),
    default="summary",
    show_default=True,
    help="Tipo de dataset de Inside Airbnb cuando el proveedor es `insideairbnb`",
)
@click.option("--force", is_flag=True, help="Fuerza redescarga aunque ya exista un snapshot válido")
def main(
    url: Optional[str],
    out_dir: Optional[str],
    tag: str,
    provider: str,
    dataset_kind: str,
    force: bool,
) -> None:
    """
    Descarga un snapshot de alquiler de NYC.

    Ejemplos:

        uv run -m src.extraccion.download_rent_data
        uv run -m src.extraccion.download_rent_data --provider acs
        uv run -m src.extraccion.download_rent_data --dataset-kind detailed
        uv run -m src.extraccion.download_rent_data --force
    """
    out_path = Path(out_dir) if out_dir else None
    download_rent_snapshot(
        url=url,
        out_dir=out_path,
        tag=tag,
        provider=provider,
        dataset_kind=dataset_kind,
        force=force,
    )


if __name__ == "__main__":
    main()
