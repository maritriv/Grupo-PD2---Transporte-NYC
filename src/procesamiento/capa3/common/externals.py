from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .constants import ALLOWED_MAX_DATE, ALLOWED_MIN_DATE, console
from .io import list_all_parquets


def load_meteo_features(
    meteo_base: Path,
    min_date: str,
    max_date: str,
) -> pd.DataFrame:
    files = list_all_parquets(meteo_base)
    if not files:
        console.print(f"[yellow]Aviso:[/yellow] No se encontraron parquets de meteo en {meteo_base}")
        return pd.DataFrame(columns=["date", "hour", "temp_c", "precip_mm"])

    dfs: List[pd.DataFrame] = []
    keep_cols = ["date", "hour", "temp_c", "precip_mm"]

    for fp in files:
        df = pd.read_parquet(fp, columns=keep_cols).copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype("Int32")
        df["temp_c"] = pd.to_numeric(df["temp_c"], errors="coerce")
        df["precip_mm"] = pd.to_numeric(df["precip_mm"], errors="coerce")
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True).dropna(subset=["date", "hour"])

    dmin = max(pd.to_datetime(min_date), ALLOWED_MIN_DATE)
    dmax = min(pd.to_datetime(max_date), ALLOWED_MAX_DATE)
    out = out[(out["date"] >= dmin) & (out["date"] <= dmax)]
    out = out[(out["hour"] >= 0) & (out["hour"] <= 23)]

    out = (
        out.groupby(["date", "hour"], dropna=False)
        .agg(
            temp_c=("temp_c", "mean"),
            precip_mm=("precip_mm", "mean"),
        )
        .reset_index()
    )

    out["temp_c"] = pd.to_numeric(out["temp_c"], errors="coerce").astype("float32")
    out["precip_mm"] = pd.to_numeric(out["precip_mm"], errors="coerce").astype("float32")
    return out


def load_event_features(
    events_base: Path,
    min_date: str,
    max_date: str,
) -> pd.DataFrame:
    files = list_all_parquets(events_base)
    if not files:
        console.print(f"[yellow]Aviso:[/yellow] No se encontraron parquets de eventos en {events_base}")
        return pd.DataFrame(columns=["date", "hour", "city_n_events", "city_has_event"])

    dfs: List[pd.DataFrame] = []
    keep_cols = ["date", "hour", "n_events"]

    for fp in files:
        df = pd.read_parquet(fp, columns=keep_cols).copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype("Int32")
        df["n_events"] = pd.to_numeric(df["n_events"], errors="coerce").astype("Int32")
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True).dropna(subset=["date", "hour", "n_events"])

    dmin = max(pd.to_datetime(min_date), ALLOWED_MIN_DATE)
    dmax = min(pd.to_datetime(max_date), ALLOWED_MAX_DATE)
    out = out[(out["date"] >= dmin) & (out["date"] <= dmax)]
    out = out[(out["hour"] >= 0) & (out["hour"] <= 23)]

    agg = (
        out.groupby(["date", "hour"], dropna=False)
        .agg(city_n_events=("n_events", "sum"))
        .reset_index()
    )

    agg["city_n_events"] = pd.to_numeric(agg["city_n_events"], errors="coerce").fillna(0).astype("float32")
    agg["city_has_event"] = (agg["city_n_events"] > 0).astype("Int8")
    return agg


def _resolve_location_static_base(base: Path) -> Path:
    base = Path(base).resolve()
    candidate = base / "df_location_static"
    if candidate.exists():
        return candidate
    return base


def _extract_year_from_path(path: Path) -> int | None:
    for part in path.parts:
        if part.startswith("year="):
            raw = part.split("=", 1)[1].strip()
            if raw.isdigit():
                return int(raw)
    return None


def _allowed_years(min_date: str, max_date: str) -> set[int]:
    dmin = max(pd.to_datetime(min_date), ALLOWED_MIN_DATE)
    dmax = min(pd.to_datetime(max_date), ALLOWED_MAX_DATE)
    if dmax < dmin:
        return set()
    return set(range(int(dmin.year), int(dmax.year) + 1))


def _load_static_zone_features(
    base: Path,
    min_date: str,
    max_date: str,
    value_cols: list[str],
    source_label: str,
) -> pd.DataFrame:
    base = _resolve_location_static_base(base)
    files = list_all_parquets(base)
    out_cols = ["pu_location_id"] + value_cols

    if not files:
        console.print(
            f"[yellow]Aviso:[/yellow] No se encontraron parquets de {source_label} en {base}"
        )
        return pd.DataFrame(columns=out_cols)

    years_ok = _allowed_years(min_date=min_date, max_date=max_date)
    dfs: list[pd.DataFrame] = []
    for fp in files:
        y = _extract_year_from_path(fp)
        if years_ok and y is not None and y not in years_ok:
            continue

        df = pd.read_parquet(fp).copy()
        if "pu_location_id" not in df.columns:
            continue
        for c in value_cols:
            if c not in df.columns:
                df[c] = pd.NA

        df = df[out_cols].copy()
        df["feature_year"] = y
        dfs.append(df)

    if not dfs:
        console.print(
            f"[yellow]Aviso:[/yellow] No hay datos de {source_label} dentro del rango {min_date}..{max_date}"
        )
        return pd.DataFrame(columns=out_cols)

    out = pd.concat(dfs, ignore_index=True)
    out["pu_location_id"] = pd.to_numeric(out["pu_location_id"], errors="coerce").astype("Int32")
    out = out.dropna(subset=["pu_location_id"]).copy()
    for c in value_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    if out["feature_year"].notna().any():
        by_year = (
            out.groupby(["pu_location_id", "feature_year"], as_index=False, dropna=False)[value_cols]
            .mean()
        )
        latest_year = by_year.groupby("pu_location_id", as_index=False)["feature_year"].max()
        out = by_year.merge(latest_year, on=["pu_location_id", "feature_year"], how="inner")
        out = out.groupby("pu_location_id", as_index=False)[value_cols].mean()
    else:
        out = out.groupby("pu_location_id", as_index=False)[value_cols].mean()

    out["pu_location_id"] = pd.to_numeric(out["pu_location_id"], errors="coerce").astype("Int32")
    for c in value_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    return out[out_cols]


def load_restaurants_zone_features(
    restaurants_base: Path,
    min_date: str,
    max_date: str,
) -> pd.DataFrame:
    value_cols = ["n_restaurants_zone", "n_cuisines_zone"]
    return _load_static_zone_features(
        base=restaurants_base,
        min_date=min_date,
        max_date=max_date,
        value_cols=value_cols,
        source_label="restaurants",
    )


def load_rent_zone_features(
    rent_base: Path,
    min_date: str,
    max_date: str,
) -> pd.DataFrame:
    value_cols = ["rent_price_zone"]
    return _load_static_zone_features(
        base=rent_base,
        min_date=min_date,
        max_date=max_date,
        value_cols=value_cols,
        source_label="rent",
    )


def _load_yearly_zone_features(
    base: Path,
    min_date: str,
    max_date: str,
    value_cols: list[str],
    source_label: str,
) -> pd.DataFrame:
    base = _resolve_location_static_base(base)
    files = list_all_parquets(base)
    out_cols = ["year", "pu_location_id"] + value_cols

    if not files:
        console.print(
            f"[yellow]Aviso:[/yellow] No se encontraron parquets de {source_label} en {base}"
        )
        return pd.DataFrame(columns=out_cols)

    years_ok = _allowed_years(min_date=min_date, max_date=max_date)
    dfs: list[pd.DataFrame] = []
    for fp in files:
        y = _extract_year_from_path(fp)
        if y is None:
            continue
        if years_ok and y not in years_ok:
            continue

        df = pd.read_parquet(fp).copy()
        if "pu_location_id" not in df.columns:
            continue
        for c in value_cols:
            if c not in df.columns:
                df[c] = pd.NA

        df = df[["pu_location_id"] + value_cols].copy()
        df["year"] = int(y)
        dfs.append(df)

    if not dfs:
        console.print(
            f"[yellow]Aviso:[/yellow] No hay datos de {source_label} dentro del rango {min_date}..{max_date}"
        )
        return pd.DataFrame(columns=out_cols)

    out = pd.concat(dfs, ignore_index=True)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int32")
    out["pu_location_id"] = pd.to_numeric(out["pu_location_id"], errors="coerce").astype("Int32")
    out = out.dropna(subset=["year", "pu_location_id"]).copy()
    for c in value_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = (
        out.groupby(["year", "pu_location_id"], as_index=False, dropna=False)[value_cols]
        .mean()
    )
    for c in value_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    return out[out_cols]


def load_restaurants_zone_features_yearly(
    restaurants_base: Path,
    min_date: str,
    max_date: str,
) -> pd.DataFrame:
    value_cols = ["n_restaurants_zone", "n_cuisines_zone"]
    return _load_yearly_zone_features(
        base=restaurants_base,
        min_date=min_date,
        max_date=max_date,
        value_cols=value_cols,
        source_label="restaurants",
    )


def _load_taxi_zone_lookup_for_imputation() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[4]
    fp = (project_root / "data/external/taxi_zone_lookup.csv").resolve()
    if not fp.exists():
        return pd.DataFrame(columns=["pu_location_id", "borough"])

    zones = pd.read_csv(fp)
    if "LocationID" not in zones.columns or "Borough" not in zones.columns:
        return pd.DataFrame(columns=["pu_location_id", "borough"])

    out = zones[["LocationID", "Borough"]].copy()
    out.columns = ["pu_location_id", "borough"]
    out["pu_location_id"] = pd.to_numeric(out["pu_location_id"], errors="coerce").astype("Int32")
    out["borough"] = out["borough"].astype("string").str.strip()
    out.loc[out["borough"].isin(["Unknown", "N/A", "NA", "nan"]), "borough"] = pd.NA
    out = out.dropna(subset=["pu_location_id"]).drop_duplicates(subset=["pu_location_id"])
    return out


def load_rent_zone_features_yearly(
    rent_base: Path,
    min_date: str,
    max_date: str,
) -> pd.DataFrame:
    value_cols = ["rent_price_zone"]
    out = _load_yearly_zone_features(
        base=rent_base,
        min_date=min_date,
        max_date=max_date,
        value_cols=value_cols,
        source_label="rent",
    )
    years_ok = _allowed_years(min_date=min_date, max_date=max_date)

    # Regla de negocio: para 2025 usar renta de 2024.
    if 2025 in years_ok:
        source_2024 = out[pd.to_numeric(out["year"], errors="coerce") == 2024].copy()

        # Si el rango pedido no incluye 2024, lo cargamos aparte para construir el fallback.
        if source_2024.empty:
            source_2024 = _load_yearly_zone_features(
                base=rent_base,
                min_date="2024-01-01",
                max_date="2024-12-31",
                value_cols=value_cols,
                source_label="rent",
            )
            source_2024 = source_2024[pd.to_numeric(source_2024["year"], errors="coerce") == 2024].copy()

        if not source_2024.empty:
            fallback_2025 = source_2024.copy()
            fallback_2025["year"] = 2025
            out = out[pd.to_numeric(out["year"], errors="coerce") != 2025].copy()
            out = pd.concat([out, fallback_2025], ignore_index=True)

    if years_ok:
        out = out[pd.to_numeric(out["year"], errors="coerce").isin(years_ok)].copy()

    if out.empty:
        return pd.DataFrame(columns=["year", "pu_location_id", "rent_price_zone"])

    # Expandimos a todas las taxi zones y aplicamos imputacion por borough -> anio -> global.
    zones = _load_taxi_zone_lookup_for_imputation()
    if not zones.empty and years_ok:
        years_sorted = sorted(years_ok)
        full_index = pd.MultiIndex.from_product(
            [years_sorted, zones["pu_location_id"].dropna().unique().tolist()],
            names=["year", "pu_location_id"],
        )
        full_df = full_index.to_frame(index=False)
        full_df["year"] = pd.to_numeric(full_df["year"], errors="coerce").astype("Int32")
        full_df["pu_location_id"] = pd.to_numeric(full_df["pu_location_id"], errors="coerce").astype("Int32")
        full_df = full_df.merge(zones, on="pu_location_id", how="left")
        out = full_df.merge(out, on=["year", "pu_location_id"], how="left")
    else:
        out["borough"] = pd.NA

    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int32")
    out["pu_location_id"] = pd.to_numeric(out["pu_location_id"], errors="coerce").astype("Int32")
    out["rent_price_zone"] = pd.to_numeric(out["rent_price_zone"], errors="coerce").astype("float32")

    if "borough" in out.columns:
        borough_mean = out.groupby(["year", "borough"], dropna=False)["rent_price_zone"].transform("mean")
        out["rent_price_zone"] = out["rent_price_zone"].fillna(borough_mean)

    year_mean = out.groupby("year", dropna=False)["rent_price_zone"].transform("mean")
    out["rent_price_zone"] = out["rent_price_zone"].fillna(year_mean)

    global_mean = pd.to_numeric(out["rent_price_zone"], errors="coerce").mean()
    if not pd.isna(global_mean):
        out["rent_price_zone"] = out["rent_price_zone"].fillna(global_mean)

    out = out.sort_values(["year", "pu_location_id"]).reset_index(drop=True)
    return out[["year", "pu_location_id", "rent_price_zone"]]
