import pandas as pd
from typing import Optional, Tuple


FINAL_RENT_COLUMNS = [
    "id",
    "zone_id",
    "zone_name",
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
]

KNOWN_BOROUGHS = {"Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"}


def _pick_first_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series(pd.NA, index=df.index)


def _clean_text_series(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "<NA>": pd.NA, "None": pd.NA})
    )


def _coerce_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.replace(r"[^0-9,\.\-]", "", regex=True)
        .str.replace(",", ".", regex=False)
        .replace({"": pd.NA, "nan": pd.NA, "<NA>": pd.NA, "None": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _estandarizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["id"] = _pick_first_series(df, ["id"])
    out["zone_id"] = _pick_first_series(df, ["zone_id", "id"])
    out["zone_name"] = _pick_first_series(df, ["zone_name", "neighborhood", "neighbourhood_cleansed", "neighbourhood"])
    out["source_snapshot_date"] = _pick_first_series(df, ["source_snapshot_date", "last_scraped"])
    out["borough"] = _pick_first_series(df, ["borough", "neighbourhood_group_cleansed", "neighbourhood_group"])
    out["neighborhood"] = _pick_first_series(df, ["neighborhood", "neighbourhood_cleansed", "neighbourhood"])
    out["latitude"] = _pick_first_series(df, ["latitude"])
    out["longitude"] = _pick_first_series(df, ["longitude"])
    out["room_type"] = _pick_first_series(df, ["room_type"])
    out["property_type"] = _pick_first_series(df, ["property_type"])
    out["accommodates"] = _pick_first_series(df, ["accommodates"])
    out["minimum_nights"] = _pick_first_series(df, ["minimum_nights"])
    out["availability_365"] = _pick_first_series(df, ["availability_365"])
    out["price"] = _pick_first_series(df, ["price"])
    out["price_moe"] = _pick_first_series(df, ["price_moe"])
    return out[FINAL_RENT_COLUMNS]


def clean_rent_batch(df: pd.DataFrame, date_file: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
    del date_file

    df = _estandarizar_columnas(df)

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df["zone_id"] = _clean_text_series(df["zone_id"])
    df["latitude"] = _coerce_numeric(df["latitude"])
    df["longitude"] = _coerce_numeric(df["longitude"])
    df["price"] = _coerce_numeric(df["price"])
    df["price_moe"] = pd.to_numeric(df["price_moe"], errors="coerce")
    df["accommodates"] = pd.to_numeric(df["accommodates"], errors="coerce")
    df["minimum_nights"] = pd.to_numeric(df["minimum_nights"], errors="coerce")
    df["availability_365"] = pd.to_numeric(df["availability_365"], errors="coerce")

    df["source_snapshot_date"] = _clean_text_series(df["source_snapshot_date"])
    df["zone_name"] = _clean_text_series(df["zone_name"]).str.title()
    df["borough"] = _clean_text_series(df["borough"]).str.title()
    df["neighborhood"] = _clean_text_series(df["neighborhood"]).str.title()
    df["room_type"] = _clean_text_series(df["room_type"]).str.title()
    df["property_type"] = _clean_text_series(df["property_type"]).str.title()

    mask_price = df["price"].notna() & (df["price"] > 0) & (df["price"] <= 10_000)
    mask_lat = df["latitude"].notna() & (df["latitude"] >= 40.4) & (df["latitude"] <= 41.0)
    mask_lon = df["longitude"].notna() & (df["longitude"] >= -74.3) & (df["longitude"] <= -73.6)
    mask_borough = df["borough"].isin(KNOWN_BOROUGHS)
    mask_room_type = df["room_type"].notna()
    mask_id = df["id"].notna()

    if "minimum_nights" in df.columns:
        mask_min_nights = df["minimum_nights"].isna() | (df["minimum_nights"] > 0)
    else:
        mask_min_nights = pd.Series(True, index=df.index)

    if "availability_365" in df.columns:
        mask_availability = df["availability_365"].isna() | (
            (df["availability_365"] >= 0) & (df["availability_365"] <= 365)
        )
    else:
        mask_availability = pd.Series(True, index=df.index)

    if "accommodates" in df.columns:
        mask_accommodates = df["accommodates"].isna() | (df["accommodates"] > 0)
    else:
        mask_accommodates = pd.Series(True, index=df.index)

    df_clean = df[
        mask_id
        & mask_price
        & mask_lat
        & mask_lon
        & mask_borough
        & mask_room_type
        & mask_min_nights
        & mask_availability
        & mask_accommodates
    ].copy()

    for cat_col in ["source_snapshot_date", "borough", "neighborhood", "room_type", "property_type"]:
        if cat_col in df_clean.columns:
            df_clean[cat_col] = df_clean[cat_col].astype("category")

    type_mapping = {
        "id": "Int64",
        "latitude": "float64",
        "longitude": "float64",
        "accommodates": "Int64",
        "minimum_nights": "Int64",
        "availability_365": "Int64",
        "price": "float64",
        "price_moe": "float64",
    }
    for col, dtype in type_mapping.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(dtype)

    return df_clean[FINAL_RENT_COLUMNS]
