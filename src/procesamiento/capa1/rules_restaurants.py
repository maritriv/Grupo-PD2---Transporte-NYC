
import pandas as pd
from typing import Optional, Tuple


FINAL_RESTAURANTS_COLUMNS = [
    "camis",
    "restaurant_name",
    "borough",
    "building",
    "street",
    "zipcode",
    "cuisine_description",
    "inspection_date",
    "inspection_type",
    "critical_flag",
    "score",
    "grade",
    "grade_date",
    "record_date",
    "latitude",
    "longitude",
    "community_board",
    "council_district",
    "census_tract",
    "bin",
    "bbl",
    "nta",
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
    out["camis"] = _pick_first_series(df, ["camis", "id"])
    out["restaurant_name"] = _pick_first_series(df, ["restaurant_name", "dba", "name"])
    out["borough"] = _pick_first_series(df, ["borough", "boro"])
    out["building"] = _pick_first_series(df, ["building"])
    out["street"] = _pick_first_series(df, ["street"])
    out["zipcode"] = _pick_first_series(df, ["zipcode", "zip"])
    out["cuisine_description"] = _pick_first_series(df, ["cuisine_description", "cuisine"])
    out["inspection_date"] = _pick_first_series(df, ["inspection_date", "date"])
    out["inspection_type"] = _pick_first_series(df, ["inspection_type"])
    out["critical_flag"] = _pick_first_series(df, ["critical_flag"])
    out["score"] = _pick_first_series(df, ["score"])
    out["grade"] = _pick_first_series(df, ["grade"])
    out["grade_date"] = _pick_first_series(df, ["grade_date"])
    out["record_date"] = _pick_first_series(df, ["record_date"])
    out["latitude"] = _pick_first_series(df, ["latitude"])
    out["longitude"] = _pick_first_series(df, ["longitude"])
    out["community_board"] = _pick_first_series(df, ["community_board"])
    out["council_district"] = _pick_first_series(df, ["council_district"])
    out["census_tract"] = _pick_first_series(df, ["census_tract"])
    out["bin"] = _pick_first_series(df, ["bin"])
    out["bbl"] = _pick_first_series(df, ["bbl"])
    out["nta"] = _pick_first_series(df, ["nta"])
    return out[FINAL_RESTAURANTS_COLUMNS]


def clean_restaurants_batch(df: pd.DataFrame, date_file: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
    del date_file

    df = _estandarizar_columnas(df)

    df["camis"] = pd.to_numeric(df["camis"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["latitude"] = _coerce_numeric(df["latitude"])
    df["longitude"] = _coerce_numeric(df["longitude"])

    for c in [
        "restaurant_name",
        "borough",
        "building",
        "street",
        "zipcode",
        "cuisine_description",
        "inspection_type",
        "critical_flag",
        "grade",
        "community_board",
        "council_district",
        "census_tract",
        "bin",
        "bbl",
        "nta",
    ]:
        df[c] = _clean_text_series(df[c])

    df["restaurant_name"] = df["restaurant_name"].str.title()
    df["borough"] = df["borough"].str.title()
    df["street"] = df["street"].str.title()
    df["cuisine_description"] = df["cuisine_description"].str.title()
    df["critical_flag"] = df["critical_flag"].str.upper()
    df["grade"] = df["grade"].str.upper()

    for c in ["inspection_date", "grade_date", "record_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    mask_id = df["camis"].notna()
    mask_date = df["inspection_date"].notna()
    mask_borough = df["borough"].isin(KNOWN_BOROUGHS)
    mask_lat = df["latitude"].isna() | ((df["latitude"] >= 40.4) & (df["latitude"] <= 41.0))
    mask_lon = df["longitude"].isna() | ((df["longitude"] >= -74.3) & (df["longitude"] <= -73.6))
    mask_score = df["score"].isna() | ((df["score"] >= 0) & (df["score"] <= 200))

    df_clean = df[mask_id & mask_date & mask_borough & mask_lat & mask_lon & mask_score].copy()
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    return df_clean[FINAL_RESTAURANTS_COLUMNS]
