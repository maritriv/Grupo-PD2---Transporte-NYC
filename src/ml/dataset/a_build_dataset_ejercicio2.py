from __future__ import annotations

import pandas as pd
from typing import Optional, Tuple

FINAL_RESTAURANT_COLUMNS = [
    "camis",
    "dba",
    "boro",
    "building",
    "street",
    "zipcode",
    "phone",
    "cuisine_description",
    "inspection_date",
    "action",
    "violation_code",
    "violation_description",
    "critical_flag",
    "score",
    "grade",
    "grade_date",
    "record_date",
    "inspection_type",
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
BOROUGH_MAP = {
    "MANHATTAN": "Manhattan",
    "BRONX": "Bronx",
    "BROOKLYN": "Brooklyn",
    "QUEENS": "Queens",
    "STATEN ISLAND": "Staten Island",
    "1": "Manhattan",
    "2": "Bronx",
    "3": "Brooklyn",
    "4": "Queens",
    "5": "Staten Island",
}


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


def _normalize_borough(series: pd.Series) -> pd.Series:
    clean = _clean_text_series(series)
    upper = clean.str.upper()
    mapped = upper.map(BOROUGH_MAP)
    return mapped.fillna(clean.str.title())


def _estandarizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["camis"] = _pick_first_series(df, ["camis", "CAMIS"])
    out["dba"] = _pick_first_series(df, ["dba", "DBA"])
    out["boro"] = _pick_first_series(df, ["boro", "borough", "Boro"])
    out["building"] = _pick_first_series(df, ["building"])
    out["street"] = _pick_first_series(df, ["street"])
    out["zipcode"] = _pick_first_series(df, ["zipcode", "zip"])
    out["phone"] = _pick_first_series(df, ["phone"])
    out["cuisine_description"] = _pick_first_series(df, ["cuisine_description"])
    out["inspection_date"] = _pick_first_series(df, ["inspection_date"])
    out["action"] = _pick_first_series(df, ["action"])
    out["violation_code"] = _pick_first_series(df, ["violation_code"])
    out["violation_description"] = _pick_first_series(df, ["violation_description"])
    out["critical_flag"] = _pick_first_series(df, ["critical_flag"])
    out["score"] = _pick_first_series(df, ["score"])
    out["grade"] = _pick_first_series(df, ["grade"])
    out["grade_date"] = _pick_first_series(df, ["grade_date"])
    out["record_date"] = _pick_first_series(df, ["record_date"])
    out["inspection_type"] = _pick_first_series(df, ["inspection_type"])
    out["latitude"] = _pick_first_series(df, ["latitude"])
    out["longitude"] = _pick_first_series(df, ["longitude"])
    out["community_board"] = _pick_first_series(df, ["community_board"])
    out["council_district"] = _pick_first_series(df, ["council_district"])
    out["census_tract"] = _pick_first_series(df, ["census_tract"])
    out["bin"] = _pick_first_series(df, ["bin"])
    out["bbl"] = _pick_first_series(df, ["bbl"])
    out["nta"] = _pick_first_series(df, ["nta"])
    return out[FINAL_RESTAURANT_COLUMNS]


def clean_restaurants_batch(df: pd.DataFrame, date_file: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
    del date_file

    df = _estandarizar_columnas(df)

    df["camis"] = pd.to_numeric(df["camis"], errors="coerce").astype("Int64")
    df["score"] = pd.to_numeric(df["score"], errors="coerce").astype("Int64")
    df["latitude"] = _coerce_numeric(df["latitude"])
    df["longitude"] = _coerce_numeric(df["longitude"])

    for col in ["inspection_date", "grade_date", "record_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in [
        "dba", "building", "street", "zipcode", "phone", "cuisine_description",
        "action", "violation_code", "violation_description", "critical_flag",
        "grade", "inspection_type", "community_board", "council_district",
        "census_tract", "bin", "bbl", "nta",
    ]:
        df[col] = _clean_text_series(df[col])

    df["boro"] = _normalize_borough(df["boro"])
    df["dba"] = df["dba"].str.title()
    df["street"] = df["street"].str.title()
    df["cuisine_description"] = df["cuisine_description"].str.title()
    df["critical_flag"] = df["critical_flag"].str.title()
    df["grade"] = df["grade"].str.upper()
    df["inspection_type"] = df["inspection_type"].str.title()

    mask_camis = df["camis"].notna()
    mask_date = df["inspection_date"].notna()
    mask_boro = df["boro"].isin(KNOWN_BOROUGHS)
    mask_lat = df["latitude"].isna() | ((df["latitude"] >= 40.4) & (df["latitude"] <= 41.0))
    mask_lon = df["longitude"].isna() | ((df["longitude"] >= -74.3) & (df["longitude"] <= -73.6))
    mask_score = df["score"].isna() | ((df["score"] >= 0) & (df["score"] <= 100))

    df_clean = df[mask_camis & mask_date & mask_boro & mask_lat & mask_lon & mask_score].copy()
    df_clean = df_clean.drop_duplicates()

    return df_clean[FINAL_RESTAURANT_COLUMNS]
