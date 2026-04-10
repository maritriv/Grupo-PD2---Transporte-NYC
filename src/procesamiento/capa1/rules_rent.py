import pandas as pd
from typing import Any, Optional, Tuple


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
    "price",
    "price_moe",
]

KNOWN_BOROUGHS = {"Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"}
NYC_CENSUS_ZONE_PATTERN = r"^36(?:005|047|061|081|085)\d{6}$"
MAX_REASONABLE_PRICE = 10_000.0
PRICE_OUTLIER_IQR_MULT = 3.0
MIN_SNAPSHOT_YEAR = 2023
MAX_SNAPSHOT_YEAR = 2024


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
    out["price"] = _pick_first_series(df, ["price"])
    out["price_moe"] = _pick_first_series(df, ["price_moe"])
    return out[FINAL_RENT_COLUMNS]


def _normalize_snapshot_date(
    series: pd.Series,
    date_file: Optional[Tuple[int, int]] = None,
) -> pd.Series:
    """
    Normaliza snapshots a 'YYYY-MM-DD'.
    Soporta:
      - YYYY-acs5 -> YYYY-01-01
      - YYYY -> YYYY-01-01
      - fechas ISO
    Si date_file viene del nombre del parquet (YYYY-MM), fuerza coherencia con ese mes.
    """
    s = _clean_text_series(series).astype("string")
    s = s.str.replace(r"(?i)^(\d{4})-acs5$", r"\1-01-01", regex=True)
    s = s.str.replace(r"^(\d{4})$", r"\1-01-01", regex=True)

    dt = pd.to_datetime(s, errors="coerce")
    if date_file is not None:
        year, month = date_file
        dt = dt.where((dt.dt.year == year) & (dt.dt.month == month))

    return dt.dt.strftime("%Y-%m-%d").astype("string")


def _price_outlier_mask(series: pd.Series) -> pd.Series:
    mask = pd.Series(True, index=series.index)
    valid = series.dropna()
    if len(valid) < 10:
        return mask

    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr <= 0:
        return mask

    low = max(0.0, q1 - PRICE_OUTLIER_IQR_MULT * iqr)
    high = q3 + PRICE_OUTLIER_IQR_MULT * iqr
    return series.between(low, high, inclusive="both")


def _build_comparable_view(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    out["zone_id"] = _coerce_numeric(df["zone_id"]).astype("Int64")
    out["zone_name"] = _clean_text_series(df["zone_name"]).str.title()
    out["source_snapshot_date"] = _normalize_snapshot_date(df["source_snapshot_date"])
    out["borough"] = _clean_text_series(df["borough"]).str.title()
    out["neighborhood"] = _clean_text_series(df["neighborhood"]).str.title()
    out["latitude"] = _coerce_numeric(df["latitude"])
    out["longitude"] = _coerce_numeric(df["longitude"])
    out["room_type"] = _clean_text_series(df["room_type"]).str.title()
    out["property_type"] = _clean_text_series(df["property_type"]).str.title()
    out["price"] = _coerce_numeric(df["price"])
    out["price_moe"] = pd.to_numeric(df["price_moe"], errors="coerce")
    return out[FINAL_RENT_COLUMNS]


def _count_changed_rows(df_before: pd.DataFrame, df_after: pd.DataFrame) -> int:
    if df_after.empty:
        return 0
    before = _build_comparable_view(df_before).reset_index(drop=True)
    after = _build_comparable_view(df_after).reset_index(drop=True)

    equal_mask = before.eq(after) | (before.isna() & after.isna())
    changed_mask = ~equal_mask.all(axis=1)
    return int(changed_mask.sum())


def clean_rent_batch(
    df: pd.DataFrame,
    date_file: Optional[Tuple[int, int]] = None,
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    df = _estandarizar_columnas(df)
    df["_source_row_id"] = range(len(df))

    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df["zone_id"] = _coerce_numeric(df["zone_id"]).astype("Int64")
    df["latitude"] = _coerce_numeric(df["latitude"])
    df["longitude"] = _coerce_numeric(df["longitude"])
    df["price"] = _coerce_numeric(df["price"])
    df["price_moe"] = pd.to_numeric(df["price_moe"], errors="coerce")
    df.loc[df["price_moe"] < 0, "price_moe"] = pd.NA

    # En ACS `id` y `zone_id` deberían representar la misma zona censal.
    df["zone_id"] = df["zone_id"].fillna(df["id"])
    df["id"] = df["id"].fillna(df["zone_id"])

    df["source_snapshot_date"] = _normalize_snapshot_date(df["source_snapshot_date"], date_file=date_file)
    df["zone_name"] = _clean_text_series(df["zone_name"]).str.title()
    df["borough"] = _clean_text_series(df["borough"]).str.title()
    df["neighborhood"] = _clean_text_series(df["neighborhood"]).str.title()
    df["room_type"] = _clean_text_series(df["room_type"]).str.title()
    df["property_type"] = _clean_text_series(df["property_type"]).str.title()

    # Completa nombres y fuerza consistencia semántica entre zone_name y neighborhood.
    df["zone_name"] = df["zone_name"].fillna(df["neighborhood"])
    df["neighborhood"] = df["neighborhood"].fillna(df["zone_name"])
    mismatch_names = (
        df["zone_name"].notna()
        & df["neighborhood"].notna()
        & (df["zone_name"].str.casefold() != df["neighborhood"].str.casefold())
    )
    df.loc[mismatch_names, "neighborhood"] = df.loc[mismatch_names, "zone_name"]

    zone_id_txt = df["zone_id"].astype("string")
    mask_zone_valid = zone_id_txt.str.fullmatch(NYC_CENSUS_ZONE_PATTERN, na=False)
    mask_id_matches_zone = (df["id"] == df["zone_id"]) | df["id"].isna() | df["zone_id"].isna()

    mask_snapshot = df["source_snapshot_date"].notna()
    snapshot_year = pd.to_datetime(df["source_snapshot_date"], errors="coerce").dt.year
    mask_snapshot_year = snapshot_year.between(MIN_SNAPSHOT_YEAR, MAX_SNAPSHOT_YEAR, inclusive="both")
    mask_zone_name = df["zone_name"].notna()
    mask_neighborhood = df["neighborhood"].notna()
    mask_price = df["price"].notna() & (df["price"] > 0) & (df["price"] <= MAX_REASONABLE_PRICE)
    mask_price_no_outlier = _price_outlier_mask(df["price"])
    mask_lat = df["latitude"].notna() & (df["latitude"] >= 40.4) & (df["latitude"] <= 41.0)
    mask_lon = df["longitude"].notna() & (df["longitude"] >= -74.3) & (df["longitude"] <= -73.6)
    mask_borough = df["borough"].isin(KNOWN_BOROUGHS)
    mask_room_type = df["room_type"].notna()
    mask_id = df["id"].notna()

    mask_all = (
        mask_snapshot
        & mask_snapshot_year
        & mask_id
        & mask_zone_valid
        & mask_id_matches_zone
        & mask_zone_name
        & mask_neighborhood
        & mask_price
        & mask_price_no_outlier
        & mask_lat
        & mask_lon
        & mask_borough
        & mask_room_type
    )

    removed_reasons: dict[str, int] = {}
    removed_by_rules = ~mask_all
    assigned = pd.Series(False, index=df.index)
    reason_masks: list[tuple[str, pd.Series]] = [
        ("snapshot_missing", ~mask_snapshot),
        ("snapshot_year_out_of_range", mask_snapshot & ~mask_snapshot_year),
        ("id_missing", mask_snapshot & mask_snapshot_year & ~mask_id),
        ("zone_id_invalid", mask_id & ~mask_zone_valid),
        ("id_zone_mismatch", mask_id & mask_zone_valid & ~mask_id_matches_zone),
        ("zone_name_missing", mask_id & mask_zone_valid & mask_id_matches_zone & ~mask_zone_name),
        ("neighborhood_missing", mask_id & mask_zone_valid & mask_id_matches_zone & mask_zone_name & ~mask_neighborhood),
        ("price_invalid_or_unreasonable", mask_neighborhood & ~mask_price),
        ("price_outlier", mask_price & ~mask_price_no_outlier),
        ("latitude_out_of_bounds_or_missing", mask_price & mask_price_no_outlier & ~mask_lat),
        ("longitude_out_of_bounds_or_missing", mask_lat & ~mask_lon),
        ("borough_invalid", mask_lon & ~mask_borough),
        ("room_type_missing", mask_borough & ~mask_room_type),
    ]
    for reason, reason_mask in reason_masks:
        fail_mask = removed_by_rules & (~assigned) & reason_mask
        n_fail = int(fail_mask.sum())
        if n_fail > 0:
            removed_reasons[reason] = n_fail
            assigned = assigned | fail_mask
    other_rule_fail = int((removed_by_rules & ~assigned).sum())
    if other_rule_fail > 0:
        removed_reasons["other_rule_failure"] = other_rule_fail

    # Dedupe con trazabilidad de motivo.
    df_pre = df.loc[mask_all].copy()
    df_pre_sorted = df_pre.sort_values(
        ["source_snapshot_date", "zone_id", "price_moe", "price"],
        ascending=[True, True, True, False],
    ).copy()
    dup_zone = df_pre_sorted.duplicated(subset=["source_snapshot_date", "zone_id"], keep="first")
    dup_id = (~dup_zone) & df_pre_sorted.duplicated(subset=["source_snapshot_date", "id"], keep="first")
    duplicate_zone_rows = int(dup_zone.sum())
    duplicate_id_rows = int(dup_id.sum())
    if duplicate_zone_rows > 0:
        removed_reasons["duplicate_zone_in_snapshot"] = duplicate_zone_rows
    if duplicate_id_rows > 0:
        removed_reasons["duplicate_id_in_snapshot"] = duplicate_id_rows
    df_clean = (
        df_pre_sorted.loc[~dup_zone & ~dup_id]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    before_survivors = (
        df.set_index("_source_row_id")
        .loc[df_clean["_source_row_id"], FINAL_RENT_COLUMNS]
        .reset_index(drop=True)
    )
    after_survivors = df_clean[FINAL_RENT_COLUMNS].reset_index(drop=True)
    changed_rows = _count_changed_rows(before_survivors, after_survivors)
    removed_rows = int(len(df) - len(df_clean))

    for cat_col in ["source_snapshot_date", "borough", "neighborhood", "room_type", "property_type"]:
        if cat_col in df_clean.columns:
            df_clean[cat_col] = df_clean[cat_col].astype("category")

    type_mapping = {
        "id": "Int64",
        "zone_id": "Int64",
        "latitude": "float64",
        "longitude": "float64",
        "price": "float64",
        "price_moe": "float64",
    }
    for col, dtype in type_mapping.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(dtype)

    return (
        df_clean[FINAL_RENT_COLUMNS],
        {
            "changed_rows": changed_rows,
            "removed_rows": removed_rows,
            "removed_reasons": removed_reasons,
        },
    )
