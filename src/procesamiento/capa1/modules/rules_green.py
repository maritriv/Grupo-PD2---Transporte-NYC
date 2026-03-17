# src/procesamiento/capa1/rules_green.py
import pandas as pd
from typing import Tuple, Optional
from .config_dicts import (
    GREEN_EXPECTED_COLUMNS,
    GREEN_ALLOWED_VENDOR_ID, 
    GREEN_ALLOWED_RATECODE_ID,
    GREEN_ALLOWED_STORE_FLAG,
    GREEN_ALLOWED_PAYMENT_TYPE,
    GREEN_ALLOWED_TRIP_TYPE,
    MAX_TRIP_DURATION_MIN,
    EXTREME_TOTAL_AMOUNT,
    EXTREME_FARE_AMOUNT,
)
from .valid_location_ids import valid_location_ids

def _estandarizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    current_cols = {col.lower(): col for col in df.columns}
    rename_dict = {}
    missing_cols = []
    for expected in GREEN_EXPECTED_COLUMNS:
        lower = expected.lower() 
        if lower in current_cols:
            rename_dict[current_cols[lower]] = expected
        else:
            missing_cols.append(expected)
    df = df.rename(columns=rename_dict)
    for col in missing_cols:
        df[col] = pd.NA
    return df[GREEN_EXPECTED_COLUMNS]

def clean_green_batch(df: pd.DataFrame, date: Optional[tuple[int, int]] = None) -> pd.DataFrame:
    df = _estandarizar_columnas(df)
    
    # Nulos
    df.loc[~df['RatecodeID'].isin(GREEN_ALLOWED_RATECODE_ID), 'RatecodeID'] = pd.NA 
    df.loc[~df['payment_type'].isin(GREEN_ALLOWED_PAYMENT_TYPE), 'payment_type'] = pd.NA 
    df.loc[~df['VendorID'].isin(GREEN_ALLOWED_VENDOR_ID), 'VendorID'] = pd.NA
    df.loc[~df['store_and_fwd_flag'].isin(GREEN_ALLOWED_STORE_FLAG), 'store_and_fwd_flag'] = pd.NA
    df.loc[~df['trip_type'].isin(GREEN_ALLOWED_TRIP_TYPE), 'trip_type'] = pd.NA
        
    if valid_location_ids is not None:
        df.loc[~df['PULocationID'].isin(valid_location_ids), 'PULocationID'] = pd.NA
        df.loc[~df['DOLocationID'].isin(valid_location_ids), 'DOLocationID'] = pd.NA

    # Fechas
    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'], errors='coerce')
    df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'], errors='coerce')
    df = df.dropna(subset=['lpep_pickup_datetime', 'lpep_dropoff_datetime'])

    mask_date = pd.Series(True, index=df.index)
    if date is not None:
        year, month = date
        pickup = df['lpep_pickup_datetime']
        mask_date = (pickup.dt.year == year) & (pickup.dt.month == month)
    
    mask_time = (df['lpep_dropoff_datetime'] > df['lpep_pickup_datetime'])
    durations_min = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60.0
    mask_max_duration = (durations_min <= MAX_TRIP_DURATION_MIN)

    mask_passenger = df['passenger_count'].isna() | (df['passenger_count'] > 0)
    mask_distance = df['trip_distance'].isna() | (df['trip_distance'] > 0)

    mask_money = (
        (df["mta_tax"].isna() | (df["mta_tax"] >= 0)) & 
        (df["improvement_surcharge"].isna() | (df["improvement_surcharge"] >= 0)) & 
        (df["congestion_surcharge"].isna() | (df["congestion_surcharge"] >= 0)) & 
        (df["cbd_congestion_fee"].isna() | (df["cbd_congestion_fee"] >= 0)) &
        (df["fare_amount"].isna() | (df['fare_amount'] >= 0)) &
        (df["fare_amount"].isna() | (df["fare_amount"] <= EXTREME_FARE_AMOUNT)) &
        (df["total_amount"].isna() | (df['total_amount'] >= 0)) &
        (df["total_amount"].isna() | (df["total_amount"] <= EXTREME_TOTAL_AMOUNT))
    )

    df_clean = df[mask_date & mask_time & mask_max_duration & mask_money & mask_passenger & mask_distance].copy()

    type_mapping = {
        'VendorID': 'Int64',
        'passenger_count': 'Int64',
        'RatecodeID': 'Int64',
        'PULocationID': 'Int64',
        'DOLocationID': 'Int64',
        'payment_type': 'Int64',
        'trip_distance': 'float64',
        'fare_amount': 'float64',
        'store_and_fwd_flag': 'category',
    }
    for col, dtype in type_mapping.items():
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').astype(dtype)

    return df_clean
