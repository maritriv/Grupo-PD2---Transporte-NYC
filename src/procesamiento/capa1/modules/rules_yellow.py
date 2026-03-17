# src/procesamiento/capa1/rules_yellow.py
import pandas as pd
from typing import Tuple, Optional
from .config_dicts import (
    YELLOW_EXPECTED_COLUMNS,
    YELLOW_ALLOWED_VENDOR_ID, 
    YELLOW_ALLOWED_RATECODE_ID, 
    YELLOW_ALLOWED_STORE_FLAG,
    YELLOW_ALLOWED_PAYMENT_TYPE,
    MAX_TRIP_DURATION_MIN,
    EXTREME_TOTAL_AMOUNT,
    EXTREME_FARE_AMOUNT,
)

from .valid_location_ids import valid_location_ids

def _estandarizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    current_cols = {col.lower(): col for col in df.columns}
    rename_dict = {}
    missing_cols = []

    for expected in YELLOW_EXPECTED_COLUMNS:
        lower = expected.lower() 
        if lower in current_cols:
            rename_dict[current_cols[lower]] = expected
        else:
            missing_cols.append(expected)

    df = df.rename(columns=rename_dict)

    for col in missing_cols:
        df[col] = pd.NA

    return df[YELLOW_EXPECTED_COLUMNS]

def clean_yellow_batch(df: pd.DataFrame, date: Optional[tuple[int, int]]) -> pd.DataFrame:
    df = _estandarizar_columnas(df)
    
    # 1. CODIFICACIÓN DE NULOS / DATA DICTIONARY
    df.loc[~df['RatecodeID'].isin(YELLOW_ALLOWED_RATECODE_ID), 'RatecodeID'] = pd.NA # 99 tratado como nulo
        
    df.loc[~df['payment_type'].isin(YELLOW_ALLOWED_PAYMENT_TYPE), 'payment_type'] = pd.NA # 5 y 6 (Unknown/Voided) los trataremos como nulos
        
    df.loc[~df['VendorID'].isin(YELLOW_ALLOWED_VENDOR_ID), 'VendorID'] = pd.NA
    
    df.loc[~df['store_and_fwd_flag'].isin(YELLOW_ALLOWED_STORE_FLAG), 'store_and_fwd_flag'] = pd.NA
    
    if valid_location_ids is not None:
        df.loc[~df['PULocationID'].isin(valid_location_ids), 'PULocationID'] = pd.NA
        df.loc[~df['DOLocationID'].isin(valid_location_ids), 'DOLocationID'] = pd.NA

    # 2. ELIMINACIÓN DE DATOS SIN SENTIDO (Filtros físicos y financieros)
    # Las máscaras deben ser tolerantes a nulos
    # Fechas válidas
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
    df = df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

    mask_date = pd.Series(True, index=df.index)
    if date is not None:
        year, month = date
        pickup = df['tpep_pickup_datetime']
        mask_date = (pickup.dt.year == year) & (pickup.dt.month == month)
    
    # Viajes en el tiempo o con duración 0
    mask_time = df['tpep_dropoff_datetime'] > df['tpep_pickup_datetime']
    
    # Filtro de duración máxima razonable
    durations_min = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60.0
    mask_max_duration = durations_min <= MAX_TRIP_DURATION_MIN

    # El número de pasajeros debe de ser mayor a 0
    mask_passenger = df['passenger_count'].isna() | (df['passenger_count'] > 0)

    # La distancia del viaje debe ser mayor a 0
    mask_distance = df['trip_distance'].isna() | (df['trip_distance'] > 0)

    # Tasas/surcharges
    # Valores monetarios deben ser >= 0 (evitar reembolsos/errores)
    mask_money = (
        (df["mta_tax"].isna() | (df["mta_tax"] >= 0)) &
        (df["improvement_surcharge"].isna() | (df["improvement_surcharge"] >= 0)) & 
        (df["congestion_surcharge"].isna() | (df["congestion_surcharge"] >= 0)) & 
        (df["airport_fee"].isna() | (df["airport_fee"] >= 0)) &
        (df["cbd_congestion_fee"].isna() | (df["cbd_congestion_fee"] >= 0)) &
        (df["fare_amount"].isna() | (df['fare_amount'] >= 0)) &
        (df["fare_amount"].isna() | (df["fare_amount"] <= EXTREME_FARE_AMOUNT)) &
        (df["total_amount"].isna() | (df['total_amount'] >= 0)) &
        (df["total_amount"].isna() | (df["total_amount"] <= EXTREME_TOTAL_AMOUNT))
    )

    # Aplicamos todos los filtros
    df_clean = df[mask_date & mask_time & mask_max_duration & mask_money & mask_passenger & mask_distance].copy()

    # Forzar tipos consistentes con soporte de nulos
    # Usamos "Int64" (con I mayúscula) o "float64"
    # El tipo "Int64" de pandas permite nulos y mantiene el número como entero
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
