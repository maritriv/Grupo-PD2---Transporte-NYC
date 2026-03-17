# src/procesamiento/capa1/rules_fhvhv.py
import pandas as pd
from typing import Tuple, Optional
from .config_dicts import (
    FHVHV_EXPECTED_COLUMNS,
    FHVHV_KNOWN_LICENSES,
    FHVHV_YN_FLAGS,
    MAX_TRIP_DURATION_MIN,
    EXTREME_BASE_FARE,
    EXTREME_DRIVER_PAY,
)
from .valid_location_ids import valid_location_ids

def _estandarizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    current_cols = {col.lower(): col for col in df.columns}
    rename_dict = {}
    missing_cols = []
    for expected in FHVHV_EXPECTED_COLUMNS:
        lower = expected.lower() 
        if lower in current_cols:
            rename_dict[current_cols[lower]] = expected
        else:
            missing_cols.append(expected)
    df = df.rename(columns=rename_dict)
    for col in missing_cols:
        df[col] = pd.NA
    return df[FHVHV_EXPECTED_COLUMNS]

def clean_fhvhv_batch(df: pd.DataFrame, date: Optional[tuple[int, int]] = None) -> pd.DataFrame:
    df = _estandarizar_columnas(df)
    
    # Nulos
    df.loc[~df['hvfhs_license_num'].isin(FHVHV_KNOWN_LICENSES), 'hvfhs_license_num'] = pd.NA
    
    for flag_col in FHVHV_YN_FLAGS:
        if flag_col in df.columns:
            df.loc[~df[flag_col].isin({"Y", "N"}), flag_col] = pd.NA
            
    if valid_location_ids is not None:
        df.loc[~df['PULocationID'].isin(valid_location_ids), 'PULocationID'] = pd.NA
        df.loc[~df['DOLocationID'].isin(valid_location_ids), 'DOLocationID'] = pd.NA

    # Fechas
    for c in ["request_datetime", "on_scene_datetime", "pickup_datetime", "dropoff_datetime"]:
        df[c] = pd.to_datetime(df[c], errors='coerce')
    df = df.dropna(subset=['pickup_datetime', 'dropoff_datetime'])

    mask_date = pd.Series(True, index=df.index)
    if date is not None:
        year, month = date
        pickup = df['pickup_datetime']
        mask_date = (pickup.dt.year == year) & (pickup.dt.month == month)
    
    mask_time = (
        df['dropoff_datetime'].isna() | df['pickup_datetime'].isna() |
        (df['dropoff_datetime'] > df['pickup_datetime'])
    ) & (
        df['pickup_datetime'].isna() | df['request_datetime'].isna() |
        (df['pickup_datetime'] > df['request_datetime'])
    ) & (
        df['pickup_datetime'].isna() | df['on_scene_datetime'].isna() |
        (df['pickup_datetime'] >= df['on_scene_datetime'])
    ) & (
        df['on_scene_datetime'].isna() | df['request_datetime'].isna() |
        (df['on_scene_datetime'] >= df['request_datetime'])
    )
    
    durations_min = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 60.0
    # trip_time ya viene en segundos (es numérico), solo dividimos entre 60
    trip_time_min = df["trip_time"] / 60.0

    trip_time_diff_min = (trip_time_min - durations_min).abs()

    mask_max_duration = (
        durations_min.isna() |
        (
            (durations_min <= MAX_TRIP_DURATION_MIN) &
            (trip_time_diff_min < 5)  # 5 minutos de tolerancia
        )
    )

    # Distancia
    mask_distance = df['trip_miles'].isna() | (df['trip_miles'] > 0)

    # Dinero (ajustado a las columnas de FHVHV)
    mask_money = pd.Series(True, index=df.index)
    money_cols = [
        "base_passenger_fare",
        "tolls",
        "bcf",
        "sales_tax",
        "congestion_surcharge",
        "airport_fee",
        "tips",
        "driver_pay",
        "cbd_congestion_fee",
    ]
    for c in money_cols:
        mask_money &= df[c].isna() | (df[c] >= 0)
    
    mask_money &= df['base_passenger_fare'].isna() | df['base_passenger_fare'] <= EXTREME_BASE_FARE
    mask_money &= df['driver_pay'].isna() | df['driver_pay'] <= EXTREME_DRIVER_PAY

    df_clean = df[mask_date & mask_time & mask_max_duration & mask_money & mask_distance].copy()

    cols_to_category = list(FHVHV_YN_FLAGS) + ['hvfhs_license_num']
    for cat_col in cols_to_category:
        if cat_col in df_clean.columns:
            df_clean[cat_col] = df_clean[cat_col].astype('category')
    
    type_mapping = {
        'PULocationID': 'Int64',
        'DOLocationID': 'Int64',
        'trip_time': 'Int64',           # El diccionario dice que son segundos, así que entero
        'trip_miles': 'float64',
        'base_passenger_fare': 'float64',
        'tolls': 'float64',
        'bcf': 'float64',
        'sales_tax': 'float64',
        'congestion_surcharge': 'float64',
        'airport_fee': 'float64',
        'tips': 'float64',
        'driver_pay': 'float64',
        'cbd_congestion_fee': 'float64'
    }

    for col, dtype in type_mapping.items():
        if col in df_clean.columns:
            # Forzamos numérico y en caso de encontrar un valor raro, lo pone a nulo en vez de dar error
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').astype(dtype)

    return df_clean