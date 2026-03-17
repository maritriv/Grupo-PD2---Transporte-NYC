# src/procesamiento/capa1/rules_meteo.py
import pandas as pd
from typing import Tuple, Optional

METEO_EXPECTED_COLUMNS = [
    "date", "hour", "temp_c", "precip_mm", 
    "rain_mm", "snowfall_mm", "wind_kmh", "weather_code"
]

def _mode_or_na(x: pd.Series):
    """Devuelve la moda de la serie, o pd.NA si está vacía/nula"""
    m = x.mode()
    return m.iloc[0] if not m.empty else pd.NA

def _estandarizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    current_cols = {col.lower(): col for col in df.columns}
    rename_dict = {}
    missing_cols = []
    
    for expected in METEO_EXPECTED_COLUMNS:
        lower = expected.lower() 
        if lower in current_cols:
            rename_dict[current_cols[lower]] = expected
        else:
            missing_cols.append(expected)
            
    df = df.rename(columns=rename_dict)
    for col in missing_cols:
        df[col] = pd.NA
        
    return df[METEO_EXPECTED_COLUMNS]

def clean_meteo_batch(df: pd.DataFrame, date: Optional[tuple[int, int]] = None) -> pd.DataFrame:
    df = _estandarizar_columnas(df)

    # 1. PARSEO DE FECHAS Y HORAS
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
    df['hour'] = pd.to_numeric(df['hour'], errors='coerce').astype('Int64')
    
    # 2. CONVERSIONES NUMÉRICAS PREVIAS
    df['temp_c'] = pd.to_numeric(df['temp_c'], errors='coerce')
    df['wind_kmh'] = pd.to_numeric(df['wind_kmh'], errors='coerce')
    df['weather_code'] = pd.to_numeric(df['weather_code'], errors='coerce')

    precip_cols = ["precip_mm", "rain_mm", "snowfall_mm"]
    for c in precip_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    
    # 3. MÁSCARAS Y FILTROS FÍSICOS
    mask_valid_dt = df['date'].notna() & df['hour'].notna()
    
    mask_date = pd.Series(True, index=df.index)
    if date is not None:
        year, month = date
        mask_date = (df['date'].dt.year == year) & (df['date'].dt.month == month)
    
    mask_hour = (df['hour'] >= 0) & (df['hour'] < 24)
    
    # Filtrar valores meteorológicos imposibles (New York no baja de -40 ni sube de 55, ni tiene vientos > 300)
    mask_temp = df['temp_c'].isna() | ((df['temp_c'] >= -40) & (df['temp_c'] <= 55))
    mask_wind = df['wind_kmh'].isna() | ((df['wind_kmh'] >= 0) & (df['wind_kmh'] <= 300))
    mask_precip = (df['precip_mm'] >= 0) & (df['rain_mm'] >= 0) & (df['snowfall_mm'] >= 0)

    # Aplicamos filtros
    df_clean = df[mask_valid_dt & mask_hour & mask_date & mask_temp & mask_wind & mask_precip].copy()

    # 4. AGREGACIÓN (Para colapsar duplicados en la misma hora)
    if not df_clean.empty:
        agg_map = {
            "temp_c": "mean",
            "precip_mm": "sum",
            "rain_mm": "sum",
            "snowfall_mm": "sum",
            "wind_kmh": "mean",
            "weather_code": _mode_or_na,
        }
        
        df_clean = (
            df_clean.groupby(["date", "hour"], as_index=False)
            .agg(agg_map)
            .sort_values(["date", "hour"])
            .reset_index(drop=True)
        )

    # 5. TYPE MAPPING (Schema Match para PyArrow)
    type_mapping = {
        'hour': 'Int64',
        'temp_c': 'float64',
        'precip_mm': 'float64',
        'rain_mm': 'float64',
        'snowfall_mm': 'float64',
        'wind_kmh': 'float64',
        'weather_code': 'Int64' # Los códigos meteorológicos son enteros
    }

    for col, dtype in type_mapping.items():
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').astype(dtype)

    return df_clean