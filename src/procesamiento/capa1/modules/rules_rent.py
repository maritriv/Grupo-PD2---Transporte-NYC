# src/procesamiento/capa1/rules_rent.py
import pandas as pd
from typing import Optional, Tuple

FINAL_RENT_COLUMNS = [
    "id", "borough", "neighborhood", "latitude", "longitude", "room_type", "price"
]

KNOWN_BOROUGHS = {"Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"}

def _estandarizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Pasamos a minúsculas y quitamos espacios invisibles de las columnas originales
    current_cols = {str(col).lower().strip(): col for col in df.columns}
    
    # 2. Mapeamos de forma flexible (soporta múltiples versiones de Airbnb)
    rename_map = {
        "id": "id",
        "neighbourhood_group_cleansed": "borough",
        "neighbourhood_group": "borough",
        "neighbourhood_cleansed": "neighborhood",
        "neighbourhood": "neighborhood",
        "latitude": "latitude",
        "longitude": "longitude",
        "room_type": "room_type",
        "price": "price"
    }
    
    new_cols = {}
    for col_lower, original_name in current_cols.items():
        if col_lower in rename_map:
            new_cols[original_name] = rename_map[col_lower]
            
    df = df.rename(columns=new_cols)
    
    # Asegurar que existan todas las finales
    for col in FINAL_RENT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
            
    return df[FINAL_RENT_COLUMNS]

def clean_rent_batch(df: pd.DataFrame, date_file: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
    df = _estandarizar_columnas(df)

    # 1. PARSEO DE NÚMEROS
    df['id'] = pd.to_numeric(df['id'], errors='coerce')
    
    # Cambiamos comas por puntos por si Airbnb guardó las coordenadas con formato europeo
    df['latitude'] = df['latitude'].astype(str).str.replace(',', '.')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    
    df['longitude'] = df['longitude'].astype(str).str.replace(',', '.')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    if 'price' in df.columns:
        # Limpieza agresiva de dinero
        df['price'] = df['price'].astype(str).str.replace(r'[\$, ]', '', regex=True)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')

    # 2. LIMPIEZA DE STRINGS
    df['borough'] = df['borough'].astype(str).str.strip().str.title()
    df['neighborhood'] = df['neighborhood'].astype(str).str.strip().str.title()
    df['room_type'] = df['room_type'].astype(str).str.strip().str.title()

    # 3. MÁSCARAS DE VALIDACIÓN FÍSICA
    mask_price = df['price'].notna() & (df['price'] > 0) & (df['price'] <= 10000)
    mask_lat = df['latitude'].notna() & (df['latitude'] >= 40.4) & (df['latitude'] <= 41.0)
    mask_lon = df['longitude'].notna() & (df['longitude'] >= -74.3) & (df['longitude'] <= -73.6)
    mask_borough = df['borough'].isin(KNOWN_BOROUGHS)

    # === CHIVATO PARA DEBUG ===
    # Esto imprimirá en tu consola qué está pasando por debajo
    print(f"\n[DEBUG] Batch de {len(df)} filas")
    print(f"[DEBUG] Pasan precio: {mask_price.sum()}")
    print(f"[DEBUG] Pasan latitud: {mask_lat.sum()}")
    print(f"[DEBUG] Pasan longitud: {mask_lon.sum()}")
    print(f"[DEBUG] Pasan borough: {mask_borough.sum()}")
    if mask_borough.sum() == 0:
        print(f"[DEBUG] Boroughs encontrados en el archivo: {df['borough'].unique()[:10]}")
    # ==========================

    # Aplicar máscaras
    df_clean = df[mask_price & mask_lat & mask_lon & mask_borough].copy()

    # 4. MAPPING DE TIPOS PARA PYARROW
    df_clean['borough'] = df_clean['borough'].astype('category')
    df_clean['neighborhood'] = df_clean['neighborhood'].astype('category')
    df_clean['room_type'] = df_clean['room_type'].astype('category')

    type_mapping = {
        'id': 'Int64',
        'latitude': 'float64',
        'longitude': 'float64',
        'price': 'float64'
    }
    for col, dtype in type_mapping.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(dtype)

    return df_clean[FINAL_RENT_COLUMNS]