# src/procesamiento/capa1/rules_eventos.py
import pandas as pd
from typing import Tuple, Optional

EVENTOS_EXPECTED_COLUMNS = ["date", "hour", "borough", "event_type", "n_events"]
KNOWN_BOROUGHS = {"Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island", "Ewr"}

def _estandarizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    current_cols = {col.lower(): col for col in df.columns}
    rename_dict = {}
    missing_cols = []
    
    for expected in EVENTOS_EXPECTED_COLUMNS:
        lower = expected.lower() 
        if lower in current_cols:
            rename_dict[current_cols[lower]] = expected
        else:
            missing_cols.append(expected)
            
    df = df.rename(columns=rename_dict)
    for col in missing_cols:
        df[col] = pd.NA
        
    return df[EVENTOS_EXPECTED_COLUMNS]

def clean_eventos_batch(df: pd.DataFrame, date: Optional[tuple[int, int]] = None) -> pd.DataFrame:
    df = _estandarizar_columnas(df)

    # 1. PARSEO DE FECHAS
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
    
    # 2. PARSEO ESTRICTO DE ENTEROS
    hour_num = pd.to_numeric(df['hour'], errors='coerce')
    df['hour'] = hour_num.where((hour_num % 1) == 0).astype('Int64')

    events_num = pd.to_numeric(df['n_events'], errors='coerce')
    df['n_events'] = events_num.where((events_num % 1) == 0).fillna(1).astype('Int64')

    # 3. MÁSCARAS
    mask_date = df['date'].notna()
    if date is not None:
        year, month = date
        mask_date &= (df['date'].dt.year == year) & (df['date'].dt.month == month)

    # Horas válidas [0, 23]
    mask_hour = df['hour'].notna() & (df['hour'] >= 0) & (df['hour'] <= 23)
    
    # Eventos positivos
    mask_events = df['n_events'] >= 0
    
    # Limpieza y filtrado de Strings (Borough y Tipo de Evento)
    df['borough'] = df['borough'].astype(str).str.strip().str.title()
    mask_borough = df['borough'].isin(KNOWN_BOROUGHS)
    
    # Event_type no debe estar vacío
    df['event_type'] = df['event_type'].astype(str).str.strip()
    mask_event_type = (df['event_type'].notna()) & (df['event_type'] != "") & (df['event_type'] != "Nan")

    # APLICAMOS TODOS LOS FILTROS
    df_clean = df[mask_date & mask_hour & mask_events & mask_borough & mask_event_type].copy()

    # 4. AGREGACIÓN
    # Si hay múltiples eventos en la misma zona a la misma hora, los sumamos.
    if not df_clean.empty:
        # Función para consolidar tipos de eventos
        def _join_events(x):
            unique_events = x.unique()
            if len(unique_events) == 1:
                return unique_events[0]
            return "Mixed Events"

        df_clean = (
            df_clean.groupby(['date', 'hour', 'borough'], as_index=False)
            .agg({
                'n_events': 'sum',
                'event_type': _join_events
            })
            .sort_values(['date', 'hour', 'borough'])
            .reset_index(drop=True)
        )

    # 5. MAPPING DE TIPOS (Schema Match)
    df_clean['borough'] = df_clean['borough'].astype('category')
    df_clean['event_type'] = df_clean['event_type'].astype('category')
    
    type_mapping = {
        'hour': 'Int64',
        'n_events': 'Int64'
    }
    for col, dtype in type_mapping.items():
        df_clean[col] = df_clean[col].astype(dtype)

    return df_clean[EVENTOS_EXPECTED_COLUMNS]