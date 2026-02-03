import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Configuración de rutas
BASE_DATA_DIR = Path("data/raw")
# Ruta para guardar los resultados
OUTPUT_DIR = Path("notebooks/faseB/faseB_resultados")
SERVICES_TO_ANALYZE = ["yellow", "green", "fhvhv"]

def analizar_y_guardar(filepath, servicio):
    """
    Procesa un archivo parquet, identifica columnas automáticamente y 
    guarda las visualizaciones solo si no existen.
    """
    try:
        filename = filepath.stem
        path_heatmap = OUTPUT_DIR / servicio / f"{filename}_heatmap.png"
        path_prices = OUTPUT_DIR / servicio / f"{filename}_prices.png"

        # --- 1. COMPROBACIÓN DE EXISTENCIA ---
        # Si ya existen los dos gráficos, saltamos el archivo para ahorrar tiempo
        if path_heatmap.exists() and path_prices.exists():
            return

        # Solo leemos el archivo si falta algún gráfico
        df = pd.read_parquet(filepath)
        cols = df.columns.tolist()

        # --- 2. BÚSQUEDA ROBUSTA DE COLUMNAS ---
        date_col = next((c for c in cols if 'pickup_datetime' in c.lower()), None)
        dist_col = next((c for c in cols if 'distance' in c.lower() or 'miles' in c.lower()), None)
        price_col = next((c for c in cols if any(x in c.lower() for x in ['total_amount', 'passenger_fare', 'fare'])), None)

        if not date_col:
            print(f"Saltando {filename}: No se encontró columna de fecha.")
            return

        # Estandarización de nombres
        rename_dict = {date_col: 'pickup'}
        if dist_col: rename_dict[dist_col] = 'trip_distance'
        if price_col: rename_dict[price_col] = 'price'
        
        df.rename(columns=rename_dict, inplace=True)

        # --- 3. LIMPIEZA Y PREPARACIÓN ---
        df['pickup'] = pd.to_datetime(df['pickup'])
        df['hour'] = df['pickup'].dt.hour
        df['day'] = df['pickup'].dt.day_name()
        
        # Filtros de seguridad (solo viajes reales)
        if 'trip_distance' in df.columns and 'price' in df.columns:
            df = df[(df['trip_distance'] > 0) & (df['price'] > 0)].copy()

        # Crear subcarpeta para el servicio
        (OUTPUT_DIR / servicio).mkdir(parents=True, exist_ok=True)

        # --- 4. GENERACIÓN DE GRÁFICOS ---
        # Gráfico 1: Mapa de Calor (Demanda)
        if not path_heatmap.exists():
            plt.figure(figsize=(10, 6))
            order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            pivot = df.groupby(['day', 'hour']).size().unstack().reindex(order)
            sns.heatmap(pivot, cmap="YlOrRd")
            plt.title(f"Demanda: {filename}")
            plt.savefig(path_heatmap)
            plt.close()

        # Gráfico 2: Precio vs Distancia
        if not path_prices.exists() and 'trip_distance' in df.columns and 'price' in df.columns:
            plt.figure(figsize=(10, 6))
            sample_size = min(5000, len(df))
            sample = df.sample(sample_size)
            sns.scatterplot(data=sample, x='trip_distance', y='price', alpha=0.4)
            plt.title(f"Dispersión de Precios: {filename}")
            plt.xlim(0, 15)
            plt.ylim(0, 80)
            plt.savefig(path_prices)
            plt.close()

    except Exception as e:
        print(f"Error procesando {filepath}: {e}")

def main():
    print(f"Iniciando análisis de la Fase B (Detectar patrones y tensiones)...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for servicio in SERVICES_TO_ANALYZE:
        service_path = BASE_DATA_DIR / servicio
        if not service_path.exists(): continue

        files = sorted(list(service_path.glob("*.parquet")))
        if not files: continue

        print(f"\nProcesando {servicio.upper()} ({len(files)} archivos)...")
        for filepath in tqdm(files):
            analizar_y_guardar(filepath, servicio)

    print(f"\n¡Análisis completado! Resultados en: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()