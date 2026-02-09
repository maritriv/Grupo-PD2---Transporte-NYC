import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm

# Configuración de rutas
BASE_DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("notebooks/faseB/visualizaciones_individuales_resultados")
SERVICES_TO_ANALYZE = ["yellow", "green", "fhvhv"]

def analizar_y_guardar(filepath, servicio):
    """Procesa archivos masivos usando PyArrow para evitar bloqueos de memoria."""
    try:
        filename = filepath.stem
        path_heatmap = OUTPUT_DIR / servicio / f"{filename}_heatmap.png"
        path_prices = OUTPUT_DIR / servicio / f"{filename}_prices.png"

        if path_heatmap.exists() and path_prices.exists():
            return

        # --- 1. DETECCIÓN DE COLUMNAS SIN CARGAR DATOS ---
        schema = pq.read_schema(filepath)
        cols = schema.names
        
        date_col = next((c for c in cols if 'pickup_datetime' in c.lower()), None)
        dist_col = next((c for c in cols if 'distance' in c.lower() or 'miles' in c.lower()), None)
        price_col = next((c for c in cols if any(x in c.lower() for x in ['total_amount', 'passenger_fare', 'fare'])), None)

        if not date_col: return

        # --- 2. CARGA QUIRÚRGICA CON PYARROW ---
        # Leemos SOLO las columnas necesarias para ahorrar RAM
        columnas_a_leer = [date_col]
        if dist_col: columnas_a_leer.append(dist_col)
        if price_col: columnas_a_leer.append(price_col)

        # Si es FHVHV (muy pesado), leemos una muestra aleatoria del 20% directamente
        # Si es yellow/green, lo leemos todo
        table = pq.read_table(filepath, columns=columnas_a_leer)
        
        if servicio == 'fhvhv':
            # Muestreo a nivel de tabla (muy rápido) para no saturar Pandas
            indices = pd.Series(range(table.num_rows)).sample(frac=0.2, random_state=42).values
            table = table.take(indices)

        df = table.to_pandas()

        # --- 3. ESTANDARIZACIÓN Y LIMPIEZA ---
        rename_dict = {date_col: 'pickup'}
        if dist_col: rename_dict[dist_col] = 'trip_distance'
        if price_col: rename_dict[price_col] = 'price'
        df.rename(columns=rename_dict, inplace=True)

        df['pickup'] = pd.to_datetime(df['pickup'])
        df['hour'] = df['pickup'].dt.hour
        df['day'] = df['pickup'].dt.day_name()
        
        if 'trip_distance' in df.columns and 'price' in df.columns:
            df = df[(df['trip_distance'] > 0) & (df['price'] > 0)].copy()

        (OUTPUT_DIR / servicio).mkdir(parents=True, exist_ok=True)

        # --- 4. GENERACIÓN DE GRÁFICOS ---
        if not path_heatmap.exists():
            plt.figure(figsize=(10, 6))
            order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            pivot = df.groupby(['day', 'hour']).size().unstack().reindex(order)
            sns.heatmap(pivot, cmap="YlOrRd")
            plt.title(f"Demanda (Muestra): {filename}")
            plt.savefig(path_heatmap)
            plt.close()

        if not path_prices.exists() and 'trip_distance' in df.columns and 'price' in df.columns:
            plt.figure(figsize=(10, 6))
            sample = df.sample(min(5000, len(df)))
            sns.scatterplot(data=sample, x='trip_distance', y='price', alpha=0.4)
            plt.title(f"Dispersión de Precios: {filename}")
            plt.xlim(0, 15); plt.ylim(0, 80)
            plt.savefig(path_prices)
            plt.close()

    except Exception as e:
        print(f"Error procesando {filepath}: {e}")

def main():
    print("Iniciando análisis optimizado con PyArrow...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for servicio in SERVICES_TO_ANALYZE:
        service_path = BASE_DATA_DIR / servicio
        if not service_path.exists(): continue

        files = sorted(list(service_path.glob("*.parquet")))
        print(f"\nProcesando {servicio.upper()} ({len(files)} archivos)...")
        for filepath in tqdm(files):
            analizar_y_guardar(filepath, servicio)

if __name__ == "__main__":
    main()