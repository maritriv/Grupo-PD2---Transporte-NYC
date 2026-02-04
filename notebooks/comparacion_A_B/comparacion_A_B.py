import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from pathlib import Path

# 1. CONFIGURACIÓN DE RUTAS
# Path(__file__).parent apunta directamente a la carpeta 'comparacion_A_B'
BASE_DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path(__file__).parent 
TARGET_MONTHS = ["2020-01", "2020-04", "2023-01", "2024-01", "2024-12"]
SERVICES = ["yellow", "green", "fhvhv"]

def cargar_datos_comparativa(servicio, mes):
    filepath = BASE_DATA_DIR / servicio / f"{servicio}_tripdata_{mes}.parquet"
    if not filepath.exists(): return None
    
    try:
        # Solo leemos distancia y precio para optimizar memoria [cite: 21]
        schema = pq.read_schema(filepath)
        cols = schema.names
        dist_col = next((c for c in cols if 'distance' in c.lower() or 'miles' in c.lower()), None)
        price_col = next((c for c in cols if any(x in c.lower() for x in ['total_amount', 'fare'])), None)
        
        # Muestreo preventivo para no saturar la RAM con FHVHV [cite: 41]
        frac = 0.05 if servicio == 'fhvhv' else 0.15
        table = pq.read_table(filepath, columns=[dist_col, price_col])
        indices = pd.Series(range(table.num_rows)).sample(frac=frac, random_state=42).values
        df = table.take(indices).to_pandas()
        
        df.rename(columns={dist_col: 'distancia', price_col: 'precio'}, inplace=True)
        df['servicio'] = servicio
        return df
    except Exception: return None

def main():
    print(f"Generando comparativas en: {OUTPUT_DIR}")

    for mes in TARGET_MONTHS:
        dfs_mes = []
        for s in SERVICES:
            df = cargar_datos_comparativa(s, mes)
            if df is not None: dfs_mes.append(df)
        
        if dfs_mes:
            df_total = pd.concat(dfs_mes)
            # Filtros básicos para una visualización profesional [cite: 41]
            df_total = df_total[(df_total['distancia'] > 0) & (df_total['distancia'] < 15)]
            df_total = df_total[(df_total['precio'] > 0) & (df_total['precio'] < 80)]

            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df_total, x='distancia', y='precio', hue='servicio', 
                            palette={'yellow': 'gold', 'green': 'green', 'fhvhv': 'blue'}, alpha=0.3)
            
            plt.title(f"Tensión de Precios Taxi vs FHVHV ({mes})")
            
            # GUARDADO: Se guarda directamente en la carpeta del script
            file_path = OUTPUT_DIR / f"comparativa_{mes}.png"
            plt.savefig(file_path)
            plt.close()
            print(f"✓ {file_path.name} generado.")

if __name__ == "__main__":
    main()