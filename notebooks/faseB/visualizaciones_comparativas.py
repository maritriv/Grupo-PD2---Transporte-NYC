import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm

# Configuración (Sincronizada con Fase A)
BASE_DATA_DIR = Path("data/raw")
TARGET_MONTHS = ["2020-01", "2020-04", "2023-01", "2024-01", "2024-12"]
SERVICES = ["yellow", "green", "fhvhv"]
OUTPUT_DIR = Path("notebooks/faseB/visualizaciones_comparativas_resultados")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def cargar_datos_mes(servicio, mes):
    """Carga y estandariza una muestra de un servicio y mes específico."""
    filename = f"{servicio}_tripdata_{mes}.parquet"
    filepath = BASE_DATA_DIR / servicio / filename
    
    if not filepath.exists():
        return None
    
    try:
        # 1. Identificación robusta de columnas (como en Fase B)
        schema = pq.read_schema(filepath)
        cols = schema.names
        
        dist_col = next((c for c in cols if 'distance' in c.lower() or 'miles' in c.lower()), None)
        price_col = next((c for c in cols if any(x in c.lower() for x in ['total_amount', 'fare'])), None)
        
        if not dist_col or not price_col:
            return None

        # 2. Carga optimizada (Muestra del 5% para FHVHV, 20% para Taxis)
        frac = 0.05 if servicio == 'fhvhv' else 0.2
        table = pq.read_table(filepath, columns=[dist_col, price_col])
        indices = pd.Series(range(table.num_rows)).sample(frac=frac, random_state=42).values
        df = table.take(indices).to_pandas()
        
        df.rename(columns={dist_col: 'distancia', price_col: 'precio'}, inplace=True)
        df['servicio'] = servicio
        return df
    except Exception as e:
        print(f"Error en {filename}: {e}")
        return None

def generar_grafico_conjunto(mes):
    """Genera la visualización comparativa para un mes concreto."""
    print(f"\nProcesando comparativa para {mes}...")
    componentes = []
    
    for s in SERVICES:
        df_s = cargar_datos_mes(s, mes)
        if df_s is not None:
            componentes.append(df_s)
    
    if not componentes:
        return

    df_total = pd.concat(componentes)
    
    # Limpieza de valores extremos para visualización clara
    df_total = df_total[(df_total['distancia'] > 0) & (df_total['distancia'] < 15)]
    df_total = df_total[(df_total['precio'] > 0) & (df_total['precio'] < 80)]

    # Visualización
    plt.figure(figsize=(12, 7))
    colores = {'yellow': 'gold', 'green': 'forestgreen', 'fhvhv': 'royalblue'}
    
    sns.scatterplot(
        data=df_total, x='distancia', y='precio', hue='servicio', 
        style='servicio', alpha=0.3, palette=colores
    )

    plt.title(f"Tensión de Precios: Taxi vs Uber/Lyft (NYC {mes})", fontsize=14)
    plt.xlabel("Distancia (Millas)")
    plt.ylabel("Precio Total ($)")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path = OUTPUT_DIR / f"comparativa_{mes}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Guardado: {save_path}")

def main():
    print("Iniciando generación de visualizaciones sistemáticas...")
    for mes in TARGET_MONTHS:
        generar_grafico_conjunto(mes)
    print(f"\n¡Proceso completado! Revisa la carpeta: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()