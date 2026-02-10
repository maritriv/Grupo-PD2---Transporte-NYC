#Utiliza los datos ya procesados en la Capa 3 para mostrar tendencias de lluvia, nieve y tipos de clima.
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Rutas de Capa 3
DAILY_PATH = Path("data/external/meteo/aggregated/df_daily/data.parquet")
CODE_PATH = Path("data/external/meteo/aggregated/df_weathercode_daily/data.parquet")
OUTPUT_DIR = Path("outputs/viz_meteo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_overview():
    df = pd.read_parquet(DAILY_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Gráfico de Temperatura y Precipitación
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(df['date'], df['temp_c_mean'], color='tab:red', label='Temp Media')
    ax1.fill_between(df['date'], df['temp_c_min'], df['temp_c_max'], color='tab:red', alpha=0.1)
    ax1.set_ylabel('Temperatura (ºC)', color='tab:red')
    
    ax2 = ax1.twinx()
    ax2.bar(df['date'], df['precip_mm_sum'], color='tab:blue', alpha=0.3)
    ax2.set_ylabel('Precipitación (mm)', color='tab:blue')
    
    plt.title("Evolución Climática NYC")
    plt.savefig(OUTPUT_DIR / "tendencias_clima.png")
    plt.close()

def plot_codes():
    if not CODE_PATH.exists(): return
    df = pd.read_parquet(CODE_PATH)
    dist = df.groupby('weather_code')['n_hours'].sum()
    
    plt.figure(figsize=(8, 8))
    dist.plot(kind='pie', autopct='%1.1f%%', cmap='tab20')
    plt.title("Distribución de Códigos de Tiempo (WMO)")
    plt.savefig(OUTPUT_DIR / "distribucion_codigos_wmo.png")
    plt.close()

if __name__ == "__main__":
    plot_overview()
    plot_codes()