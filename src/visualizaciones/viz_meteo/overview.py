# Este script utiliza el archivo df_daily para mostrar la evolución del clima a lo largo del tiempo, 
# permitiéndote identificar días de lluvia o nieve extrema que puedan haber afectado al transporte.
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Ruta al df_daily generado en la Capa 3
INPUT_PATH = Path("data/external/meteo/aggregated/df_daily/data.parquet")
# Nueva ruta de salida organizada
OUTPUT_DIR = Path("outputs/viz_meteo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_meteo_trends():
    if not INPUT_PATH.exists(): 
        print(f"❌ Error: No se encuentra {INPUT_PATH}. Ejecuta primero capa3_meteo.py")
        return

    df = pd.read_parquet(INPUT_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Eje 1: Temperatura (usando nombres de columna de capa3)
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Temperatura (ºC)', color='tab:red')
    ax1.plot(df['date'], df['temp_c_mean'], color='tab:red', label='Temp Media', alpha=0.6)
    ax1.fill_between(df['date'], df['temp_c_min'], df['temp_c_max'], color='tab:red', alpha=0.1)
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Eje 2: Precipitación
    ax2 = ax1.twinx()
    ax2.set_ylabel('Precipitación Total (mm)', color='tab:blue')
    ax2.bar(df['date'], df['precip_mm_sum'], color='tab:blue', alpha=0.3, label='Lluvia/Nieve')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title("Evolución Meteorológica NYC (Histórico Diario)", fontsize=14)
    fig.tight_layout()
    
    save_path = OUTPUT_DIR / "tendencias_clima_nyc.png"
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Gráfico de tendencias guardado en: {save_path}")

if __name__ == "__main__":
    plot_meteo_trends()