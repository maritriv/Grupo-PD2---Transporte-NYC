# Este script consume el archivo df_hourly_pattern para mostrar cómo varía la temperatura en un
# "día promedio" en Nueva York, incluyendo la desviación estándar.
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Ruta al patrón horario de la Capa 3
INPUT_PATH = Path("data/external/meteo/aggregated/df_hourly_pattern/data.parquet")
OUTPUT_DIR = Path("outputs/viz_meteo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_hourly_weather():
    if not INPUT_PATH.exists():
        print(f"❌ Error: No se encuentra {INPUT_PATH}")
        return
    
    # Leemos y ordenamos por hora
    df = pd.read_parquet(INPUT_PATH).sort_values('hour')

    plt.figure(figsize=(10, 6))
    # Usamos la media y std dev calculadas en el procesamiento
    plt.errorbar(df['hour'], df['temp_c_mean'], yerr=df['temp_c_std'], 
                 fmt='-o', color='darkorange', ecolor='gray', capsize=4, label='Temperatura (ºC)')
    
    plt.title("Patrón Horario de Temperatura en NYC (Clima Típico)", fontsize=14)
    plt.xlabel("Hora del día")
    plt.ylabel("Temperatura (ºC)")
    plt.xticks(range(0, 24))
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    
    save_path = OUTPUT_DIR / "patron_horario_temp.png"
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Patrón horario generado en: {save_path}")

if __name__ == "__main__":
    plot_hourly_weather()