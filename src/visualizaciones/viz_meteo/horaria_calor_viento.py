# Estos scripts analizan la "radiografía" del día a día en NYC, usando tanto los archivos mensuales como los
# patrones agregados.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Rutas actualizadas
INPUT_DIR = Path("data/external/meteo/standarized")
OUTPUT_DIR = Path("outputs/viz_meteo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generar_heatmaps(year, month):
    file_name = f"meteo_{year}-{int(month):02d}.parquet"
    path = INPUT_DIR / file_name
    if not path.exists(): return

    df = pd.read_parquet(path)
    df['day'] = pd.to_datetime(df['date']).dt.day_name()
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Visualización 1: Temperatura
    plt.figure(figsize=(12, 6))
    pivot_t = df.pivot_table(index='day', columns='hour', values='temp_c', aggfunc='mean').reindex(order)
    sns.heatmap(pivot_t, cmap="coolwarm")
    plt.title(f"Heatmap Temperatura: {year}-{month}")
    plt.savefig(OUTPUT_DIR / f"temp_heatmap_{year}_{month}.png")
    plt.close()

    # Visualización 2: Viento
    plt.figure(figsize=(12, 6))
    pivot_v = df.pivot_table(index='day', columns='hour', values='wind_kmh', aggfunc='mean').reindex(order)
    sns.heatmap(pivot_v, cmap="YlGnBu")
    plt.title(f"Heatmap Viento (km/h): {year}-{month}")
    plt.savefig(OUTPUT_DIR / f"viento_heatmap_{year}_{month}.png")
    plt.close()

if __name__ == "__main__":
    for y, m in [("2024", "01"), ("2024", "12")]:
        generar_heatmaps(y, m)