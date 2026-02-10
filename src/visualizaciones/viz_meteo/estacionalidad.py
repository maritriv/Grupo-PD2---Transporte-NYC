#Este script genera boxplots para entender los extremos climáticos mensuales.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_DIR = Path("data/external/meteo/standarized")
OUTPUT_DIR = Path("outputs/viz_meteo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_seasonal():
    files = list(INPUT_DIR.glob("meteo_*.parquet"))
    if not files: return
    df = pd.concat([pd.read_parquet(f) for f in files])
    
    # Variabilidad por mes
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='month', y='temp_c', palette="vlag")
    plt.title("Variabilidad de Temperatura por Mes en NYC")
    plt.savefig(OUTPUT_DIR / "boxplot_mensual_temp.png")
    plt.close()

    # Diferencia Laborable vs Finde
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='is_weekend', y='precip_mm', estimator='mean')
    plt.xticks([0, 1], ['Laborable', 'Fin de Semana'])
    plt.title("Precipitación Media: Laborables vs Fines de Semana")
    plt.savefig(OUTPUT_DIR / "precip_laborable_vs_finde.png")
    plt.close()

if __name__ == "__main__":
    plot_seasonal()