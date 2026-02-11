# src/procesamiento/capa0/capa0.py
import pyarrow.parquet as pq
import pandas as pd
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config.settings import obtener_ruta

console = Console()
sns.set_theme(style="whitegrid")

# Configuración de rutas y épocas clave (Fase A)
BASE_DATA_DIR = obtener_ruta("data/raw")
TARGET_MONTHS = ["2020-01", "2020-04", "2023-01", "2024-01", "2024-12"]
SERVICES = ["yellow", "green", "fhvhv"]

def explorar_archivo_fase_a(filepath, servicio):
    try:
        parquet_file = pq.ParquetFile(filepath)
        cols = parquet_file.schema.names

        date_col = next((c for c in cols if 'pickup_datetime' in c.lower()), None)
        price_col = next((c for c in cols if any(x in c.lower() for x in ['total_amount', 'base_passenger_fare', 'fare_amount'])), None)
        dist_col = next((c for c in cols if 'distance' in c.lower() or 'miles' in c.lower()), None)
        
        columnas_necesarias = [c for c in [date_col, price_col, dist_col, 'PULocationID'] if c is not None]

        table = parquet_file.read_row_group(0, columns=columnas_necesarias)
        df = table.to_pandas()
        df = df.sample(frac=0.1, random_state=42) 

        df.rename(columns={date_col: 'pickup', price_col: 'price'}, inplace=True)
        df['pickup'] = pd.to_datetime(df['pickup'])

        stats = {
            "Servicio": servicio,
            "Mes_Año": filepath.stem.split('_')[-1],
            "Viajes_Est": len(df) * 10,
            "Precio_Medio": round(df['price'].mean(), 2),
            "Hora_Pico": f"{df['pickup'].dt.hour.mode()[0]}:00",
            "Num_Vars": len(cols)
        }
        return stats

    except Exception as e:
        console.print(f"[red]Error en {filepath.name}: {e}[/red]")
        return None

def main():
    console.print(Panel.fit("[bold cyan]FASE A: EXPLORACIÓN ABIERTA Y MUESTREO[/bold cyan]"))
    
    resumen_total = []

    for servicio in SERVICES:
        console.rule(f"[bold yellow]{servicio.upper()}[/bold yellow]")
        for mes in TARGET_MONTHS:
            filename = f"{servicio}_tripdata_{mes}.parquet"
            filepath = BASE_DATA_DIR / servicio / filename
            if filepath.exists():
                with console.status(f"[cyan]Analizando {filename}..."):
                    res = explorar_archivo_fase_a(filepath, servicio)
                    if res: resumen_total.append(res)

    # --- TABLA RICH PARA CONSOLA ---
    table = Table(title="Resultados Fase A: Análisis del Sistema NYC", header_style="bold magenta")
    for col in ["Servicio", "Mes/Año", "Viajes Est.", "Precio Medio", "Hora Pico", "Nº Vars"]:
        table.add_column(col)

    for r in resumen_total:
        table.add_row(r['Servicio'], r['Mes_Año'], f"{r['Viajes_Est']:,}", f"${r['Precio_Medio']}", r['Hora_Pico'], str(r['Num_Vars']))

    console.print(table)

    if resumen_total:
        df_final = pd.DataFrame(resumen_total)

        output_folder = obtener_ruta('outputs/procesamiento/')
        output_folder.mkdir(parents=True, exist_ok=True)

        csv_path = output_folder / "capa0_exploracion.csv"
        md_path = output_folder / "capa0_exploracion.md"
         
        # Guardar en CSV y Markdown
        df_final.to_csv(csv_path, index=False)
        df_final.to_markdown(md_path, index=False)
        
        console.print(f"\n[bold green]Archivos guardados en la carpeta '{output_folder.name}':[/bold green]")
        console.print(f"  - {csv_path}")
        console.print(f"  - {md_path}")

if __name__ == "__main__":
    main()