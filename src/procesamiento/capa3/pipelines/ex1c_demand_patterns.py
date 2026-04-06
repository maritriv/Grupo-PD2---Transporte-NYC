import logging
from pathlib import Path
import pyspark.sql.functions as F
from pyspark.sql.window import Window

from config.spark_manager import SparkManager
from config.settings import obtener_ruta

logger = logging.getLogger("SparkManager")

def generar_patrones_demanda():
    logger.info("[bold cyan]Iniciando proceso Ejercicio 1c - Patrones de Demanda (Nivel Avanzado)...[/bold cyan]")
    spark = SparkManager.get_session()
    
    input_path = str(obtener_ruta("data/aggregated/ex1a/df_demand_zone_hour_day"))
    output_path = str(obtener_ruta("data/aggregated/ex1c/ex1c_demand_patterns"))
    
    try:
        df = spark.read.parquet(input_path)
        
        # Agrupar la demanda media histórica por Zona, Finde y Franja Horaria
        df_grouped = df.groupBy("pu_location_id", "is_weekend", "hour_block_3h") \
                       .agg(F.avg("target_n_trips").alias("avg_trips"))
        
        # Window Function para calcular percentiles POR ZONA
        # Así evaluamos si la demanda es alta para el estándar de ESE barrio
        window_zona = Window.partitionBy("pu_location_id")
        
        # usamos rank() o ntile() para dividir los registros de cada zona en 3 grupos (terciles)
        df_classified = df_grouped.withColumn(
            "tercil_local",
            F.ntile(3).over(window_zona.orderBy("avg_trips"))
        )
        
        # Asignamos las etiquetas basadas en el tercil local
        df_classified = df_classified.withColumn(
            "demand_level",
            F.when(F.col("tercil_local") == 3, "Alta")
             .when(F.col("tercil_local") == 2, "Media")
             .otherwise("Baja")
        ).drop("tercil_local")
        
        # Guardar resultados
        df_classified.write.mode("overwrite").parquet(output_path)
        logger.info(f"[bold green]Proceso completado. Datos guardados en {output_path}[/bold green]")
        
    except Exception as e:
        logger.error(f"[bold red]Error en el pipeline de patrones de demanda: {e}[/bold red]")
        raise e
    
    finally:
        SparkManager.stop_session()

if __name__ == "__main__":
    generar_patrones_demanda()