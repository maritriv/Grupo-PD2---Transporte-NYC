# src/procesamiento/capa2/capa2.py
# EJECUTAR : wget -P data/external/ https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

def get_spark():
    return (SparkSession.builder
        .appName("NYC_Standardization")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        # Evita conflictos de esquemas al leer múltiples archivos
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.sql.caseSensitive", "false")
        .getOrCreate())

def get_service_config(service):
    """Define el esquema exacto y el mapeo de columnas según el diccionario."""
    
    if service == "fhvhv":
        # Basado en HVFHS Trip Records 
        schema = StructType([
            StructField("hvfhs_license_num", StringType(), True),
            StructField("pickup_datetime", TimestampType(), True),
            StructField("dropoff_datetime", TimestampType(), True),
            StructField("PULocationID", LongType(), True),
            StructField("DOLocationID", LongType(), True),
            StructField("trip_miles", DoubleType(), True),
            StructField("total_amount", DoubleType(), True)
        ])
        rename_map = {} # Ya viene con pickup_datetime
        
    elif service == "yellow":
        # Basado en Yellow Taxi Trip Records 
        schema = StructType([
            StructField("VendorID", LongType(), True),
            StructField("tpep_pickup_datetime", TimestampType(), True),
            StructField("tpep_dropoff_datetime", TimestampType(), True),
            StructField("passenger_count", DoubleType(), True),
            StructField("trip_distance", DoubleType(), True),
            StructField("PULocationID", LongType(), True),
            StructField("DOLocationID", LongType(), True),
            StructField("fare_amount", DoubleType(), True),
            StructField("total_amount", DoubleType(), True)
        ])
        rename_map = {"tpep_pickup_datetime": "pickup_datetime", 
                      "tpep_dropoff_datetime": "dropoff_datetime"}
                      
    elif service == "green":
        # Basado en LPEP Trip Records 
        schema = StructType([
            StructField("VendorID", LongType(), True),
            StructField("lpep_pickup_datetime", TimestampType(), True),
            StructField("lpep_dropoff_datetime", TimestampType(), True),
            StructField("PULocationID", LongType(), True),
            StructField("DOLocationID", LongType(), True),
            StructField("passenger_count", DoubleType(), True),
            StructField("trip_distance", DoubleType(), True),
            StructField("fare_amount", DoubleType(), True),
            StructField("total_amount", DoubleType(), True)
        ])
        rename_map = {"lpep_pickup_datetime": "pickup_datetime", 
                      "lpep_dropoff_datetime": "dropoff_datetime"}
    
    return schema, rename_map

def process_and_save():
    spark = get_spark()
    base_raw = project_root / "data" / "raw"
    base_out = project_root / "data" / "standarized"

    for service in ["yellow", "green", "fhvhv"]:
        print(f"\n[INFO] === Procesando: {service.upper()} ===")
        schema, rename_map = get_service_config(service)
        
        input_path = base_raw / service
        files = list(input_path.glob("*.parquet"))
        
        for f in sorted(files):
            print(f"  > Leyendo: {f.name}")
            try:
                # 1. Leer con el esquema específico del servicio
                df = spark.read.schema(schema).parquet(str(f))
                
                # 2. Renombrar columnas conflictivas inmediatamente
                for old_col, new_col in rename_map.items():
                    df = df.withColumnRenamed(old_col, new_col)
                
                # 3. Transformaciones (usando nombres ya normalizados)
                df = df.withColumn("year", F.year("pickup_datetime")) \
                       .withColumn("month", F.month("pickup_datetime")) \
                       .withColumn("service_type", F.lit(service))
                
                # 4. Escritura independiente por servicio para evitar colisiones
                # Usamos una subcarpeta para cada servicio en la salida
                output_service_path = base_out / service
                df.write.mode("append") \
                    .partitionBy("year", "month") \
                    .parquet(str(output_service_path))
                
            except Exception as e:
                print(f"  [ERROR] en {f.name}: {str(e)[:100]}...")

    print("\n[SUCCESS] Datos estandarizados en data/standarized/")

if __name__ == "__main__":
    process_and_save()