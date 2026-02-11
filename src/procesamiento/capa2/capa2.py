# src/procesamiento/capa2/capa2.py
# EJECUTAR : wget -P data/external/ https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pathlib import Path
import sys
import shutil
import re

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

def get_spark():
    return (SparkSession.builder
        .appName("NYC_Final_Check")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.caseSensitive", "false")
        .getOrCreate())

def get_standard_schema():
    return [
        ("VendorID", LongType()),
        ("pickup_datetime", TimestampType()),
        ("dropoff_datetime", TimestampType()),
        ("passenger_count", DoubleType()),
        ("trip_distance", DoubleType()),
        ("PULocationID", LongType()),
        ("DOLocationID", LongType()),
        ("fare_amount", DoubleType()),
        ("total_amount", DoubleType()),
        ("congestion_surcharge", DoubleType()),
        ("airport_fee", DoubleType())
    ]

def process_and_save():
    spark = get_spark()
    base_raw = project_root / "data" / "raw"
    base_out = project_root / "data" / "standarized"

    column_maps = {
        "yellow": {"tpep_pickup_datetime": "pickup_datetime", "tpep_dropoff_datetime": "dropoff_datetime"},
        "green": {"lpep_pickup_datetime": "pickup_datetime", "lpep_dropoff_datetime": "dropoff_datetime"},
        "fhvhv": {"trip_miles": "trip_distance", "base_passenger_fare": "fare_amount"}
    }

    for service in ["yellow", "green", "fhvhv"]:
        print(f"\n[INFO] === Servicio: {service.upper()} ===")
        input_path = base_raw / service
        output_dir = base_out / service
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = list(input_path.glob("*.parquet"))
        
        for f in sorted(files):
            # 1. EXTRAER AÑO Y MES DEL NOMBRE ORIGINAL
            # Busca algo como 2020-01 en 'yellow_tripdata_2020-01.parquet'
            match = re.search(r"(\d{4})-(\d{2})", f.name)
            if not match: continue
            
            year_month = match.group(0) # Resultado: "2020-01"
            
            # 2. CONSTRUIR EL NOMBRE QUE QUIERES (COMO EN LA IMAGEN)
            # Resultado: "yellow_2020-01.parquet"
            target_name = f"{service}_{year_month}.parquet"
            target_file = output_dir / target_name
            
            # 3. COMPROBACIÓN REAL
            if target_file.exists():
                print(f"  [SKIP] Saltando: {target_name} (Ya existe en destino)")
                continue # AHORA SÍ SALTARÁ PORQUE EL NOMBRE COINCIDE

            print(f"  [PROC] Estandarizando: {f.name} -> {target_name}")
            try:
                df_raw = spark.read.parquet(str(f))
                
                # Renombrar
                mapping = column_maps.get(service, {})
                for old_name, new_name in mapping.items():
                    if old_name in df_raw.columns:
                        df_raw = df_raw.withColumnRenamed(old_name, new_name)

                # Cast y Schema
                final_cols = []
                for col_name, col_type in get_standard_schema():
                    if col_name in df_raw.columns:
                        final_cols.append(F.col(col_name).cast(col_type))
                    else:
                        final_cols.append(F.lit(None).cast(col_type).alias(col_name))
                
                # Guardar temporal
                temp_dir = output_dir / f"tmp_{service}_{year_month}"
                df_raw.select(*final_cols).coalesce(1).write.mode("overwrite").parquet(str(temp_dir))
                
                # Mover el archivo de datos real al destino con el nombre LIMPIO
                part_file = list(temp_dir.glob("part-*.parquet"))[0]
                shutil.move(str(part_file), str(target_file))
                
                # Limpiar basura de Spark
                shutil.rmtree(temp_dir)
                
            except Exception as e:
                print(f"  [ERROR] Fallo en {f.name}: {e}")

    print("\n[SUCCESS] Capa standarized sincronizada.")

if __name__ == "__main__":
    process_and_save()