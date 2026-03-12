import pandas as pd

print("Leyendo parquet...")

df = pd.read_parquet("data/aggregated/df_daily_service/service_type=fhvhv/part_14e6431566ef41cd9b600c444a78dc4d.parquet")

print("Leído!")
print(df.shape)
print(df.head())