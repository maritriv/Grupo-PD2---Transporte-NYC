import pandas as pd

print("Leyendo parquet...")

df = pd.read_parquet("data/ml/dataset_completo.parquet")

print("Leído!")
print(df.shape)
print(df.head())
print(df.columns)
print("un dia")
df["date"] = pd.to_datetime(df["date"])
print(df[(df["date"] == pd.Timestamp("2023-01-07")) & (df["service_type"] == "fhvhv")])

