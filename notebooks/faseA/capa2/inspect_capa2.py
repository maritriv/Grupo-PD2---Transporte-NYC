from pyspark.sql import SparkSession

def get_spark(app_name="Inspect-Capa2"):
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def main():
    spark = get_spark()

    df = spark.read.parquet("data/processed/layer2_trips")

    print("\n=== SCHEMA ===")
    df.printSchema()

    print("\n=== HEAD (10 filas) ===")
    df.show(10, truncate=False)

    print("\n=== CONTEO TOTAL ===")
    print(df.count())

    spark.stop()


if __name__ == "__main__":
    main()