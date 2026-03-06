# config/test_spark.py
from config.spark_manager import SparkManager

if __name__ == "__main__":
    try:
        spark = SparkManager.get_session()
        df = spark.createDataFrame([{"hola": "mundo", "valor": 42}])
        df.show()
        print("¡Spark funciona perfectamente en tu dispositivo!")
        SparkManager.stop_session()
        
    except Exception as e:
        print(f"FAILED: {e}")