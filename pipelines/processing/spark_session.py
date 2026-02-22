from pyspark.sql import SparkSession


def get_spark_session():
    spark = (
        SparkSession.builder
        .appName("cobranza_analytics_pipeline")
        .master("local[*]") 
        # Aumentamos memoria del driver para manejar el catálogo de 4k columnas
        .config("spark.driver.memory", "8g") 
        # Habilitamos la optimización nativa de Arrow
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
        # Evitamos que el driver muera al recolectar resultados
        .config("spark.driver.maxResultSize", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR") # Menos ruido en consola
    return spark