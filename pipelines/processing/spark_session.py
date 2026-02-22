from pyspark.sql import SparkSession

def get_spark_session():

    spark = (  # noqa: F841
        SparkSession.builder
        .appName("cobranza_analytics_pipeline")
        .master("local[*]") 
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    return spark