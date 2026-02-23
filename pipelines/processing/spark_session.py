from pyspark.sql import SparkSession
import os


def get_spark_session(app_name="op_cobro"):
    # ✅ ruta con "/" para que Hadoop/JVM no dañe el path en Windows
    HADOOP_HOME = "C:/Users/serrios/hadoop"

    os.environ["HADOOP_HOME"] = HADOOP_HOME
    os.environ["hadoop.home.dir"] = HADOOP_HOME
    os.environ["PATH"] = f"{HADOOP_HOME}/bin" + os.pathsep + os.environ.get("PATH", "")

    
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[2]")  # menos concurrencia = menos memoria pico
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "6g")
        .config("spark.executor.memoryOverhead", "1g")

        # 🔥 Parquet (clave para tablas anchas)
        .config("spark.sql.parquet.columnarReaderBatchSize", "128")
        .config("spark.sql.parquet.enableVectorizedReader", "false")

        # (opcional) reduce particiones shuffle para no inflar memoria
        .config("spark.sql.shuffle.partitions", "16")

        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")
    return spark