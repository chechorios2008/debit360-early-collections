# pipelines/processing/clean.py

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def clean_dataset(df: DataFrame) -> DataFrame:
    """
    Limpieza mínima y segura para datasets en Spark.

    - Normaliza nombres de columnas a lower_case
    - Trim a strings
    - No altera llaves ni cambia granularidad
    """

    # 1) Normalizar nombres de columnas
    for c in df.columns:
        clean = c.strip().lower().replace(" ", "_").replace(".", "_")
        if clean != c:
            df = df.withColumnRenamed(c, clean)

    # 2) Trim a columnas string (evita espacios raros)
    for c, t in df.dtypes:
        if t == "string":
            df = df.withColumn(c, F.trim(F.col(c)))

    return df