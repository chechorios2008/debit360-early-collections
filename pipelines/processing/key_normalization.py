from pyspark.sql import functions as F
from pyspark.sql.types import StringType, DateType

def _obl17_to_key(col):
    """
    Convierte obl17 a un string entero SIN notación científica.
    - Si viene como string con 'E'/'e' -> lo trata como double y lo formatea.
    - Si viene como bigint/double -> lo formatea como entero sin decimales.
    """
    c = F.col(col)

    return (
        F.when(c.isNull(), None)
         # Si es string y trae notación científica
         .when(F.col(col).cast("string").rlike(r"[eE]"), F.format_string("%.0f", c.cast("double")))
         # Si no trae E, pero podría venir como "123.0"
         .otherwise(F.regexp_replace(F.trim(c.cast("string")), r"\.0$", ""))
    )

def add_key_columns(df):
    """
    Agrega columnas canónicas para join:
      - num_doc_key: string limpio
      - obl17_key: string entero sin E16
      - f_analisis: date
    """
    return (
        df
        .withColumn("num_doc_key", F.trim(F.col("num_doc").cast(StringType())))
        .withColumn("obl17_key", _obl17_to_key("obl17").cast(StringType()))
        .withColumn("f_analisis", F.col("f_analisis").cast(DateType()))
    )