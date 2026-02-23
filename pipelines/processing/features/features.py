from pyspark.sql.functions import col

def create_features(df):
    # Ejemplo ratio digital.
    if "monto_app" in df.columns and "monto_total" in df.columns:
        df = df.withColumn(
            "ratio_digital",
            col("monto_app") / col("monto_total")
        )

    # Ejemplo indicador mora
    if "dias_mora" in df.columns:
        df = df.withColumn(
            "flag_mora_alta",
            (col("dias_mora") > 30).cast("int")
        )

    return df