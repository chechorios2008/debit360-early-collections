from pyspark.sql import functions as F
from pyspark.sql.types import StringType, DateType


def build_base_dataframe(clientes, moras, tanque_movimiento, canales, gestiones, excedentes):
    """
    Evolución a GOLD: Une fuentes, estandariza llaves y aplica reglas de integridad.
    """
    
    # 1. ESTANDARIZACIÓN DE LLAVES (Antes del Join para evitar nulos por tipos distintos)
    def standardize_keys(df):
        return df.withColumn("num_doc", F.col("num_doc").cast(StringType())) \
                 .withColumn("obl17", F.col("obl17").cast(StringType())) \
                 .withColumn("f_analisis", F.col("f_analisis").cast(DateType()))

    clientes = standardize_keys(clientes)
    moras = standardize_keys(moras)
    tanque_movimiento = standardize_keys(tanque_movimiento)
    canales = standardize_keys(canales)
    gestiones = standardize_keys(gestiones)
    excedentes = standardize_keys(excedentes)

    # 2. FILTRO DE INTEGRIDAD (Hallazgo del EDA)
    # Eliminamos registros donde la obligación es nula para no dañar el ABT
    clientes = clientes.filter(F.col("obl17").isNotNull())

    # 3. UNIÓN MAESTRA
    join_keys = ["num_doc", "obl17", "f_analisis"]
    dfs_to_join = [moras, tanque_movimiento, canales, gestiones, excedentes]

    df = clientes
    for incoming_df in dfs_to_join:
        # Nota: Usamos left join para mantener el universo de clientes de la Gerencia
        df = df.join(incoming_df, on=join_keys, how="left")

    # 4. TRATAMIENTO DE NULOS POST-JOIN (Imputación por defecto)
    # Llenamos con 0 todas las métricas donde no hubo actividad
    fill_cols = [c for c in df.columns if c not in join_keys]
    df = df.fillna(0, subset=fill_cols)

    return df