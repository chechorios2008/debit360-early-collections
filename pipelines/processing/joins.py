from pyspark.sql import functions as F
from pyspark.sql.types import StringType, DateType


def build_base_dataframe(clientes, moras, tanque_movimiento, canales, gestiones, excedentes):
    """
    Evolución a GOLD: Une fuentes, normaliza IDs (evita notación científica) 
    y limpia espacios para garantizar el match de llaves.
    """

    # 1. ESTANDARIZACIÓN ROBUSTA DE LLAVES
    def standardize_keys(df):
        # Cast a decimal(38,0) elimina la notación científica (e+16) 
        # Luego cast a String y Trim elimina espacios.
        df = df.withColumn("num_doc", F.trim(F.col("num_doc").cast("decimal(38,0)").cast(StringType()))) \
               .withColumn("obl17", F.trim(F.col("obl17").cast("decimal(38,0)").cast(StringType()))) \
               .withColumn("f_analisis", F.col("f_analisis").cast(DateType()))
        
        # Creamos una llave auxiliar de fecha como String para asegurar el Join en entornos mixtos
        return df.withColumn("f_analisis_key", F.col("f_analisis").cast(StringType()))

    print("🧹 Normalizando IDs y fechas para evitar nulos en el Join...")
    clientes = standardize_keys(clientes)
    moras = standardize_keys(moras)
    tanque_movimiento = standardize_keys(tanque_movimiento)
    canales = standardize_keys(canales)
    gestiones = standardize_keys(gestiones)
    excedentes = standardize_keys(excedentes)

    # 2. FILTRO DE INTEGRIDAD
    # Eliminamos nulos en la llave primaria del driver (clientes)
    clientes = clientes.filter(F.col("obl17").isNotNull())

    # 3. UNIÓN MAESTRA
    # Usamos la llave de fecha normalizada como String para mayor seguridad
    join_keys = ["num_doc", "obl17", "f_analisis_key"]
    
    # Lista de dataframes para iterar (excluyendo la fecha original de las tablas secundarias 
    # para evitar columnas duplicadas)
    dfs_to_join = [moras, tanque_movimiento, canales, gestiones, excedentes]

    df = clientes
    for incoming_df in dfs_to_join:
        # Seleccionamos las columnas quitando la fecha original (ya tenemos f_analisis_key)
        cols_to_keep = [c for c in incoming_df.columns if c != "f_analisis"]
        df = df.join(incoming_df.select(cols_to_keep), on=join_keys, how="left")

    # 4. TRATAMIENTO DE NULOS POST-JOIN
    # Ignoramos las llaves y la fecha original en la imputación de ceros
    exclude_from_fill = join_keys + ["f_analisis"]
    fill_cols = [c for c in df.columns if c not in exclude_from_fill]
    df = df.fillna(0, subset=fill_cols)

    # Limpieza final: removemos la llave auxiliar y nos quedamos con la fecha original
    df = df.drop("f_analisis_key")

    return df