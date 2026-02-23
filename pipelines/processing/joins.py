from pyspark.sql import functions as F

def build_base_dataframe(clientes, moras, tanque_movimiento, canales, gestiones, excedentes):
    """
    Construcción de ABT (Gold): Join por llaves canónicas generadas en SILVER_V2:
      - num_doc_key
      - obl17_key
      - f_analisis
    """

    join_keys = ["num_doc_key", "obl17_key", "f_analisis"]

    # Driver: clientes (universo base)
    df = clientes.filter(F.col("obl17_key").isNotNull())

    # Lista de dataframes a unir
    dfs_to_join = [moras, tanque_movimiento, canales, gestiones, excedentes]

    for inc in dfs_to_join:
        # ✅ IMPORTANTÍSIMO: quitar columnas originales para evitar duplicados/ambigüedad
        cols_to_drop = [c for c in ["num_doc", "obl17"] if c in inc.columns]
        inc_clean = inc.drop(*cols_to_drop)

        df = df.join(inc_clean, on=join_keys, how="left")

    # ✅ Fillna SOLO para numéricas y evitando llaves/IDs
    numeric_types = {"double", "bigint", "int", "float"}
    exclude = set(join_keys + ["num_doc", "obl17", "num_doc_key", "obl17_key", "f_analisis"])
    numeric_cols = [c for c, t in df.dtypes if (t in numeric_types and c not in exclude)]

    if numeric_cols:
        df = df.fillna(0, subset=numeric_cols)

    return df